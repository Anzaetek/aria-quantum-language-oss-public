// SPDX-License-Identifier: Apache-2.0
//! Resource governor — admission control for circuit execution.
//!
//! The deployment this protects is a **shared** box (a DGX / CUDA host several
//! people submit to). There, the failure mode is not "my job is slow", it is
//! "my job OOM-killed someone else's". The per-IP limiter in
//! [`crate::auth::rate_limit`] counts *requests*, and a single request can be
//! 256 GB — so it cannot help here.
//!
//! The leverage is that simulation cost is **knowable before allocating**: a
//! dense statevector is `2^n × 16` bytes and `n` is declared in the request. So
//! a job can be priced and refused in microseconds rather than discovered by
//! the OOM killer:
//!
//! | qubits | statevector | + adjoint |
//! |---|---|---|
//! | 28 | 4 GB | 8 GB |
//! | 30 | 16 GB | 32 GB |
//! | 32 | 64 GB | 128 GB |
//! | 34 | 256 GB | 512 GB |
//!
//! Two qubits is 4×, and there is no graceful degradation — it fits or the box
//! dies. Hence pre-flight rejection with the numbers in the message
//! ("needs 32 GB, 12 GB free" is actionable; an OOM-kill is not).
//!
//! Admission is denominated in **bytes, not job count**: a "max 4 concurrent
//! jobs" limit is meaningless when job cost spans six orders of magnitude.

use std::sync::{Arc, OnceLock};

use tokio::sync::{OwnedSemaphorePermit, Semaphore};

use crate::limits::{
    detect_cores, env_bytes, env_count, env_fraction, resolve_concurrency, resolve_memory, Limit,
    LimitError, Provenance, ResourceProfile,
};
use crate::topology::{classify, probe_machine, Topology, TopologyOverride};

/// Bytes per amplitude of a dense statevector (`Complex64` = 2×f64).
pub const BYTES_PER_AMPLITUDE: u64 = 16;

/// Widest dense register we will price rather than reject outright. `2^60`
/// amplitudes already exceeds any real machine; refusing above this keeps the
/// shift below from wrapping.
const MAX_PRICEABLE_QUBITS: u32 = 60;

/// Multiplier applied when a full statevector crosses the response boundary.
/// The amplitudes are re-encoded as `Vec<[f64;2]>` and then as a
/// `serde_json::Value` tree (~120 B per amplitude) before serialisation, so the
/// response dominates the state. Measured against `quantum_bridge.rs`'s
/// `exec_result_to_json`; conservative rather than exact, because being wrong
/// downward here is what kills the host.
const JSON_STATEVECTOR_FACTOR: u64 = 8;

/// How the backend represents the state, which is what decides the cost curve.
///
/// **Name the allocation, not the backend.** An earlier version of this priced
/// MPS by its tensors (`n·χ²·2·16`) because that is what "MPS" suggests. But
/// `MpsBackend::expectation` calls `execute(shots: None)`, which contracts to a
/// full statevector (`omega-backend-mps/src/mps.rs:418`) — so the real cost is
/// dense. A 34-qubit `Auto` circuit (which `resolve_backend` sends to MPS at
/// n ≥ 20) priced at 4 MiB and allocated 256 GiB. A governor that under-prices
/// is worse than none: `/health` advertises headroom and the refusals build
/// confidence, while the box dies anyway.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CostKind {
    /// Dense `2^n` amplitude vector (statevector, CPU or GPU).
    DenseStatevector,
    /// Matrix-product state. Cheap in shot mode; **dense** whenever the mode
    /// densifies, because the backend contracts to a statevector there.
    Mps { max_bond_dim: u32 },
    /// Stabilizer tableau: `O(n²)` bits in shot mode. Analytic mode allocates
    /// `2^n` probabilities (the backend caps that at n ≤ 24).
    Stabilizer,
    /// Fock-space photonics. Dimension is `C(m+p-1, p)` — combinatorial in the
    /// *photon* count, not the mode count, so a mode ceiling does not bound it.
    /// The default input is one photon in each of `ceil(m/2)` modes
    /// (`omega-backend-photonics/src/sim.rs:88`), which is enough to price it.
    Photonic,
    /// A backend whose memory profile this server cannot know — today, a
    /// dynamically-loaded plugin. Not priceable, so governed by the qubit
    /// ceiling alone. Stated plainly rather than dressed up as a priced
    /// admission.
    Opaque,
}

impl CostKind {
    /// Whether cost can be computed from the request shape at all.
    fn is_priceable(&self) -> bool {
        !matches!(self, CostKind::Opaque)
    }
}

/// Where a job executes, and therefore which memory it consumes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExecTarget {
    Cpu,
    /// Only constructed on a device-capable build (`--features opencl`, and
    /// CUDA/Metal when those land). The variant and its pool logic stay
    /// compiled in regardless so the behaviour is unit-tested everywhere, not
    /// only on hardware nobody has in CI.
    #[cfg_attr(not(feature = "opencl"), allow(dead_code))]
    Device(u32),
}

impl ExecTarget {
    /// Bytes per amplitude. The CPU backends are `Complex64` (16 B); the CUDA /
    /// Metal / OpenCL statevector kernels are f32, i.e. 8 B. Pricing device
    /// work at the CPU width refuses jobs that would comfortably fit.
    fn bytes_per_amplitude(&self) -> u64 {
        match self {
            ExecTarget::Cpu => BYTES_PER_AMPLITUDE,
            ExecTarget::Device(_) => BYTES_PER_AMPLITUDE / 2,
        }
    }
}

/// A budgeted memory pool.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PoolId {
    /// Host RAM on a machine whose device memory is separate.
    Host,
    /// One GPU's own memory.
    Device(u32),
    /// CPU and GPU share one physical pool (Apple Silicon, DGX Spark GB10).
    /// Both kinds of work debit this single budget — charging them separately
    /// would double-count memory that exists once.
    Unified,
}

impl PoolId {
    fn label(&self) -> String {
        match self {
            PoolId::Host => "host".into(),
            PoolId::Device(i) => format!("device{i}"),
            PoolId::Unified => "unified".into(),
        }
    }
}

/// The shape of a submitted job, as declared in the request.
#[derive(Clone, Copy, Debug)]
pub struct JobShape {
    pub num_qubits: u32,
    pub kind: CostKind,
    /// Number of circuits in the request (`circuits: [...]`).
    pub batch: usize,
    /// Whether an adjoint/gradient sweep runs, which keeps a second state
    /// resident alongside the forward one.
    pub gradient: bool,
    /// Whether the requested mode materialises a full `2^n` statevector.
    ///
    /// True for `shots: None` on execute, and for **every** expectation call —
    /// both `MpsBackend::expectation` and the stabilizer's analytic mode
    /// contract to a dense array regardless of how they store state internally.
    /// This flag is the difference between pricing the representation and
    /// pricing the allocation.
    pub densifies: bool,
    /// Where this job's memory comes from. Decides which pool is debited and
    /// the amplitude width: the GPU statevector backends are f32, so pricing
    /// device work at the CPU's 16 B/amplitude over-refuses by 2x.
    pub target: ExecTarget,
    /// Whether the *response* carries every amplitude.
    ///
    /// Distinct from `densifies`: `/expectation` materialises a 2^n array
    /// internally but returns scalars, whereas `/execute` with `shots: None`
    /// re-encodes the whole statevector as a `serde_json::Value` tree, which
    /// costs several times the state itself. Conflating the two over-prices
    /// every expectation call by ~8x and would refuse ordinary QML batches.
    pub returns_statevector: bool,
}

impl JobShape {
    pub fn new(num_qubits: u32, kind: CostKind) -> Self {
        Self {
            num_qubits,
            kind,
            batch: 1,
            gradient: false,
            densifies: true,
            returns_statevector: false,
            target: ExecTarget::Cpu,
        }
    }

    /// Run this job on a GPU device (f32 amplitudes, device pool).
    #[cfg_attr(not(feature = "opencl"), allow(dead_code))]
    pub fn on_device(mut self, index: u32) -> Self {
        self.target = ExecTarget::Device(index);
        self
    }

    /// `/execute` with `shots: None` — the full statevector crosses the wire.
    pub fn returning_statevector(mut self) -> Self {
        self.densifies = true;
        self.returns_statevector = true;
        self
    }

    /// Shot-sampled run: no full statevector is returned.
    pub fn with_shots(mut self) -> Self {
        self.densifies = false;
        self.returns_statevector = false;
        self
    }

    /// Whether this job's price is driven by `2^n`, and so should also obey the
    /// qubit ceiling.
    fn is_dense_driven(&self) -> bool {
        match self.kind {
            CostKind::DenseStatevector | CostKind::Opaque => true,
            CostKind::Mps { .. } | CostKind::Stabilizer => self.densifies,
            // Photonic width is combinatorial and priced exactly; the ceiling
            // is not the right instrument for it.
            CostKind::Photonic => false,
        }
    }
}

/// `2^n` amplitudes at a given element width.
fn dense_bytes_at(n: u32, bytes_per_amp: u64) -> Option<u64> {
    if n > MAX_PRICEABLE_QUBITS {
        return None;
    }
    (1u64 << n).checked_mul(bytes_per_amp)
}

/// Number of Fock basis states for `modes` modes holding `photons` photons:
/// `C(modes + photons - 1, photons)`.
fn fock_dim(modes: u64, photons: u64) -> Option<u64> {
    if modes == 0 {
        return Some(1);
    }
    let n = modes.checked_add(photons)?.checked_sub(1)?;
    let k = photons.min(n.checked_sub(photons)?);
    let mut acc: u64 = 1;
    for i in 0..k {
        // Exact: acc is always C(n, i) here, so the division has no remainder.
        acc = acc.checked_mul(n.checked_sub(i)?)? / (i + 1);
    }
    Some(acc)
}

/// Peak resident bytes this job needs, or `None` when it cannot be priced.
///
/// Peak is deliberately **not** multiplied by `batch`: the bridge executes a
/// batch as a sequential loop (`quantum_bridge::expectation_quantum_route`), so
/// rows reuse one allocation and the peak is the widest row, not their sum.
/// Multiplying would reject exactly the QML batches this path exists to serve.
/// (Correctness of that claim depends on rows not overlapping — verified
/// against the loop, where each `ExecResult` drops per iteration.)
pub fn estimate_peak_bytes(shape: &JobShape) -> Option<u64> {
    let n = shape.num_qubits;
    let width = shape.target.bytes_per_amplitude();
    let base = match shape.kind {
        CostKind::DenseStatevector => {
            let sv = dense_bytes_at(n, width)?;
            if shape.returns_statevector {
                // Re-encoded as a `Vec<[f64;2]>` and then as a
                // `serde_json::Value` tree (one Value per amplitude, ~120 B)
                // before serialising. That tree dwarfs the state itself, and
                // OOMs just as dead.
                sv.checked_mul(JSON_STATEVECTOR_FACTOR)?
            } else if shape.densifies {
                // Expectation: the state is resident but only scalars return.
                sv
            } else {
                // Shot sampling additionally builds `probs` and `cumulative`,
                // one f64 each per amplitude (omega-backend-statevector
                // src/sim.rs:895,898) — exactly one more statevector's worth.
                sv.checked_mul(2)?
            }
        }
        CostKind::Mps { max_bond_dim } => {
            let chi = max_bond_dim.max(1) as u64;
            let per_site = chi.checked_mul(chi)?.checked_mul(2)?;
            let tensors = per_site
                .checked_mul(BYTES_PER_AMPLITUDE)?
                .checked_mul(n.max(1) as u64)?;
            if shape.densifies {
                // `to_statevector()` allocates the full 2^n array on top of the
                // tensors, and `expectation()` always takes that path.
                let dense = dense_bytes_at(n, width)?;
                let dense = if shape.returns_statevector {
                    dense.checked_mul(JSON_STATEVECTOR_FACTOR)?
                } else {
                    dense
                };
                tensors.checked_add(dense)?
            } else {
                tensors
            }
        }
        CostKind::Stabilizer => {
            let n64 = n.max(1) as u64;
            let tableau = n64.checked_mul(n64)?.checked_mul(4)?.max(4096);
            if shape.densifies {
                // Analytic mode allocates 2^n f64 probabilities. The backend
                // refuses above 24 qubits itself; price what it would do.
                let probs = (1u64 << n.min(MAX_PRICEABLE_QUBITS)).checked_mul(8)?;
                tableau.checked_add(probs)?
            } else {
                tableau
            }
        }
        CostKind::Photonic => {
            // Default input: one photon in each of ceil(m/2) modes.
            let modes = n.max(1) as u64;
            let photons = modes.div_ceil(2);
            let states = fock_dim(modes, photons)?;
            // Each basis state is a Vec<u32> of length m plus its amplitude.
            let per_state = modes.checked_mul(4)?.checked_add(48)?;
            states.checked_mul(per_state)?
        }
        CostKind::Opaque => return None,
    };
    // An adjoint sweep holds the backward state alongside the forward one.
    if shape.gradient {
        base.checked_mul(2)
    } else {
        Some(base)
    }
}

/// Why a job was not admitted.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Rejection {
    /// Larger than the whole budget — no amount of waiting helps. → 413.
    ExceedsCapacity {
        required_bytes: u64,
        capacity_bytes: u64,
    },
    /// Would fit, but not right now alongside what is running. → 429.
    Busy {
        required_bytes: u64,
        available_bytes: u64,
    },
    /// Wider than the configured qubit ceiling. → 413.
    TooManyQubits { num_qubits: u32, max_qubits: u32 },
    /// Cost could not be determined and the governor refuses to gamble. → 413.
    Unpriceable { reason: &'static str },
    /// The ledger had room but the MACHINE does not. → 429.
    ///
    /// The ledger counts what the governor admitted; it cannot see a job's
    /// classical side (dataset, optimizer state, autograd tape) or anything
    /// else sharing the box. When the two disagree, the machine wins.
    MachinePressure {
        required_bytes: u64,
        measured_available_bytes: u64,
    },
}

impl Rejection {
    /// Operator-facing message. Always states the numbers: a caller can only
    /// act on "needs 32 GB, 12 GB free", never on "internal error".
    pub fn message(&self) -> String {
        let gb = |b: u64| format!("{:.2} GB", b as f64 / (1u64 << 30) as f64);
        match self {
            Rejection::ExceedsCapacity {
                required_bytes,
                capacity_bytes,
            } => format!(
                "circuit needs {} but this host's execution budget is {} — it cannot run here \
                 regardless of load. Reduce qubits (each one doubles the cost), use the MPS \
                 backend with a bond-dimension cap, or raise OMEGA_MAX_MEM if the host really \
                 has room.",
                gb(*required_bytes),
                gb(*capacity_bytes)
            ),
            Rejection::Busy {
                required_bytes,
                available_bytes,
            } => format!(
                "circuit needs {} but only {} of the execution budget is free right now; \
                 other jobs hold the rest. Retry shortly.",
                gb(*required_bytes),
                gb(*available_bytes)
            ),
            Rejection::TooManyQubits {
                num_qubits,
                max_qubits,
            } => format!(
                "circuit declares {num_qubits} qubits, above this host's ceiling of \
                 {max_qubits} (OMEGA_MAX_QUBITS)."
            ),
            Rejection::MachinePressure {
                required_bytes,
                measured_available_bytes,
            } => format!(
                "circuit needs {} and the execution budget has room, but the MACHINE only has \
                 {} actually available — the budget cannot see memory it did not hand out \
                 (a job's dataset, optimizer state, or other processes on this host). \
                 Refusing rather than trusting the ledger. Retry shortly.",
                gb(*required_bytes),
                gb(*measured_available_bytes)
            ),
            Rejection::Unpriceable { reason } => format!(
                "refusing to admit a job whose resource cost cannot be determined ({reason}). \
                 The governor fails closed rather than risk the host."
            ),
        }
    }

    /// `true` when waiting could help — the caller should retry.
    pub fn is_transient(&self) -> bool {
        matches!(
            self,
            Rejection::Busy { .. } | Rejection::MachinePressure { .. }
        )
    }
}

/// Governor configuration, resolved once at boot.
#[derive(Clone, Debug)]
pub struct GovernorConfig {
    /// Total bytes of concurrent execution allowed across all jobs.
    pub capacity_bytes: u64,
    /// Hard ceiling on declared qubits, applied before pricing so an absurd
    /// request is cheap to reject.
    pub max_qubits: u32,
    /// Cap on simultaneous jobs, independent of their size. Bytes bound total
    /// memory; this bounds CPU contention, which many small jobs can cause
    /// without ever approaching the memory budget.
    pub max_concurrency: u32,
    /// Fixed value for the machine-pressure probe instead of asking the OS.
    ///
    /// Admission consults live free memory, which would otherwise make every
    /// test depend on whatever else is running on the box — the same
    /// non-determinism that makes a resource guard hard to trust. Tests inject
    /// a known value; production leaves this `None` and measures.
    pub machine_available_override: Option<u64>,
    pub capacity_from: Provenance,
    pub qubits_from: Provenance,
    pub concurrency_from: Provenance,
}

/// Conservative fallback when RAM cannot be detected. Deliberately small: an
/// unnecessary rejection is recoverable, an OOM-killed neighbour is not.
const FALLBACK_CAPACITY_BYTES: u64 = 4 << 30;

impl Default for GovernorConfig {
    fn default() -> Self {
        Self {
            capacity_bytes: FALLBACK_CAPACITY_BYTES,
            max_qubits: 30,
            max_concurrency: 8,
            // Tests build configs from `..default()`; a huge value keeps the
            // machine probe out of their way so they exercise the LEDGER,
            // which is what they are about. The pressure check has its own
            // tests with injected values.
            machine_available_override: Some(u64::MAX / 4),
            capacity_from: Provenance::Default,
            qubits_from: Provenance::Default,
            concurrency_from: Provenance::Default,
        }
    }
}

impl GovernorConfig {
    /// Resolve from the environment.
    ///
    /// | variable | meaning |
    /// |---|---|
    /// | `OMEGA_MAX_MEM` | absolute budget, human units accepted (`48G`, `8GiB`) |
    /// | `OMEGA_MEM_FRACTION` | share of detected memory (`0.25`, `25%`) |
    /// | `OMEGA_MAX_CONCURRENCY` | absolute cap on simultaneous jobs |
    /// | `OMEGA_CPU_FRACTION` | share of cores, converted to a job cap |
    /// | `OMEGA_RESOURCE_PROFILE` | `gentle` / `balanced` / `greedy` preset |
    /// | `OMEGA_MAX_QUBITS` | hard width ceiling |
    ///
    /// **A malformed value is fatal**, not a silent fallback: `OMEGA_MAX_MEM=48G`
    /// used to parse as garbage and quietly leave a 4 GiB budget, which looks
    /// exactly like a throttle that worked.
    pub fn from_env() -> Self {
        match Self::try_from_env() {
            Ok(cfg) => cfg,
            Err(e) => {
                eprintln!("[governor] {e}");
                eprintln!("[governor] refusing to start with a resource limit it cannot honour");
                std::process::exit(2);
            }
        }
    }

    /// Fallible form, so the parsing rules are testable without exiting.
    pub fn try_from_env() -> Result<Self, LimitError> {
        let profile = match std::env::var("OMEGA_RESOURCE_PROFILE") {
            Ok(raw) if !raw.trim().is_empty() => match ResourceProfile::parse(&raw) {
                Some(p) => Some(p),
                None => {
                    return Err(LimitError {
                        var: "OMEGA_RESOURCE_PROFILE".into(),
                        value: raw,
                        expected: "one of: gentle, balanced, greedy",
                    })
                }
            },
            _ => None,
        };

        let memory = resolve_memory(
            env_bytes("OMEGA_MAX_MEM")?,
            env_fraction("OMEGA_MEM_FRACTION")?,
            profile,
            detect_available_memory_bytes(),
            FALLBACK_CAPACITY_BYTES,
        );

        let concurrency = resolve_concurrency(
            env_count("OMEGA_MAX_CONCURRENCY")?,
            env_fraction("OMEGA_CPU_FRACTION")?,
            profile,
            detect_cores(),
        );

        let qubits = match env_count("OMEGA_MAX_QUBITS")? {
            Some(l) => l,
            None => Limit::new(default_qubit_ceiling(memory.value), Provenance::Detected),
        };

        Ok(Self {
            // Production measures; only tests inject.
            machine_available_override: None,
            capacity_bytes: memory.value,
            max_qubits: qubits.value,
            max_concurrency: concurrency.value,
            capacity_from: memory.from,
            qubits_from: qubits.from,
            concurrency_from: concurrency.from,
        })
    }
}

/// Widest register whose forward+adjoint pair still fits `capacity_bytes`.
///
/// A register of `n` qubits with an adjoint copy costs `2 · 2^n · 16 =
/// 2^(n+1) · 16` bytes. The loop therefore prices `n + 1` *before* stepping to
/// it — pricing `n` and then incrementing returns a ceiling one qubit too wide,
/// which advertises a limit the capacity check would then refuse.
fn default_qubit_ceiling(capacity_bytes: u64) -> u32 {
    let mut n = 1u32;
    while n < MAX_PRICEABLE_QUBITS {
        let need_for_next = (1u64 << (n + 2)).saturating_mul(BYTES_PER_AMPLITUDE);
        if need_for_next > capacity_bytes {
            break;
        }
        n += 1;
    }
    n
}

/// Memory actually available to this process.
///
/// **The cgroup limit wins when there is one.** A server in a container on a
/// 1 TiB DGX with a 16 GiB limit must budget against 16 GiB: budgeting against
/// host RAM would compute a 512 GiB allowance, admit a 400 GiB job, and hand it
/// to the cgroup OOM-killer — the exact outcome this module exists to prevent,
/// arrived at with more confidence. Containers are the normal DGX deployment,
/// so this is the common case, not the exotic one.
fn detect_available_memory_bytes() -> Option<u64> {
    if let Some(limited) = detect_cgroup_limit_bytes() {
        return Some(limited);
    }
    detect_total_memory_bytes()
}

/// cgroup v2 (`memory.max`) then v1 (`memory.limit_in_bytes`).
///
/// Both report an effectively-unlimited sentinel when unconstrained — v2 the
/// literal string `max`, v1 a huge number — which must be treated as "no
/// limit" rather than as a gigantic budget.
fn detect_cgroup_limit_bytes() -> Option<u64> {
    const UNLIMITED_FLOOR: u64 = 1 << 50; // 1 PiB: no real container cap is this big.
    for path in [
        "/sys/fs/cgroup/memory.max",                   // v2, unified
        "/sys/fs/cgroup/memory/memory.limit_in_bytes", // v1
    ] {
        let Ok(raw) = std::fs::read_to_string(path) else {
            continue;
        };
        let raw = raw.trim();
        if raw == "max" {
            continue;
        }
        if let Ok(v) = raw.parse::<u64>() {
            if v > 0 && v < UNLIMITED_FLOOR {
                return Some(v);
            }
        }
    }
    None
}

/// Total physical RAM, when the platform makes it cheap to ask.
fn detect_total_memory_bytes() -> Option<u64> {
    #[cfg(target_os = "linux")]
    {
        let meminfo = std::fs::read_to_string("/proc/meminfo").ok()?;
        for line in meminfo.lines() {
            if let Some(rest) = line.strip_prefix("MemTotal:") {
                let kb: u64 = rest.trim().trim_end_matches(" kB").trim().parse().ok()?;
                return kb.checked_mul(1024);
            }
        }
        None
    }
    #[cfg(target_os = "macos")]
    {
        // Boot-time only, so the one subprocess is acceptable; std has no
        // sysctl binding and the crate takes no extra dependency for this.
        let out = std::process::Command::new("sysctl")
            .args(["-n", "hw.memsize"])
            .output()
            .ok()?;
        String::from_utf8_lossy(&out.stdout).trim().parse().ok()
    }
    #[cfg(not(any(target_os = "linux", target_os = "macos")))]
    {
        None
    }
}

/// Fraction of *measured* free memory a single job may claim.
///
/// Not 1.0: the measurement is a snapshot, the job's own classical side lands
/// after admission, and something else on the box may allocate meanwhile. The
/// margin is what makes the check a guard rather than a race.
const MACHINE_HEADROOM_FRACTION: f64 = 0.8;

/// Would admitting a job of `required` bytes over-commit the machine, given
/// what the OS says is actually free?
///
/// Pure so the policy is testable without a loaded box. `None` measurement
/// means the platform would not say — fall back to the ledger rather than
/// invent a number, and say so rather than silently trusting it.
pub fn exceeds_machine_headroom(required: u64, measured_available: Option<u64>) -> bool {
    match measured_available {
        None => false,
        Some(avail) => (required as f64) > (avail as f64) * MACHINE_HEADROOM_FRACTION,
    }
}

/// Admission controller. Permits are denominated in MiB so the counts fit the
/// semaphore's `u32` while still resolving every job worth distinguishing.
pub struct Governor {
    config: GovernorConfig,
    /// One budget per physical memory pool. On a unified machine there is
    /// exactly one entry and *both* CPU and GPU work debit it — which is the
    /// entire point of distinguishing topologies.
    pools: Vec<Pool>,
    /// Job-count cap, orthogonal to bytes.
    slots: Arc<Semaphore>,
}

struct Pool {
    id: PoolId,
    capacity_bytes: u64,
    budget_mib: u32,
    slots: Arc<Semaphore>,
}

impl Pool {
    fn new(id: PoolId, capacity_bytes: u64) -> Self {
        let budget_mib = to_mib(capacity_bytes);
        Self {
            id,
            capacity_bytes,
            budget_mib,
            slots: Arc::new(Semaphore::new(budget_mib as usize)),
        }
    }

    fn available_bytes(&self) -> u64 {
        (self.slots.available_permits() as u64).saturating_mul(1 << 20)
    }
}

/// Bytes rounded up to whole MiB, floored at 1 so every admitted job consumes
/// something (a zero-weight job could otherwise be admitted without limit).
fn to_mib(bytes: u64) -> u32 {
    const MIB: u64 = 1 << 20;
    bytes.div_ceil(MIB).clamp(1, u32::MAX as u64) as u32
}

/// Held for the lifetime of an execution; releases the reservation on drop.
#[derive(Debug)]
pub struct Reservation {
    _permit: OwnedSemaphorePermit,
    /// Held alongside the memory permit: bytes bound total memory, this bounds
    /// CPU contention. Many small jobs can saturate the cores without ever
    /// approaching the memory budget.
    _slot: Option<OwnedSemaphorePermit>,
}

impl Governor {
    /// Single host pool sized by `config` — the CPU-only shape. Production
    /// builds go through [`Governor::from_env`]; this is the direct constructor
    /// used by tests and by embedders that already know their topology.
    #[allow(dead_code)]
    pub fn new(config: GovernorConfig) -> Self {
        Self {
            pools: vec![Pool::new(PoolId::Host, config.capacity_bytes)],
            slots: Arc::new(Semaphore::new(config.max_concurrency.max(1) as usize)),
            config,
        }
    }

    /// Build pools from a detected [`Topology`].
    ///
    /// `config.capacity_bytes` remains the ceiling on *host* capacity, so an
    /// operator's `OMEGA_MAX_MEM` still binds; device pools are sized by the
    /// hardware. On a unified machine there is one pool and one budget, never
    /// host + device, because that memory exists once.
    pub fn with_topology(config: GovernorConfig, topology: &Topology) -> Self {
        let pools = match topology {
            Topology::HostOnly { host_bytes } => vec![Pool::new(
                PoolId::Host,
                config.capacity_bytes.min(*host_bytes).max(1),
            )],
            Topology::Unified { total_bytes } => vec![Pool::new(
                PoolId::Unified,
                config.capacity_bytes.min(*total_bytes).max(1),
            )],
            Topology::Discrete {
                host_bytes,
                devices,
            } => {
                let mut pools = vec![Pool::new(
                    PoolId::Host,
                    config.capacity_bytes.min(*host_bytes).max(1),
                )];
                for (idx, bytes) in devices {
                    pools.push(Pool::new(PoolId::Device(*idx), (*bytes).max(1)));
                }
                pools
            }
        };
        Self {
            slots: Arc::new(Semaphore::new(config.max_concurrency.max(1) as usize)),
            config,
            pools,
        }
    }

    pub fn from_env() -> Self {
        let forced = std::env::var("OMEGA_MEM_TOPOLOGY")
            .ok()
            .and_then(|v| TopologyOverride::parse(&v));
        let classified = classify(&probe_machine(), forced);
        Self::with_topology(GovernorConfig::from_env(), &classified.topology)
    }

    pub fn config(&self) -> GovernorConfig {
        self.config.clone()
    }

    /// Which pool a job's memory comes from.
    ///
    /// On a unified machine every target resolves to the one shared pool — the
    /// single most important line in this module, since routing device work to
    /// a separate budget there is what double-counts and kills the box.
    fn pool_for(&self, target: ExecTarget) -> &Pool {
        if let Some(p) = self.pools.iter().find(|p| p.id == PoolId::Unified) {
            return p;
        }
        let wanted = match target {
            ExecTarget::Cpu => PoolId::Host,
            ExecTarget::Device(i) => PoolId::Device(i),
        };
        self.pools
            .iter()
            .find(|p| p.id == wanted)
            // A device we have no pool for must not silently fall back to the
            // host budget — that is how a 64 GB job reaches a 24 GB card. The
            // caller checks `has_pool_for` and refuses instead; this arm only
            // guards against a target that passed that check.
            .unwrap_or(&self.pools[0])
    }

    /// Whether this machine has a budget for `target` at all.
    fn has_pool_for(&self, target: ExecTarget) -> bool {
        if self.pools.iter().any(|p| p.id == PoolId::Unified) {
            return true;
        }
        let wanted = match target {
            ExecTarget::Cpu => PoolId::Host,
            ExecTarget::Device(i) => PoolId::Device(i),
        };
        self.pools.iter().any(|p| p.id == wanted)
    }

    /// Bytes free in the host (or unified) budget.
    pub fn available_bytes(&self) -> u64 {
        self.pool_for(ExecTarget::Cpu).available_bytes()
    }

    /// Price `shape` and reserve for it, or say precisely why not.
    ///
    /// Non-blocking on purpose: a caller that queues instead of failing turns
    /// an admission problem into an unbounded-queue problem. Refusing with
    /// `Retry-After` keeps the backpressure visible to the client.
    pub fn admit(&self, shape: &JobShape) -> Result<Reservation, Rejection> {
        // Fail closed: a device with no budget must never fall back to the host
        // pool. That fallback is exactly how a 64 GB job reaches a 24 GB card.
        if !self.has_pool_for(shape.target) {
            return Err(Rejection::Unpriceable {
                reason: "no memory budget for the requested device",
            });
        }
        let pool = self.pool_for(shape.target);
        let priced = estimate_peak_bytes(shape);

        // Most informative refusal first. "needs 16384.00 GB, budget 24.00 GB"
        // tells a caller exactly what to change; "above the ceiling of 28"
        // makes them go looking for why 28. Both are correct, so lead with the
        // one that carries the numbers.
        if let Some(required) = priced {
            if required > pool.capacity_bytes {
                return Err(Rejection::ExceedsCapacity {
                    required_bytes: required,
                    capacity_bytes: pool.capacity_bytes,
                });
            }
        }

        // The ledger says it fits. Now ask the MACHINE, which can disagree:
        // the governor never sees a job's classical side (dataset, optimizer
        // state, autograd tape) nor anything else sharing the box. A TLA+ check
        // of the ledger-only policy found exactly this — two QML rows priced at
        // 8 GB against a 64 GB budget, reported as 56 GB of headroom, while the
        // machine held 68 GB. Admission was satisfied and the box was over.
        if let Some(required) = priced {
            let measured = self
                .config
                .machine_available_override
                .or_else(crate::topology::detect_available_memory_now);
            if exceeds_machine_headroom(required, measured) {
                return Err(Rejection::MachinePressure {
                    required_bytes: required,
                    measured_available_bytes: measured.unwrap_or(0),
                });
            }
        }

        // The qubit ceiling is DENSE reasoning — it exists because `2^n`
        // explodes. Applying it to every backend would refuse a 30-qubit
        // Clifford circuit the stabilizer runs in kilobytes, or a wide MPS
        // circuit bounded by its bond dimension. So enforce it only where
        // width actually drives cost: the dense path, and the unpriceable
        // backends where it is the sole guard available.
        let ceiling_applies = shape.is_dense_driven();
        if ceiling_applies && shape.num_qubits > self.config.max_qubits {
            return Err(Rejection::TooManyQubits {
                num_qubits: shape.num_qubits,
                max_qubits: self.config.max_qubits,
            });
        }

        // Memory first (it is the refusal with the actionable numbers), then
        // the job-slot cap.
        let reservation = match priced {
            Some(required) => Self::reserve_in(pool, required, to_mib(required)),
            // Photonics is dimensioned by photon number, and a plugin's
            // profile is opaque; neither can be priced from the request. The
            // ceiling above is the only bound available, so admit rather than
            // break a path that works today — and say so out loud instead of
            // pretending the job was priced.
            None if !shape.kind.is_priceable() => Self::reserve_in(pool, 0, 1),
            None => Err(Rejection::Unpriceable {
                reason: "register too wide to represent",
            }),
        };
        let mut reservation = reservation?;
        match Arc::clone(&self.slots).try_acquire_owned() {
            Ok(slot) => reservation._slot = Some(slot),
            Err(_) => {
                return Err(Rejection::Busy {
                    required_bytes: priced.unwrap_or(0),
                    available_bytes: pool.available_bytes(),
                })
            }
        }
        Ok(reservation)
    }

    fn reserve_in(pool: &Pool, required: u64, mib: u32) -> Result<Reservation, Rejection> {
        let mib = mib.clamp(1, pool.budget_mib);
        match Arc::clone(&pool.slots).try_acquire_many_owned(mib) {
            Ok(permit) => Ok(Reservation {
                _permit: permit,
                _slot: None,
            }),
            Err(_) => Err(Rejection::Busy {
                required_bytes: required,
                available_bytes: pool.available_bytes(),
            }),
        }
    }
}

/// Process-wide governor, resolved from the environment on first use.
///
/// Deliberately not part of `AppState`: admission must not queue behind the
/// state lock, and the budget is fixed at boot anyway.
static GOVERNOR: OnceLock<Governor> = OnceLock::new();

/// The process-wide governor.
pub fn governor() -> &'static Governor {
    GOVERNOR.get_or_init(Governor::from_env)
}

impl Governor {
    /// Snapshot for `/health`, so a caller can size work *before* submitting
    /// instead of discovering the ceiling by being rejected.
    pub fn health_snapshot(&self) -> serde_json::Value {
        // Per pool, so a caller can see WHICH resource is scarce rather than
        // one aggregate that hides a full GPU behind an idle host.
        let pools: Vec<serde_json::Value> = self
            .pools
            .iter()
            .map(|p| {
                serde_json::json!({
                    "pool": p.id.label(),
                    "capacity_bytes": p.capacity_bytes,
                    "available_bytes": p.available_bytes(),
                })
            })
            .collect();
        let primary = self.pool_for(ExecTarget::Cpu);
        serde_json::json!({
            // Kept for compatibility with clients written against the
            // single-budget shape; `pools` is the complete picture.
            "capacity_bytes": primary.capacity_bytes,
            "available_bytes": primary.available_bytes(),
            "max_qubits": self.config.max_qubits,
            "max_concurrency": self.config.max_concurrency,
            "jobs_running": self.config.max_concurrency as usize
                - self.slots.available_permits().min(self.config.max_concurrency as usize),
            "unified": self.pools.iter().any(|p| p.id == PoolId::Unified),
            "pools": pools,
            // Where each limit came from, so "why was my job refused?" is
            // answerable without reading the server's source.
            "limits_from": {
                "capacity_bytes": self.config.capacity_from.label(),
                "max_qubits": self.config.qubits_from.label(),
                "max_concurrency": self.config.concurrency_from.label(),
            },
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const GB: u64 = 1 << 30;

    /// The cost table from the module docs, pinned. If these move, the
    /// admission decisions everyone else depends on have moved too.
    #[test]
    fn dense_statevector_cost_matches_the_documented_table() {
        // The representation itself, as tabled in the module docs.
        let dense_bytes = |n| dense_bytes_at(n, BYTES_PER_AMPLITUDE);
        assert_eq!(dense_bytes(28), Some(4 * GB));
        assert_eq!(dense_bytes(30), Some(16 * GB));
        assert_eq!(dense_bytes(32), Some(64 * GB));
        assert_eq!(dense_bytes(34), Some(256 * GB));
        // Each extra qubit doubles: that is why this must be pre-flight.
        for n in 4..40 {
            assert_eq!(
                dense_bytes(n + 1).unwrap(),
                dense_bytes(n).unwrap() * 2,
                "qubit {n}->{}",
                n + 1
            );
        }
        // An expectation call holds the state but returns scalars.
        let expectation = JobShape::new(28, CostKind::DenseStatevector);
        assert_eq!(estimate_peak_bytes(&expectation), Some(4 * GB));
        // Shot sampling also builds probs + cumulative: one extra state.
        let shots = JobShape::new(28, CostKind::DenseStatevector).with_shots();
        assert_eq!(estimate_peak_bytes(&shots), Some(8 * GB));
        // Returning the statevector pays for the JSON tree on top.
        let returned = JobShape::new(28, CostKind::DenseStatevector).returning_statevector();
        assert_eq!(
            estimate_peak_bytes(&returned),
            Some(4 * GB * JSON_STATEVECTOR_FACTOR)
        );
    }

    /// THE regression that made the first version of this governor worse than
    /// nothing. MPS was priced by its tensors, but `MpsBackend::expectation`
    /// contracts to a full statevector (`omega-backend-mps/src/mps.rs:418`),
    /// and `resolve_backend` sends every non-Clifford circuit of n >= 20 to MPS
    /// under the documented default `backend: "Auto"`. A 34-qubit job priced at
    /// 4 MiB and allocated 256 GiB.
    #[test]
    fn mps_is_priced_dense_when_the_backend_densifies() {
        let chi = 64;
        let expectation = JobShape::new(34, CostKind::Mps { max_bond_dim: chi });
        let priced = estimate_peak_bytes(&expectation).expect("priceable");
        assert!(
            priced >= 256 * GB,
            "MPS expectation must be priced as the dense contraction it performs, got {priced}"
        );
        // In shot mode it genuinely is just tensors, and must stay cheap or we
        // would refuse the workload MPS exists for.
        let sampled = JobShape::new(34, CostKind::Mps { max_bond_dim: chi }).with_shots();
        let cheap = estimate_peak_bytes(&sampled).expect("priceable");
        assert!(
            cheap < 16 << 20,
            "MPS shot mode should stay tensor-sized, got {cheap}"
        );
    }

    /// Photonic cost is combinatorial in PHOTONS, so a mode ceiling does not
    /// bound it. 26 modes sits under a typical ceiling yet is ~300 GB.
    #[test]
    fn photonic_fock_space_is_priced_not_waved_through() {
        assert_eq!(fock_dim(4, 2), Some(10)); // C(5,2)
        assert_eq!(fock_dim(1, 5), Some(1));
        let modest = estimate_peak_bytes(&JobShape::new(4, CostKind::Photonic)).unwrap();
        assert!(modest < (1 << 20), "4 modes must stay tiny, got {modest}");
        // 26 modes -> 13 photons -> C(38,13) ~ 2.3e9 states.
        let big = estimate_peak_bytes(&JobShape::new(26, CostKind::Photonic)).unwrap();
        assert!(
            big > 100 * GB,
            "26-mode photonics is hundreds of GB and must be priced, got {big}"
        );
    }

    #[test]
    fn gradient_doubles_the_requirement() {
        let mut s = JobShape::new(30, CostKind::DenseStatevector);
        assert_eq!(estimate_peak_bytes(&s), Some(16 * GB));
        s.gradient = true;
        assert_eq!(estimate_peak_bytes(&s), Some(32 * GB));
        let _ = s;
    }

    /// A batch is executed as a sequential loop, so its peak is one row — not
    /// the sum. Pricing it as a sum would reject exactly the QML batches this
    /// path exists to serve.
    #[test]
    fn batch_does_not_multiply_peak_memory() {
        let mut s = JobShape::new(20, CostKind::DenseStatevector);
        let one = estimate_peak_bytes(&s).unwrap();
        s.batch = 256;
        assert_eq!(estimate_peak_bytes(&s), Some(one));
    }

    /// Compact representations are cheap ONLY in the mode that keeps them
    /// compact. Asserting they are always cheap is what made the first cost
    /// model wrong.
    #[test]
    fn mps_and_stabilizer_are_cheap_only_in_shot_mode() {
        let dense =
            estimate_peak_bytes(&JobShape::new(30, CostKind::DenseStatevector).with_shots())
                .unwrap();
        let mps = estimate_peak_bytes(
            &JobShape::new(30, CostKind::Mps { max_bond_dim: 64 }).with_shots(),
        )
        .unwrap();
        let stab =
            estimate_peak_bytes(&JobShape::new(30, CostKind::Stabilizer).with_shots()).unwrap();
        assert!(mps < dense / 1000, "mps {mps} vs dense {dense}");
        assert!(stab < dense / 1000, "stabilizer {stab} vs dense {dense}");
    }

    #[test]
    fn absurd_register_is_unpriceable_not_wrapped() {
        // Must not wrap around to a small number and get admitted.
        assert_eq!(
            estimate_peak_bytes(&JobShape::new(200, CostKind::DenseStatevector)),
            None
        );
    }

    #[test]
    fn oversized_job_is_refused_with_the_numbers() {
        let g = Governor::new(GovernorConfig {
            capacity_bytes: 8 * GB,
            max_qubits: 40,
            ..GovernorConfig::default()
        });
        // 34 qubits sampled: 2^34 x 16 B state + an equal-sized probs/cumulative
        // pair = 512 GB.
        let err = g
            .admit(&JobShape::new(34, CostKind::DenseStatevector).with_shots())
            .expect_err("512 GB must not be admitted against an 8 GB budget");
        assert!(matches!(err, Rejection::ExceedsCapacity { .. }));
        assert!(
            !err.is_transient(),
            "waiting cannot make 512 GB fit in 8 GB"
        );
        let msg = err.message();
        assert!(
            msg.contains("512.00 GB") && msg.contains("8.00 GB"),
            "refusal must quote both numbers, got: {msg}"
        );
    }

    #[test]
    fn qubit_ceiling_rejects_before_pricing() {
        let g = Governor::new(GovernorConfig {
            capacity_bytes: 1024 * GB,
            max_qubits: 24,
            ..GovernorConfig::default()
        });
        let err = g
            .admit(&JobShape::new(30, CostKind::DenseStatevector))
            .expect_err("above the ceiling");
        assert!(matches!(err, Rejection::TooManyQubits { .. }));
    }

    /// The point of weighting by bytes: concurrent admissions can never sum
    /// past the budget, however many threads race for it.
    ///
    /// This spawns real threads. An earlier version of this test was a
    /// sequential `(0..4).map(admit)` loop, which tested tokio's arithmetic
    /// rather than the governor and would have passed against a non-atomic
    /// counter.
    #[test]
    fn concurrent_admissions_never_exceed_the_budget() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        use std::sync::Barrier;

        const BUDGET_GB: u64 = 4;
        let g = Arc::new(Governor::new(GovernorConfig {
            capacity_bytes: BUDGET_GB * GB,
            max_qubits: 40,
            // Raised so the byte budget, not the slot cap, is what binds here.
            max_concurrency: 1024,
            ..GovernorConfig::default()
        }));
        // 26 qubits = 64M amps x 16 B = 1 GB, shots mode (x2 => 2 GB)... use
        // the analytic-free shot path so the arithmetic stays a round 2 GB.
        let shape = JobShape::new(25, CostKind::DenseStatevector).with_shots();
        let each = estimate_peak_bytes(&shape).unwrap();
        assert_eq!(each, GB, "25q shot-mode should price at exactly 1 GB");

        let threads = 32;
        let barrier = Arc::new(Barrier::new(threads));
        let admitted = Arc::new(AtomicUsize::new(0));
        let mut handles = Vec::new();
        for _ in 0..threads {
            let g = Arc::clone(&g);
            let barrier = Arc::clone(&barrier);
            let admitted = Arc::clone(&admitted);
            handles.push(std::thread::spawn(move || {
                barrier.wait(); // maximise the race
                match g.admit(&shape) {
                    Ok(res) => {
                        admitted.fetch_add(1, Ordering::SeqCst);
                        // Hold until every thread has had its turn, so the
                        // count reflects simultaneous occupancy.
                        std::thread::sleep(std::time::Duration::from_millis(50));
                        drop(res);
                        true
                    }
                    Err(_) => false,
                }
            }));
        }
        for h in handles {
            h.join().unwrap();
        }
        let n = admitted.load(Ordering::SeqCst);
        assert!(
            n as u64 * (each / GB) <= BUDGET_GB,
            "{n} concurrent 1 GB jobs admitted against a {BUDGET_GB} GB budget"
        );
        assert!(n > 0, "the budget should admit something");
    }

    /// A job-count limit cannot express this: the same governor admits many
    /// tiny jobs *and* refuses one large one, and the tiny jobs really do
    /// consume the budget rather than sitting there as decoration.
    #[test]
    fn byte_weighting_expresses_what_a_job_count_limit_cannot() {
        // Budget 64 MiB so the tiny jobs genuinely fill it. Concurrency is
        // raised out of the way so this isolates the BYTE property — the slot
        // cap is tested separately.
        let g = Governor::new(GovernorConfig {
            capacity_bytes: 64 << 20,
            max_qubits: 40,
            max_concurrency: 1024,
            ..GovernorConfig::default()
        });
        // 16 qubits shot-mode = 65536 amps x 16 B x 2 = 2 MiB.
        let tiny = JobShape::new(16, CostKind::DenseStatevector).with_shots();
        assert_eq!(estimate_peak_bytes(&tiny), Some(2 << 20));

        let mut held = Vec::new();
        for _ in 0..32 {
            held.push(g.admit(&tiny).expect("32 x 2 MiB fits 64 MiB"));
        }
        // The budget is now genuinely exhausted -- this exercises the
        // semaphore, not the capacity precheck.
        let err = g.admit(&tiny).expect_err("33rd 2 MiB job must not fit");
        assert!(matches!(err, Rejection::Busy { .. }), "got {err:?}");
        assert!(err.is_transient());

        // ...and freeing one lets exactly one more in.
        held.pop();
        g.admit(&tiny)
            .expect("capacity returns when a job finishes");
    }

    /// The ceiling must not punish backends whose cost is not `2^n`. A wide
    /// Clifford circuit is kilobytes on the stabilizer; refusing it because a
    /// *statevector* of that width would be huge is the governor being wrong,
    /// not safe.
    #[test]
    fn wide_cheap_backends_are_not_refused_by_the_dense_qubit_ceiling() {
        let g = Governor::new(GovernorConfig {
            capacity_bytes: 1024 * GB,
            max_qubits: 24,
            ..GovernorConfig::default()
        });
        g.admit(&JobShape::new(64, CostKind::Stabilizer).with_shots())
            .expect("64-qubit stabilizer sampling is kilobytes and must be admitted");
        g.admit(&JobShape::new(64, CostKind::Mps { max_bond_dim: 64 }).with_shots())
            .expect("64-qubit MPS sampling at chi=64 is megabytes and must be admitted");
        // The dense path still obeys the ceiling.
        let err = g
            .admit(&JobShape::new(30, CostKind::DenseStatevector).with_shots())
            .expect_err("dense width is still governed");
        assert!(matches!(err, Rejection::TooManyQubits { .. }));
    }

    /// The failure this whole topology split exists to prevent. On a unified
    /// machine, CPU and GPU work must debit ONE budget — charging them to
    /// separate pools would let them jointly exceed physical memory.
    #[test]
    fn unified_topology_charges_cpu_and_gpu_to_the_same_pool() {
        let g = Governor::with_topology(
            GovernorConfig {
                capacity_bytes: 4 * GB,
                max_qubits: 40,
                ..GovernorConfig::default()
            },
            &Topology::Unified {
                total_bytes: 4 * GB,
            },
        );
        // 2 GB of CPU work (25q shot-mode at 16 B/amp = 1 GB... take 26q = 2 GB).
        let cpu = JobShape::new(26, CostKind::DenseStatevector).with_shots();
        assert_eq!(estimate_peak_bytes(&cpu), Some(2 * GB));
        let _held = g.admit(&cpu).expect("2 GB of 4 GB");

        // A GPU job on a unified box competes for the SAME memory. 27q f32
        // shot-mode = 2^27 * 8 * 2 = 2 GB, which exactly fills the rest...
        let gpu = JobShape::new(27, CostKind::DenseStatevector)
            .with_shots()
            .on_device(0);
        assert_eq!(estimate_peak_bytes(&gpu), Some(2 * GB));
        let _held2 = g.admit(&gpu).expect("the remaining 2 GB");

        // ...and nothing more fits, because there is only one pool.
        let err = g
            .admit(&JobShape::new(20, CostKind::DenseStatevector).with_shots())
            .expect_err("unified pool is full; a second budget would be fiction");
        assert!(matches!(err, Rejection::Busy { .. }), "got {err:?}");
    }

    /// On a discrete machine the pools are genuinely independent: a full GPU
    /// must not block host work, and vice versa.
    #[test]
    fn discrete_topology_keeps_host_and_device_budgets_independent() {
        let g = Governor::with_topology(
            GovernorConfig {
                capacity_bytes: 64 * GB,
                max_qubits: 40,
                ..GovernorConfig::default()
            },
            &Topology::Discrete {
                host_bytes: 64 * GB,
                devices: vec![(0, 2 * GB)],
            },
        );
        // Fill the small device pool.
        let gpu = JobShape::new(27, CostKind::DenseStatevector)
            .with_shots()
            .on_device(0);
        let _held = g.admit(&gpu).expect("2 GB fits the 2 GB card");
        let err = g.admit(&gpu).expect_err("card is full");
        assert!(matches!(err, Rejection::Busy { .. }));

        // Host work is unaffected — the pools are separate.
        g.admit(&JobShape::new(26, CostKind::DenseStatevector).with_shots())
            .expect("host budget is untouched by a full GPU");
    }

    /// A job larger than device memory must be refused even when host RAM is
    /// ample — pricing GPU work against host RAM is wrong in both directions.
    #[test]
    fn device_job_is_sized_against_the_card_not_the_host() {
        let g = Governor::with_topology(
            GovernorConfig {
                capacity_bytes: 1024 * GB,
                max_qubits: 40,
                ..GovernorConfig::default()
            },
            &Topology::Discrete {
                host_bytes: 1024 * GB,
                devices: vec![(0, 8 * GB)],
            },
        );
        // 31 qubits f32 = 2^31 * 8 = 16 GB: trivial for the host, twice the card.
        let gpu = JobShape::new(31, CostKind::DenseStatevector).on_device(0);
        let err = g.admit(&gpu).expect_err("16 GB cannot fit an 8 GB card");
        match err {
            Rejection::ExceedsCapacity { capacity_bytes, .. } => {
                assert_eq!(capacity_bytes, 8 * GB, "must quote the CARD's capacity");
            }
            other => panic!("expected ExceedsCapacity against the device, got {other:?}"),
        }
        // The same width on the host is fine.
        g.admit(&JobShape::new(31, CostKind::DenseStatevector))
            .expect("16 GB (f64: 32 GB) fits a 1 TB host");
    }

    /// GPU statevector kernels are f32. Pricing device work at the CPU's 16 B
    /// per amplitude would refuse jobs that fit comfortably.
    #[test]
    fn device_work_is_priced_at_f32_width() {
        let cpu = JobShape::new(30, CostKind::DenseStatevector);
        let gpu = JobShape::new(30, CostKind::DenseStatevector).on_device(0);
        assert_eq!(estimate_peak_bytes(&cpu), Some(16 * GB));
        assert_eq!(
            estimate_peak_bytes(&gpu),
            Some(8 * GB),
            "f32 device amplitudes are half the CPU width"
        );
    }

    /// Fail closed: a device with no pool must be refused, never quietly
    /// charged to the host budget.
    #[test]
    fn unknown_device_is_refused_not_charged_to_the_host() {
        let g = Governor::with_topology(
            GovernorConfig {
                capacity_bytes: 1024 * GB,
                max_qubits: 40,
                ..GovernorConfig::default()
            },
            &Topology::Discrete {
                host_bytes: 1024 * GB,
                devices: vec![(0, 8 * GB)],
            },
        );
        let err = g
            .admit(&JobShape::new(10, CostKind::DenseStatevector).on_device(7))
            .expect_err("device 7 does not exist here");
        assert!(matches!(err, Rejection::Unpriceable { .. }), "got {err:?}");
    }

    /// Bytes bound total memory; the job-slot cap bounds CPU contention. Many
    /// tiny jobs can saturate the cores while barely touching the byte budget,
    /// which is exactly the case a memory-only limit cannot express.
    #[test]
    fn concurrency_cap_bounds_job_count_independently_of_bytes() {
        let g = Governor::new(GovernorConfig {
            capacity_bytes: 64 * GB,
            max_concurrency: 3,
            ..GovernorConfig::default()
        });
        let tiny = JobShape::new(4, CostKind::DenseStatevector).with_shots();
        let held: Vec<_> = (0..3)
            .map(|i| g.admit(&tiny).unwrap_or_else(|e| panic!("job {i}: {e:?}")))
            .collect();
        let err = g
            .admit(&tiny)
            .expect_err("4th job exceeds the slot cap even though bytes are free");
        assert!(matches!(err, Rejection::Busy { .. }), "got {err:?}");
        // Freeing one slot lets exactly one more in.
        drop(held);
        g.admit(&tiny).expect("slots return when jobs finish");
    }

    /// The gap a TLA+ check found in the ledger-only policy: two QML rows
    /// priced at 8 GB against a 64 GB budget report 56 GB of headroom while
    /// the machine holds 68 GB, because the governor never sees a job's
    /// dataset or optimizer state. Admission satisfied, box over budget.
    ///
    /// The measurement is the second opinion.
    #[test]
    fn machine_pressure_overrides_a_ledger_that_says_there_is_room() {
        const GB64: u64 = 64 << 30;
        // Ledger view: plenty free. Machine view: almost nothing.
        assert!(
            exceeds_machine_headroom(16 * GB, Some(4 * GB)),
            "16 GB must be refused when only 4 GB is actually available"
        );
        // Comfortably within the measured headroom: admit.
        assert!(!exceeds_machine_headroom(4 * GB, Some(GB64)));
        // Right at the margin: 0.8 * 10 = 8, so 8 fits and 9 does not.
        assert!(!exceeds_machine_headroom(8 * GB, Some(10 * GB)));
        assert!(exceeds_machine_headroom(9 * GB, Some(10 * GB)));
    }

    /// A platform that will not report memory must fall back to the ledger,
    /// not to a guess. Refusing everything would be worse than the gap.
    #[test]
    fn unmeasurable_machine_falls_back_to_the_ledger() {
        assert!(!exceeds_machine_headroom(1024 * GB, None));
    }

    #[test]
    fn machine_pressure_is_retryable_and_says_both_numbers() {
        let r = Rejection::MachinePressure {
            required_bytes: 16 * GB,
            measured_available_bytes: 4 * GB,
        };
        // Memory frees up; this is a wait, not a permanent no.
        assert!(r.is_transient());
        let m = r.message();
        assert!(m.contains("16.00 GB") && m.contains("4.00 GB"), "{m}");
        assert!(
            m.contains("MACHINE"),
            "must distinguish ledger from machine: {m}"
        );
    }

    #[test]
    fn default_qubit_ceiling_is_derived_from_the_budget() {
        // 16 GB budget: n=29 needs 8 GB forward + 8 GB adjoint, which fits;
        // n=30 would need 32 GB and must not be allowed by default.
        assert_eq!(default_qubit_ceiling(16 * GB), 29);
        assert_eq!(default_qubit_ceiling(32 * GB), 30);
        let g = Governor::new(GovernorConfig {
            capacity_bytes: 16 * GB,
            max_qubits: default_qubit_ceiling(16 * GB),
            ..GovernorConfig::default()
        });
        let mut s = JobShape::new(29, CostKind::DenseStatevector);
        s.gradient = true;
        // 2^29 x 16 x 2 = 16 GB exactly.
        assert_eq!(estimate_peak_bytes(&s), Some(16 * GB));
        g.admit(&s).expect("29q with adjoint fits 16 GB");
    }
}
