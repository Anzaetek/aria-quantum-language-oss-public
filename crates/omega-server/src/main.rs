use std::sync::Arc;
use tokio::sync::RwLock;

mod auth;
mod lambda;
mod pki;
mod pqc;
mod quantum_bridge;
mod registry;
mod routes;
mod worker;
mod ws;

#[cfg(test)]
mod routes_integration_tests;

use registry::Registry;

/// CLI options parsed from argv. Anything not covered here still comes from
/// the OMEGA_* env vars (OMEGA_PORT, OMEGA_DB_PATH, OMEGA_BOOTSTRAP_TTL).
#[derive(Default)]
struct CliOpts {
    /// Write the bootstrap admin token to this file instead of dumping it
    /// to stdout. The file ends with a single newline, so
    /// `OMEGA_TOKEN=$(cat path)` and `xargs < path` both work.
    save_token_to: Option<std::path::PathBuf>,
    /// Authentication / transport mode. `pqc` (default, current behaviour)
    /// exposes both bearer-token HTTP and the PQC-encrypted WebSocket at
    /// `/v1/ws`; `bearer-only` skips the WS endpoint and the server-cert
    /// generation entirely. `bearer-only` is useful when fronted by TLS,
    /// or when the operator only needs the simpler curl-style flow.
    auth_mode: AuthMode,
}

#[derive(Default, Clone, Copy, PartialEq, Eq)]
enum AuthMode {
    #[default]
    Pqc,
    BearerOnly,
}

fn parse_cli() -> CliOpts {
    let mut opts = CliOpts::default();
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--save-token-to" => match args.next() {
                Some(p) => opts.save_token_to = Some(p.into()),
                None => {
                    eprintln!("--save-token-to: missing PATH argument");
                    std::process::exit(2);
                }
            },
            s if s.starts_with("--save-token-to=") => {
                let p = &s["--save-token-to=".len()..];
                opts.save_token_to = Some(p.into());
            }
            "--auth" => match args.next().as_deref() {
                Some("pqc") => opts.auth_mode = AuthMode::Pqc,
                Some("bearer-only") => opts.auth_mode = AuthMode::BearerOnly,
                Some(other) => {
                    eprintln!("--auth: expected `pqc` or `bearer-only`, got `{other}`");
                    std::process::exit(2);
                }
                None => {
                    eprintln!("--auth: missing MODE argument");
                    std::process::exit(2);
                }
            },
            s if s.starts_with("--auth=") => match &s["--auth=".len()..] {
                "pqc" => opts.auth_mode = AuthMode::Pqc,
                "bearer-only" => opts.auth_mode = AuthMode::BearerOnly,
                other => {
                    eprintln!("--auth: expected `pqc` or `bearer-only`, got `{other}`");
                    std::process::exit(2);
                }
            },
            "-h" | "--help" => {
                print_usage();
                std::process::exit(0);
            }
            other => {
                eprintln!("Unknown argument: {other}");
                print_usage();
                std::process::exit(2);
            }
        }
    }
    // Env-var fallback so `OMEGA_TOKEN_FILE=/etc/omega/admin.token` works
    // alongside the OMEGA_* family.
    if opts.save_token_to.is_none() {
        if let Ok(p) = std::env::var("OMEGA_TOKEN_FILE") {
            opts.save_token_to = Some(p.into());
        }
    }
    if let Ok(v) = std::env::var("OMEGA_AUTH_MODE") {
        match v.as_str() {
            "pqc" => opts.auth_mode = AuthMode::Pqc,
            "bearer-only" => opts.auth_mode = AuthMode::BearerOnly,
            other => {
                eprintln!("OMEGA_AUTH_MODE: expected `pqc` or `bearer-only`, got `{other}`");
                std::process::exit(2);
            }
        }
    }
    opts
}

fn print_usage() {
    eprintln!("Usage: omega-server [--save-token-to PATH] [--auth pqc|bearer-only]");
    eprintln!();
    eprintln!("Options:");
    eprintln!("  --save-token-to PATH   Write the bootstrap admin token to PATH (mode 0600 on");
    eprintln!("                         Unix) instead of printing it to stdout. Equivalent to");
    eprintln!("                         setting OMEGA_TOKEN_FILE=PATH.");
    eprintln!("  --auth MODE            `pqc` (default): expose bearer-token HTTP routes and");
    eprintln!("                         the PQC-encrypted WebSocket at /v1/ws.");
    eprintln!("                         `bearer-only`: skip the WS endpoint and the server-cert");
    eprintln!("                         generation; the simplest curl-friendly flow. Equivalent");
    eprintln!("                         to setting OMEGA_AUTH_MODE=bearer-only.");
    eprintln!("  -h, --help             Show this help.");
    eprintln!();
    eprintln!("Environment variables:");
    eprintln!("  OMEGA_PORT             Listen port (default 8080).");
    eprintln!("  OMEGA_DB_PATH          SQLite path (default ./omega.db).");
    eprintln!("  OMEGA_BOOTSTRAP_TTL    Bootstrap token TTL in seconds (default 86400).");
    eprintln!("  OMEGA_TOKEN_FILE       Fallback for --save-token-to.");
    eprintln!("  OMEGA_AUTH_MODE        Fallback for --auth (`pqc` or `bearer-only`).");
}

fn write_token_file(path: &std::path::Path, token: &str) -> std::io::Result<()> {
    use std::io::Write;
    // Open with restrictive permissions on Unix so the file isn't world-readable
    // the moment it appears on disk.
    #[cfg(unix)]
    let mut f = {
        use std::os::unix::fs::OpenOptionsExt;
        std::fs::OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .mode(0o600)
            .open(path)?
    };
    #[cfg(not(unix))]
    let mut f = std::fs::OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .open(path)?;
    f.write_all(token.as_bytes())?;
    f.write_all(b"\n")?;
    f.flush()?;
    Ok(())
}

#[tokio::main]
async fn main() {
    let cli = parse_cli();

    let db_path = std::env::var("OMEGA_DB_PATH").unwrap_or_else(|_| "omega.db".to_string());
    let port: u16 = std::env::var("OMEGA_PORT")
        .ok()
        .and_then(|p| p.parse().ok())
        .unwrap_or(8080);

    let registry = Registry::new(&db_path).expect("Failed to initialize database");

    // Initialize auth tables and signing key
    {
        let conn = registry.conn();
        auth::store::init_auth_tables(&conn).expect("Failed to initialize auth tables");
    }

    let (active_kid, active_pk, active_sk) = {
        let conn = registry.conn();
        auth::store::ensure_signing_key(&conn).expect("Failed to initialize signing key")
    };

    // Generate bootstrap admin token on first run
    let default_ttl: i64 = std::env::var("OMEGA_BOOTSTRAP_TTL")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(86400); // 24 hours default

    let (bootstrap_token, bootstrap_claims) = auth::token::issue_token(
        "admin",
        auth::rights::ADMIN_ROLE,
        default_ttl,
        &active_kid,
        &active_sk,
    )
    .expect("Failed to generate bootstrap token");

    {
        let conn = registry.conn();
        auth::store::store_token(&conn, &bootstrap_claims)
            .expect("Failed to store bootstrap token");
    }

    // Load the PKI trust store from OMEGA_PKI_TRUST_STORE at boot.
    // Unset / empty → empty store; missing or corrupt directory →
    // panic so the operator notices before serving traffic.
    let trust_store = pqc::trust_store::TrustStore::from_env()
        .expect("Failed to load PKI trust store (check OMEGA_PKI_TRUST_STORE)");
    if let Some(dir) = trust_store.source_dir() {
        eprintln!(
            "PKI trust store loaded {} root(s) from {}",
            trust_store.len(),
            dir.display()
        );
    }
    let trust_store = Arc::new(trust_store);

    // CRL is parallel: OMEGA_PKI_CRL_FILE optional, panic on
    // corrupt-but-set so the operator catches misconfig early.
    let crl =
        pqc::crl::OmegaCrl::from_env().expect("Failed to load PKI CRL (check OMEGA_PKI_CRL_FILE)");
    if let Some(c) = &crl {
        eprintln!(
            "PKI CRL loaded {} revoked entries from issuer {:?}",
            c.body.revoked.len(),
            c.body.issuer
        );
    }
    let crl = crl.map(Arc::new);

    let ws_state = match cli.auth_mode {
        AuthMode::Pqc => {
            // Generate server certificate for PQC WebSocket (Phase 2).
            let server_cert = pqc::certificate::OmegaCert::self_sign(
                "omega-server",
                default_ttl,
                &active_sk,
                None,
            )
            .expect("Failed to create server certificate");
            // Phase 10: chain-aware client cert validation. Policy is
            // env-driven (`OMEGA_PKI_CLIENT_CERT_POLICY=off|optional|required`,
            // default `off` for back-compat). Trust store + CRL are
            // already loaded above; the WS handler needs its own
            // handles to consult them inside the handshake without
            // taking the AppState lock.
            let client_cert_policy = ws::handshake::ClientCertPolicy::from_env();
            eprintln!(
                "PQC WS client-cert policy: {} (trust roots: {}, CRL: {})",
                client_cert_policy.as_str(),
                trust_store.len(),
                if crl.is_some() { "yes" } else { "no" },
            );
            Some(Arc::new(RwLock::new(ws::handler::WsState {
                server_cert,
                trust_store: Arc::clone(&trust_store),
                crl: crl.clone(),
                client_cert_policy,
            })))
        }
        AuthMode::BearerOnly => None,
    };

    let state = Arc::new(RwLock::new(AppState {
        registry,
        active_kid,
        active_pk,
        active_sk: active_sk.clone(),
        trust_store,
        crl,
    }));

    let app = routes::create_router(state, ws_state);

    let addr = format!("0.0.0.0:{}", port);
    println!("Omega Functions server starting on {}", addr);
    println!("API: http://localhost:{}/v1/", port);
    println!(
        "auth: {} ({})",
        match cli.auth_mode {
            AuthMode::Pqc => "pqc",
            AuthMode::BearerOnly => "bearer-only",
        },
        match cli.auth_mode {
            AuthMode::Pqc => "bearer HTTP + PQC WebSocket at /v1/ws",
            AuthMode::BearerOnly => "bearer HTTP only — /v1/ws disabled",
        }
    );
    println!();
    println!("=== ADMIN TOKEN ===");
    if let Some(path) = cli.save_token_to.as_deref() {
        match write_token_file(path, &bootstrap_token) {
            Ok(()) => {
                println!("saved to: {} (mode 0600 on Unix)", path.display());
                println!("load:     export OMEGA_TOKEN=$(cat {})", path.display());
            }
            Err(e) => {
                eprintln!("Failed to write bootstrap token to {}: {e}", path.display());
                eprintln!("Falling back to stdout dump:");
                println!("{}", bootstrap_token);
            }
        }
    } else {
        println!("{}", bootstrap_token);
    }
    println!("jti:     {}", bootstrap_claims.jti);
    println!("expires: {} seconds", default_ttl);
    println!("===================");

    let listener = tokio::net::TcpListener::bind(&addr).await.unwrap();
    // `into_make_service_with_connect_info::<SocketAddr>()` populates
    // the per-request `ConnectInfo<SocketAddr>` extension that the
    // per-IP rate-limit middleware reads.
    let make_service = app.into_make_service_with_connect_info::<std::net::SocketAddr>();

    // Phase 14a: optional rustls listener alongside the plain HTTP one.
    // Activated only when both `OMEGA_TLS_CERT` and `OMEGA_TLS_KEY` are
    // set in the environment AND the binary is built with
    // `--features tls`. Without the feature compiled in, the env vars
    // are accepted but the TLS listener simply doesn't spin up — the
    // server prints a notice to stderr so operators don't silently
    // mis-deploy. With both feature + env, both listeners run
    // concurrently via tokio::select on the same router.
    #[cfg(feature = "tls")]
    {
        if let (Ok(cert_path), Ok(key_path)) = (
            std::env::var("OMEGA_TLS_CERT"),
            std::env::var("OMEGA_TLS_KEY"),
        ) {
            let tls_addr =
                std::env::var("OMEGA_TLS_ADDR").unwrap_or_else(|_| "0.0.0.0:8443".into());
            let tls_socket: std::net::SocketAddr =
                tls_addr.parse().expect("OMEGA_TLS_ADDR must be host:port");
            let tls_cfg =
                match axum_server::tls_rustls::RustlsConfig::from_pem_file(&cert_path, &key_path)
                    .await
                {
                    Ok(c) => c,
                    Err(e) => {
                        eprintln!(
                            "TLS: failed to load cert={cert_path} / key={key_path}: {e}. \
                         Falling back to plain-HTTP listener only.",
                        );
                        axum::serve(listener, make_service).await.unwrap();
                        return;
                    }
                };
            println!("TLS:  https://{tls_addr}/v1/");
            let tls_make = make_service.clone();
            tokio::select! {
                r = axum::serve(listener, make_service) => r.unwrap(),
                r = axum_server::bind_rustls(tls_socket, tls_cfg).serve(tls_make) =>
                    r.unwrap(),
            }
            return;
        }
    }
    #[cfg(not(feature = "tls"))]
    {
        if std::env::var("OMEGA_TLS_CERT").is_ok() || std::env::var("OMEGA_TLS_KEY").is_ok() {
            eprintln!(
                "Note: OMEGA_TLS_CERT / OMEGA_TLS_KEY set but the server was built \
                 without --features tls; the rustls listener stays disabled."
            );
        }
    }
    axum::serve(listener, make_service).await.unwrap();
}

pub struct AppState {
    pub registry: Registry,
    pub active_kid: String,
    pub active_pk: Vec<u8>,
    pub active_sk: Vec<u8>,
    /// Trust roots loaded from `OMEGA_PKI_TRUST_STORE` at boot. The
    /// WS handshake / mTLS-like client-cert verification step (still
    /// open in TODO) consults this. Empty when the env var is unset.
    pub trust_store: Arc<pqc::trust_store::TrustStore>,
    /// CRL loaded from `OMEGA_PKI_CRL_FILE` at boot. `None` when the
    /// env var is unset (operators not yet running revocations).
    /// Cached as `Arc` so the WS handshake path can consult it
    /// without holding the AppState lock.
    pub crl: Option<Arc<pqc::crl::OmegaCrl>>,
}
