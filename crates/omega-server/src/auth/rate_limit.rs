//! Phase 14a — sliding-window rate limiter.
//!
//! Two surfaces:
//! - [`rate_limit_ip`] — runs *before* auth, throttles anonymous traffic
//!   per source IP. Limit configured via `OMEGA_RATELIMIT_PER_IP_PER_MIN`
//!   (default 60). Set to 0 to disable.
//! - [`rate_limit_subject`] — runs *after* auth, throttles per
//!   `TokenClaims::sub`. Limit configured via
//!   `OMEGA_RATELIMIT_PER_SUBJECT_PER_MIN` (default 600). Set to 0 to
//!   disable.
//!
//! Implementation: per-key sliding window of request timestamps in a
//! `Mutex<HashMap<K, VecDeque<Instant>>>`. On each request the entries
//! older than the window are popped, the current one is pushed, and
//! the count is compared against the limit. A 429 response with a
//! `Retry-After` header is returned when the limit is exceeded.
//!
//! Map entries with empty deques are GC'd opportunistically every
//! `GC_EVERY` requests so a long-running server doesn't accumulate
//! one-shot client IPs forever.

use std::collections::{HashMap, VecDeque};
use std::net::SocketAddr;
use std::sync::Mutex;
use std::time::{Duration, Instant};

use axum::body::Body;
use axum::extract::{ConnectInfo, Extension, State};
use axum::http::{HeaderValue, Request, StatusCode};
use axum::middleware::Next;
use axum::response::{IntoResponse, Response};
use axum::Json;

use crate::auth::token::TokenClaims;

/// Per-key sliding-window rate limiter. `K` is the bucket key
/// (`IpAddr` for the per-IP path, `String` subject for the per-subject
/// path). Cheap to clone since `Self` is `Arc`-shaped via the
/// `axum::middleware::from_fn_with_state` plumbing.
pub struct SlidingWindowLimiter<K: Eq + std::hash::Hash + Clone> {
    pub limit: u32,
    pub window: Duration,
    inner: Mutex<Inner<K>>,
}

struct Inner<K> {
    buckets: HashMap<K, VecDeque<Instant>>,
    /// Counter for opportunistic GC sweeps.
    requests_since_gc: u64,
}

/// Sweep map for empty deques every N requests.
const GC_EVERY: u64 = 1024;

impl<K: Eq + std::hash::Hash + Clone> SlidingWindowLimiter<K> {
    pub fn new(limit: u32, window: Duration) -> Self {
        Self {
            limit,
            window,
            inner: Mutex::new(Inner {
                buckets: HashMap::new(),
                requests_since_gc: 0,
            }),
        }
    }

    /// Check & record one request from `key`. Returns `Ok(())` if
    /// under the limit (and the request is recorded), or `Err(retry_after_seconds)`
    /// when the bucket is full. A `limit` of 0 disables the limiter
    /// (always returns Ok).
    pub fn check(&self, key: &K) -> Result<(), u64> {
        if self.limit == 0 {
            return Ok(());
        }
        let now = Instant::now();
        let mut inner = self.inner.lock().expect("limiter mutex poisoned");
        inner.requests_since_gc += 1;
        if inner.requests_since_gc >= GC_EVERY {
            inner.requests_since_gc = 0;
            inner.buckets.retain(|_, deque| {
                while deque
                    .front()
                    .is_some_and(|t| now.duration_since(*t) >= self.window)
                {
                    deque.pop_front();
                }
                !deque.is_empty()
            });
        }
        let deque = inner.buckets.entry(key.clone()).or_default();
        while deque
            .front()
            .is_some_and(|t| now.duration_since(*t) >= self.window)
        {
            deque.pop_front();
        }
        if deque.len() as u32 >= self.limit {
            // Retry-after = seconds until the oldest tracked request
            // ages out of the window. Round up.
            let oldest = *deque.front().expect("non-empty by len check");
            let elapsed = now.duration_since(oldest);
            let remaining = self.window.saturating_sub(elapsed);
            return Err(remaining.as_secs().max(1));
        }
        deque.push_back(now);
        Ok(())
    }
}

fn limit_from_env(var: &str, default: u32) -> u32 {
    std::env::var(var)
        .ok()
        .and_then(|s| s.parse::<u32>().ok())
        .unwrap_or(default)
}

/// Read the per-IP limit from `OMEGA_RATELIMIT_PER_IP_PER_MIN`
/// (default 60; 0 disables).
pub fn per_ip_limit() -> u32 {
    limit_from_env("OMEGA_RATELIMIT_PER_IP_PER_MIN", 60)
}

/// Read the per-subject limit from `OMEGA_RATELIMIT_PER_SUBJECT_PER_MIN`
/// (default 600; 0 disables).
pub fn per_subject_limit() -> u32 {
    limit_from_env("OMEGA_RATELIMIT_PER_SUBJECT_PER_MIN", 600)
}

/// Build a fresh per-IP limiter from env config. The `Limiter` itself
/// is wrapped by the middleware in an `Arc` via `axum::Extension` /
/// closure capture.
pub fn build_ip_limiter() -> SlidingWindowLimiter<std::net::IpAddr> {
    SlidingWindowLimiter::new(per_ip_limit(), Duration::from_secs(60))
}

/// Build a fresh per-subject limiter from env config.
pub fn build_subject_limiter() -> SlidingWindowLimiter<String> {
    SlidingWindowLimiter::new(per_subject_limit(), Duration::from_secs(60))
}

fn rate_limit_response(retry_after: u64) -> Response {
    let mut resp = (
        StatusCode::TOO_MANY_REQUESTS,
        Json(serde_json::json!({"error": "rate limit exceeded"})),
    )
        .into_response();
    if let Ok(v) = HeaderValue::from_str(&retry_after.to_string()) {
        resp.headers_mut().insert("retry-after", v);
    }
    resp
}

/// Middleware: throttle by source IP. Wire via
/// `axum::middleware::from_fn_with_state(Arc::clone(&ip_limiter), rate_limit_ip)`
/// so the limiter is bound through axum's state machinery rather than
/// `Extension(...)`. The Extension-layer wiring used previously was
/// silently broken — rate_limit_ip ran before its companion Extension
/// layer added the limiter Arc to request extensions, so every request
/// (including `/health`) returned 500 with "Missing request extension".
/// `from_fn_with_state` plumbs the value through a typed channel that
/// can't get the ordering wrong.
pub async fn rate_limit_ip(
    State(limiter): State<std::sync::Arc<SlidingWindowLimiter<std::net::IpAddr>>>,
    addr: Option<ConnectInfo<SocketAddr>>,
    req: Request<Body>,
    next: Next,
) -> Response {
    if let Some(ConnectInfo(socket)) = addr {
        if let Err(retry_after) = limiter.check(&socket.ip()) {
            return rate_limit_response(retry_after);
        }
    }
    next.run(req).await
}

/// Middleware: throttle by authenticated subject (TokenClaims::sub).
/// Must run *after* `middleware::require_auth` so the `TokenClaims`
/// extension is populated. Like [`rate_limit_ip`], the limiter is
/// passed via `from_fn_with_state` rather than `Extension(...)`.
pub async fn rate_limit_subject(
    State(limiter): State<std::sync::Arc<SlidingWindowLimiter<String>>>,
    claims: Option<Extension<TokenClaims>>,
    req: Request<Body>,
    next: Next,
) -> Response {
    if let Some(Extension(c)) = claims {
        if let Err(retry_after) = limiter.check(&c.sub) {
            return rate_limit_response(retry_after);
        }
    }
    next.run(req).await
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::{IpAddr, Ipv4Addr};

    #[test]
    fn limiter_allows_below_limit() {
        let l = SlidingWindowLimiter::<IpAddr>::new(3, Duration::from_secs(60));
        let ip: IpAddr = Ipv4Addr::new(127, 0, 0, 1).into();
        assert!(l.check(&ip).is_ok());
        assert!(l.check(&ip).is_ok());
        assert!(l.check(&ip).is_ok());
    }

    #[test]
    fn limiter_blocks_over_limit_with_retry_after() {
        let l = SlidingWindowLimiter::<IpAddr>::new(2, Duration::from_secs(60));
        let ip: IpAddr = Ipv4Addr::new(127, 0, 0, 1).into();
        assert!(l.check(&ip).is_ok());
        assert!(l.check(&ip).is_ok());
        match l.check(&ip) {
            Err(retry) => assert!(retry >= 1, "retry-after must be ≥ 1s, got {retry}"),
            Ok(()) => panic!("third request should have been throttled"),
        }
    }

    #[test]
    fn limiter_independent_keys() {
        // Two different IPs each get their own bucket.
        let l = SlidingWindowLimiter::<IpAddr>::new(1, Duration::from_secs(60));
        let a: IpAddr = Ipv4Addr::new(10, 0, 0, 1).into();
        let b: IpAddr = Ipv4Addr::new(10, 0, 0, 2).into();
        assert!(l.check(&a).is_ok());
        assert!(l.check(&b).is_ok(), "different IP should not be throttled");
        assert!(l.check(&a).is_err(), "same IP at limit should be throttled");
    }

    #[test]
    fn limit_zero_disables() {
        let l = SlidingWindowLimiter::<IpAddr>::new(0, Duration::from_secs(60));
        let ip: IpAddr = Ipv4Addr::new(127, 0, 0, 1).into();
        // Hit it many times — must always succeed.
        for _ in 0..1000 {
            assert!(l.check(&ip).is_ok());
        }
    }

    #[test]
    fn entries_age_out_of_window() {
        // 1 request per 50 ms window. After waiting 60 ms, the next
        // request should be allowed again.
        let l = SlidingWindowLimiter::<IpAddr>::new(1, Duration::from_millis(50));
        let ip: IpAddr = Ipv4Addr::new(127, 0, 0, 1).into();
        assert!(l.check(&ip).is_ok());
        assert!(l.check(&ip).is_err());
        std::thread::sleep(Duration::from_millis(70));
        assert!(l.check(&ip).is_ok(), "after window the bucket should reset");
    }

    #[test]
    fn env_defaults_are_sensible() {
        // Without the env var the default kicks in (60 / minute).
        std::env::remove_var("OMEGA_RATELIMIT_PER_IP_PER_MIN");
        assert_eq!(per_ip_limit(), 60);
        std::env::remove_var("OMEGA_RATELIMIT_PER_SUBJECT_PER_MIN");
        assert_eq!(per_subject_limit(), 600);
    }
}
