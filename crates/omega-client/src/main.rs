//! omega-client — curl-style CLI for the omega-server REST API.
//!
//! The server exposes a bearer-token HTTP surface (see
//! `omega-server --help`). This binary wraps the common verbs
//! (`/v1/circuits`, `/v1/functions`, `/v1/lambdas`, `/v1/invocations`)
//! so operators don't have to remember curl flags or hand-roll JSON.
//!
//! Design constraints:
//! - depends on std + serde_json + base64 only (no reqwest, no tokio);
//! - plain HTTP only — front the server with TLS via a reverse proxy
//!   if you need the wire to be encrypted on the open internet;
//! - PQC WebSocket (`/v1/ws`) is a follow-up; for now use `--auth pqc`
//!   on the server with a separate WS client.
//!
//! Token resolution order:
//! 1. `--token VALUE` flag,
//! 2. `--token-file PATH` flag,
//! 3. `OMEGA_TOKEN` env var,
//! 4. `OMEGA_TOKEN_FILE` env var.
//!
//! Server URL resolution: `--server URL` then `OMEGA_SERVER` env var
//! (default `http://localhost:8080`).

use std::collections::BTreeMap;
use std::io::{Read, Write};
use std::net::TcpStream;
use std::process::ExitCode;
use std::time::Duration;

use base64::Engine;

const DEFAULT_SERVER: &str = "http://localhost:8080";
const USER_AGENT: &str = concat!("omega-client/", env!("CARGO_PKG_VERSION"));

fn main() -> ExitCode {
    let argv: Vec<String> = std::env::args().collect();
    let opts = match parse_cli(&argv[1..]) {
        Ok(o) => o,
        Err(e) => {
            eprintln!("{}", e);
            print_usage();
            return ExitCode::from(2);
        }
    };

    if opts.show_help {
        print_usage();
        return ExitCode::SUCCESS;
    }

    let cmd = match &opts.command {
        Some(c) => c,
        None => {
            print_usage();
            return ExitCode::from(2);
        }
    };

    match dispatch(&opts, cmd) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {}", e);
            ExitCode::from(1)
        }
    }
}

#[derive(Default)]
struct Opts {
    server: String,
    token: Option<String>,
    raw: bool,
    show_help: bool,
    command: Option<Vec<String>>,
}

fn parse_cli(args: &[String]) -> Result<Opts, String> {
    let mut opts = Opts {
        server: std::env::var("OMEGA_SERVER").unwrap_or_else(|_| DEFAULT_SERVER.to_string()),
        token: None,
        raw: false,
        show_help: false,
        command: None,
    };

    // Token: env-var fallback resolved up-front so flags can override.
    if let Ok(t) = std::env::var("OMEGA_TOKEN") {
        if !t.is_empty() {
            opts.token = Some(t);
        }
    } else if let Ok(p) = std::env::var("OMEGA_TOKEN_FILE") {
        opts.token = Some(read_token_file(&p)?);
    }

    let mut i = 0;
    while i < args.len() {
        let a = &args[i];
        match a.as_str() {
            "-h" | "--help" => {
                opts.show_help = true;
                i += 1;
            }
            "--server" => {
                opts.server = args.get(i + 1).ok_or("--server: missing URL")?.clone();
                i += 2;
            }
            s if s.starts_with("--server=") => {
                opts.server = s["--server=".len()..].to_string();
                i += 1;
            }
            "--token" => {
                opts.token = Some(args.get(i + 1).ok_or("--token: missing VALUE")?.clone());
                i += 2;
            }
            s if s.starts_with("--token=") => {
                opts.token = Some(s["--token=".len()..].to_string());
                i += 1;
            }
            "--token-file" => {
                let p = args.get(i + 1).ok_or("--token-file: missing PATH")?.clone();
                opts.token = Some(read_token_file(&p)?);
                i += 2;
            }
            s if s.starts_with("--token-file=") => {
                let p = &s["--token-file=".len()..];
                opts.token = Some(read_token_file(p)?);
                i += 1;
            }
            "--raw" => {
                opts.raw = true;
                i += 1;
            }
            _ => {
                opts.command = Some(args[i..].to_vec());
                break;
            }
        }
    }
    Ok(opts)
}

fn read_token_file(path: &str) -> Result<String, String> {
    let raw = std::fs::read_to_string(path).map_err(|e| format!("read {}: {}", path, e))?;
    Ok(raw.trim().to_string())
}

fn print_usage() {
    eprintln!("omega-client — curl-style CLI for omega-server");
    eprintln!();
    eprintln!("USAGE:");
    eprintln!("  omega-client [GLOBAL OPTIONS] <COMMAND> [ARGS]");
    eprintln!();
    eprintln!("GLOBAL OPTIONS:");
    eprintln!(
        "  --server URL       Base URL (default $OMEGA_SERVER or {})",
        DEFAULT_SERVER
    );
    eprintln!("  --token VALUE      Bearer token (overrides $OMEGA_TOKEN)");
    eprintln!("  --token-file PATH  Read bearer token from PATH (overrides --token)");
    eprintln!("  --raw              Print raw response body (default: pretty JSON)");
    eprintln!("  -h, --help         Show this help");
    eprintln!();
    eprintln!("COMMANDS:");
    eprintln!("  health                                Server liveness check");
    eprintln!("  whoami                                Decode the bearer token claims");
    eprintln!("  backends                              List available execution backends");
    eprintln!("  circuits upload FILE                  POST /v1/circuits");
    eprintln!("  circuits list                         GET  /v1/circuits");
    eprintln!("  circuits get ID                       GET  /v1/circuits/:id");
    eprintln!("  circuits delete ID                    DELETE /v1/circuits/:id");
    eprintln!("  functions create CIRCUIT_ID NAME [--shots N]");
    eprintln!("  functions list");
    eprintln!("  functions get ID");
    eprintln!("  functions invoke ID [--params P,P,...] [--shots N] [--seed N]");
    eprintln!("  lambdas register FILE NAME            POST /v1/lambdas (FILE is .wasm)");
    eprintln!("  lambdas list");
    eprintln!("  lambdas get ID");
    eprintln!("  lambdas delete ID");
    eprintln!("  lambdas invoke ID [--input JSON | --input-file PATH]");
    eprintln!("  invocation get ID                     GET /v1/invocations/:id");
    eprintln!("  raw METHOD PATH [--data JSON]         escape hatch");
    eprintln!();
    eprintln!("ENV:");
    eprintln!("  OMEGA_SERVER       Default --server");
    eprintln!("  OMEGA_TOKEN        Default --token");
    eprintln!("  OMEGA_TOKEN_FILE   Default --token-file");
    eprintln!();
    eprintln!("Notes:");
    eprintln!("  - This client speaks plain HTTP. For the open internet, front omega-server");
    eprintln!("    with a TLS-terminating reverse proxy.");
    eprintln!("  - The PQC WebSocket at /v1/ws is not yet wrapped here; use --auth pqc on");
    eprintln!("    the server only when you have a separate WS client.");
}

fn dispatch(opts: &Opts, cmd: &[String]) -> Result<(), String> {
    let head = cmd[0].as_str();
    let tail = &cmd[1..];
    match head {
        "health" => run_request(opts, "GET", "/health", None),
        "whoami" => cmd_whoami(opts),
        "backends" => run_request(opts, "GET", "/v1/backends", None),
        "circuits" => cmd_circuits(opts, tail),
        "functions" => cmd_functions(opts, tail),
        "lambdas" => cmd_lambdas(opts, tail),
        "invocation" => cmd_invocation(opts, tail),
        "raw" => cmd_raw(opts, tail),
        other => Err(format!("unknown command: {}", other)),
    }
}

// ---- Sub-commands ----------------------------------------------------------

fn cmd_whoami(opts: &Opts) -> Result<(), String> {
    let token = opts
        .token
        .as_deref()
        .ok_or("whoami: no bearer token (set --token or $OMEGA_TOKEN)")?;
    let parts: Vec<&str> = token.splitn(2, '.').collect();
    if parts.len() != 2 {
        return Err("whoami: token does not look like <payload>.<sig>".to_string());
    }
    let payload_bytes = base64::engine::general_purpose::URL_SAFE_NO_PAD
        .decode(parts[0])
        .map_err(|e| format!("base64 decode: {}", e))?;
    let v: serde_json::Value =
        serde_json::from_slice(&payload_bytes).map_err(|e| format!("payload JSON: {}", e))?;
    print_json(&v, opts.raw);
    Ok(())
}

fn cmd_circuits(opts: &Opts, args: &[String]) -> Result<(), String> {
    let sub = args.first().map(String::as_str).unwrap_or("");
    match sub {
        "upload" => {
            let file = args.get(1).ok_or("circuits upload: missing FILE")?;
            let source =
                std::fs::read_to_string(file).map_err(|e| format!("read {}: {}", file, e))?;
            let body = serde_json::json!({ "source": source });
            run_request(opts, "POST", "/v1/circuits", Some(body.to_string()))
        }
        "list" => run_request(opts, "GET", "/v1/circuits", None),
        "get" => {
            let id = args.get(1).ok_or("circuits get: missing ID")?;
            run_request(opts, "GET", &format!("/v1/circuits/{}", id), None)
        }
        "delete" => {
            let id = args.get(1).ok_or("circuits delete: missing ID")?;
            run_request(opts, "DELETE", &format!("/v1/circuits/{}", id), None)
        }
        _ => Err("circuits: expected upload|list|get|delete".to_string()),
    }
}

fn cmd_functions(opts: &Opts, args: &[String]) -> Result<(), String> {
    let sub = args.first().map(String::as_str).unwrap_or("");
    match sub {
        "create" => {
            let circuit_id = args.get(1).ok_or("functions create: missing CIRCUIT_ID")?;
            let name = args.get(2).ok_or("functions create: missing NAME")?;
            let mut body = serde_json::json!({
                "circuit_id": circuit_id,
                "name": name,
            });
            if let Some(shots) = parse_kv_u64(&args[3..], "--shots")? {
                body["default_shots"] = serde_json::json!(shots);
            }
            run_request(opts, "POST", "/v1/functions", Some(body.to_string()))
        }
        "list" => run_request(opts, "GET", "/v1/functions", None),
        "get" => {
            let id = args.get(1).ok_or("functions get: missing ID")?;
            run_request(opts, "GET", &format!("/v1/functions/{}", id), None)
        }
        "invoke" => {
            let id = args.get(1).ok_or("functions invoke: missing ID")?;
            let mut body = serde_json::Map::new();
            if let Some(params) = parse_kv_str(&args[2..], "--params")? {
                let xs: Vec<f64> = params
                    .split(',')
                    .map(|s| s.trim().parse::<f64>())
                    .collect::<Result<_, _>>()
                    .map_err(|e| format!("--params: {}", e))?;
                body.insert("params".to_string(), serde_json::json!(xs));
            }
            if let Some(shots) = parse_kv_u64(&args[2..], "--shots")? {
                body.insert("shots".to_string(), serde_json::json!(shots));
            }
            if let Some(seed) = parse_kv_u64(&args[2..], "--seed")? {
                body.insert("seed".to_string(), serde_json::json!(seed));
            }
            run_request(
                opts,
                "POST",
                &format!("/v1/functions/{}/invoke", id),
                Some(serde_json::Value::Object(body).to_string()),
            )
        }
        _ => Err("functions: expected create|list|get|invoke".to_string()),
    }
}

fn cmd_lambdas(opts: &Opts, args: &[String]) -> Result<(), String> {
    let sub = args.first().map(String::as_str).unwrap_or("");
    match sub {
        "register" => {
            let file = args.get(1).ok_or("lambdas register: missing FILE")?;
            let name = args.get(2).ok_or("lambdas register: missing NAME")?;
            let bytes = std::fs::read(file).map_err(|e| format!("read {}: {}", file, e))?;
            let wasm_b64 = base64::engine::general_purpose::STANDARD.encode(&bytes);
            let body = serde_json::json!({
                "name": name,
                "wasm_b64": wasm_b64,
            });
            run_request(opts, "POST", "/v1/lambdas", Some(body.to_string()))
        }
        "list" => run_request(opts, "GET", "/v1/lambdas", None),
        "get" => {
            let id = args.get(1).ok_or("lambdas get: missing ID")?;
            run_request(opts, "GET", &format!("/v1/lambdas/{}", id), None)
        }
        "delete" => {
            let id = args.get(1).ok_or("lambdas delete: missing ID")?;
            run_request(opts, "DELETE", &format!("/v1/lambdas/{}", id), None)
        }
        "invoke" => {
            let id = args.get(1).ok_or("lambdas invoke: missing ID")?;
            // The lambda invoke handler expects `{"input": "<json-string>"}`;
            // the guest's omega_input_read returns the bytes of that string.
            // We always wrap so callers can hand us the raw payload.
            let input_str = if let Some(json_str) = parse_kv_str(&args[2..], "--input")? {
                Some(json_str)
            } else if let Some(path) = parse_kv_str(&args[2..], "--input-file")? {
                Some(std::fs::read_to_string(&path).map_err(|e| format!("read {}: {}", path, e))?)
            } else {
                None
            };
            let body = input_str.map(|s| serde_json::json!({ "input": s }).to_string());
            run_request(opts, "POST", &format!("/v1/lambdas/{}/invoke", id), body)
        }
        _ => Err("lambdas: expected register|list|get|delete|invoke".to_string()),
    }
}

fn cmd_invocation(opts: &Opts, args: &[String]) -> Result<(), String> {
    let sub = args.first().map(String::as_str).unwrap_or("");
    match sub {
        "get" => {
            let id = args.get(1).ok_or("invocation get: missing ID")?;
            run_request(opts, "GET", &format!("/v1/invocations/{}", id), None)
        }
        _ => Err("invocation: expected get".to_string()),
    }
}

fn cmd_raw(opts: &Opts, args: &[String]) -> Result<(), String> {
    let method = args.first().ok_or("raw: missing METHOD")?.to_uppercase();
    let path = args.get(1).ok_or("raw: missing PATH")?;
    let body = parse_kv_str(&args[2..], "--data")?;
    run_request(opts, &method, path, body)
}

// ---- Helpers ----------------------------------------------------------------

fn parse_kv_str(args: &[String], flag: &str) -> Result<Option<String>, String> {
    let prefix = format!("{}=", flag);
    for (i, a) in args.iter().enumerate() {
        if a == flag {
            return Ok(Some(
                args.get(i + 1)
                    .ok_or_else(|| format!("{}: missing value", flag))?
                    .clone(),
            ));
        }
        if let Some(v) = a.strip_prefix(&prefix) {
            return Ok(Some(v.to_string()));
        }
    }
    Ok(None)
}

fn parse_kv_u64(args: &[String], flag: &str) -> Result<Option<u64>, String> {
    parse_kv_str(args, flag)?
        .map(|s| s.parse::<u64>().map_err(|e| format!("{}: {}", flag, e)))
        .transpose()
}

fn print_json(value: &serde_json::Value, raw: bool) {
    if raw {
        println!("{}", value);
    } else {
        match serde_json::to_string_pretty(value) {
            Ok(s) => println!("{}", s),
            Err(_) => println!("{}", value),
        }
    }
}

// ---- HTTP transport ---------------------------------------------------------

fn run_request(opts: &Opts, method: &str, path: &str, body: Option<String>) -> Result<(), String> {
    let (host, port) = parse_server(&opts.server)?;
    let mut headers: BTreeMap<&str, String> = BTreeMap::new();
    headers.insert("Host", format!("{}:{}", host, port));
    headers.insert("User-Agent", USER_AGENT.to_string());
    headers.insert("Accept", "application/json".to_string());
    headers.insert("Connection", "close".to_string());
    if let Some(t) = &opts.token {
        headers.insert("Authorization", format!("Bearer {}", t));
    }
    let body_bytes = body.as_deref().unwrap_or("").as_bytes();
    if body.is_some() {
        headers.insert("Content-Type", "application/json".to_string());
        headers.insert("Content-Length", body_bytes.len().to_string());
    } else {
        headers.insert("Content-Length", "0".to_string());
    }

    let mut request = Vec::new();
    write!(&mut request, "{} {} HTTP/1.1\r\n", method, path).unwrap();
    for (k, v) in &headers {
        write!(&mut request, "{}: {}\r\n", k, v).unwrap();
    }
    request.extend_from_slice(b"\r\n");
    request.extend_from_slice(body_bytes);

    let stream = TcpStream::connect((host.as_str(), port))
        .map_err(|e| format!("connect {}:{}: {}", host, port, e))?;
    stream
        .set_read_timeout(Some(Duration::from_secs(60)))
        .map_err(|e| format!("set_read_timeout: {}", e))?;
    stream
        .set_write_timeout(Some(Duration::from_secs(30)))
        .map_err(|e| format!("set_write_timeout: {}", e))?;
    let mut stream = stream;
    stream
        .write_all(&request)
        .map_err(|e| format!("write: {}", e))?;
    stream.flush().ok();

    let mut response = Vec::new();
    stream
        .read_to_end(&mut response)
        .map_err(|e| format!("read: {}", e))?;

    let (status, resp_body) = parse_response(&response)?;
    print_response(status, &resp_body, opts.raw)?;
    if status >= 400 {
        std::process::exit(1);
    }
    Ok(())
}

fn parse_server(url: &str) -> Result<(String, u16), String> {
    let stripped = url
        .strip_prefix("http://")
        .ok_or_else(|| format!("--server: only http:// is supported, got `{}`", url))?;
    let stripped = stripped.trim_end_matches('/');
    if let Some((host, port)) = stripped.rsplit_once(':') {
        let port: u16 = port
            .parse()
            .map_err(|e| format!("--server: bad port `{}`: {}", port, e))?;
        Ok((host.to_string(), port))
    } else {
        Ok((stripped.to_string(), 80))
    }
}

fn parse_response(buf: &[u8]) -> Result<(u16, Vec<u8>), String> {
    // Locate end-of-headers (\r\n\r\n).
    let split = buf
        .windows(4)
        .position(|w| w == b"\r\n\r\n")
        .ok_or("malformed HTTP response (no header terminator)")?;
    let head =
        std::str::from_utf8(&buf[..split]).map_err(|e| format!("non-UTF-8 headers: {}", e))?;
    let body_start = split + 4;
    let mut lines = head.lines();
    let status_line = lines.next().ok_or("empty HTTP response")?;
    let status: u16 = status_line
        .split_whitespace()
        .nth(1)
        .ok_or("malformed status line")?
        .parse()
        .map_err(|e| format!("bad status code: {}", e))?;

    // Detect Transfer-Encoding: chunked. Most Axum responses use
    // Content-Length + Connection: close so we get a flat body, but
    // chunked is allowed by HTTP/1.1.
    let mut chunked = false;
    for line in lines {
        if line.eq_ignore_ascii_case("transfer-encoding: chunked")
            || line.to_ascii_lowercase().starts_with("transfer-encoding:")
                && line.to_ascii_lowercase().contains("chunked")
        {
            chunked = true;
        }
    }
    let body = if chunked {
        decode_chunked(&buf[body_start..])?
    } else {
        buf[body_start..].to_vec()
    };
    Ok((status, body))
}

fn decode_chunked(buf: &[u8]) -> Result<Vec<u8>, String> {
    let mut out = Vec::new();
    let mut cursor = 0;
    while cursor < buf.len() {
        // Read chunk-size line.
        let nl = buf[cursor..]
            .windows(2)
            .position(|w| w == b"\r\n")
            .ok_or("chunked: missing CRLF after size")?;
        let size_line = std::str::from_utf8(&buf[cursor..cursor + nl])
            .map_err(|e| format!("chunked size line: {}", e))?;
        let size = usize::from_str_radix(size_line.split(';').next().unwrap_or("").trim(), 16)
            .map_err(|e| format!("chunked size parse: {}", e))?;
        cursor += nl + 2;
        if size == 0 {
            break;
        }
        if cursor + size > buf.len() {
            return Err("chunked: truncated body".to_string());
        }
        out.extend_from_slice(&buf[cursor..cursor + size]);
        cursor += size + 2; // trailing \r\n
    }
    Ok(out)
}

fn print_response(status: u16, body: &[u8], raw: bool) -> Result<(), String> {
    let text = std::str::from_utf8(body).unwrap_or("");
    if let Ok(v) = serde_json::from_str::<serde_json::Value>(text) {
        eprintln!("HTTP {}", status);
        print_json(&v, raw);
    } else {
        eprintln!("HTTP {}", status);
        if !text.is_empty() {
            println!("{}", text);
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- parse_server ----

    #[test]
    fn parse_server_with_explicit_port() {
        let (host, port) = parse_server("http://localhost:8080").unwrap();
        assert_eq!(host, "localhost");
        assert_eq!(port, 8080);
    }

    #[test]
    fn parse_server_default_port_when_missing() {
        let (host, port) = parse_server("http://example.com").unwrap();
        assert_eq!(host, "example.com");
        assert_eq!(port, 80);
    }

    #[test]
    fn parse_server_strips_trailing_slash() {
        let (host, port) = parse_server("http://omega:8443/").unwrap();
        assert_eq!(host, "omega");
        assert_eq!(port, 8443);
    }

    #[test]
    fn parse_server_rejects_https_scheme() {
        // The client only does plain HTTP today (TLS via reverse proxy).
        // A user typing https:// should get a clear error rather than
        // silently connecting to plaintext on the wrong port.
        let err = parse_server("https://example.com").expect_err("https rejected");
        assert!(err.contains("only http://"), "msg: {err}");
    }

    #[test]
    fn parse_server_rejects_bad_port() {
        let err = parse_server("http://localhost:notaport").expect_err("bad port");
        assert!(err.contains("bad port"), "msg: {err}");
    }

    // ---- parse_kv_str / parse_kv_u64 ----

    #[test]
    fn parse_kv_str_accepts_separated_form() {
        let args = vec!["--name".to_string(), "alice".to_string()];
        assert_eq!(
            parse_kv_str(&args, "--name").unwrap().as_deref(),
            Some("alice")
        );
    }

    #[test]
    fn parse_kv_str_accepts_equals_form() {
        let args = vec!["--name=bob".to_string()];
        assert_eq!(
            parse_kv_str(&args, "--name").unwrap().as_deref(),
            Some("bob")
        );
    }

    #[test]
    fn parse_kv_str_returns_none_when_absent() {
        let args = vec!["--other".to_string(), "x".to_string()];
        assert_eq!(parse_kv_str(&args, "--name").unwrap(), None);
    }

    #[test]
    fn parse_kv_str_errors_when_separated_value_missing() {
        let args = vec!["--name".to_string()]; // no following value
        let err = parse_kv_str(&args, "--name").expect_err("missing value");
        assert!(err.contains("missing value"), "msg: {err}");
    }

    #[test]
    fn parse_kv_u64_parses_decimal() {
        let args = vec!["--shots=1024".to_string()];
        assert_eq!(parse_kv_u64(&args, "--shots").unwrap(), Some(1024));
    }

    #[test]
    fn parse_kv_u64_errors_on_non_numeric() {
        let args = vec!["--shots=abc".to_string()];
        let err = parse_kv_u64(&args, "--shots").expect_err("non-numeric");
        assert!(err.contains("--shots"), "msg: {err}");
    }

    // ---- parse_response ----

    #[test]
    fn parse_response_extracts_status_and_body() {
        let buf = b"HTTP/1.1 200 OK\r\nContent-Length: 5\r\n\r\nhello";
        let (status, body) = parse_response(buf).unwrap();
        assert_eq!(status, 200);
        assert_eq!(body, b"hello");
    }

    #[test]
    fn parse_response_handles_4xx_5xx() {
        let buf = b"HTTP/1.1 404 Not Found\r\n\r\n";
        let (status, body) = parse_response(buf).unwrap();
        assert_eq!(status, 404);
        assert!(body.is_empty());
    }

    #[test]
    fn parse_response_decodes_chunked_body() {
        // 5\r\nhello\r\n6\r\n world\r\n0\r\n\r\n  → "hello world".
        let buf = b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n5\r\nhello\r\n6\r\n world\r\n0\r\n\r\n";
        let (status, body) = parse_response(buf).unwrap();
        assert_eq!(status, 200);
        assert_eq!(body, b"hello world");
    }

    #[test]
    fn parse_response_rejects_missing_terminator() {
        let buf = b"HTTP/1.1 200 OK\r\nContent-Length: 0\r\n";
        let err = parse_response(buf).expect_err("no body terminator");
        assert!(err.contains("malformed"), "msg: {err}");
    }

    #[test]
    fn parse_response_rejects_bad_status_line() {
        let buf = b"NOT-HTTP\r\n\r\n";
        let err = parse_response(buf).expect_err("bad status");
        assert!(
            err.contains("status line") || err.contains("status code"),
            "msg: {err}"
        );
    }

    // ---- decode_chunked ----

    #[test]
    fn decode_chunked_concatenates_chunks() {
        let buf = b"5\r\nhello\r\n6\r\n world\r\n0\r\n\r\n";
        let body = decode_chunked(buf).unwrap();
        assert_eq!(body, b"hello world");
    }

    #[test]
    fn decode_chunked_handles_empty_first_chunk() {
        // Chunked encoding with no data: "0\r\n\r\n" alone.
        let buf = b"0\r\n\r\n";
        let body = decode_chunked(buf).unwrap();
        assert!(body.is_empty());
    }

    #[test]
    fn decode_chunked_rejects_truncated_body() {
        // Chunk size says 10 bytes but only 3 follow.
        let buf = b"a\r\nabc";
        let err = decode_chunked(buf).expect_err("truncated");
        assert!(err.contains("truncated"), "msg: {err}");
    }

    #[test]
    fn decode_chunked_handles_chunk_extensions() {
        // Chunk-size lines may carry `;ext=val` extensions per RFC 7230.
        // We strip everything after the first `;` before parsing.
        let buf = b"5;name=value\r\nhello\r\n0\r\n\r\n";
        let body = decode_chunked(buf).unwrap();
        assert_eq!(body, b"hello");
    }

    // ---- read_token_file ----

    #[test]
    fn read_token_file_strips_trailing_whitespace() {
        // The bootstrap admin token written via --save-token-to ends
        // with a newline; the loader must strip it cleanly so the
        // Authorization header doesn't become "Bearer eyJ…\n".
        let dir = std::env::temp_dir();
        let path = dir.join(format!(
            "omega-client-test-token-{}.txt",
            std::process::id()
        ));
        std::fs::write(&path, "the-token-bytes\n  \n").unwrap();
        let s = read_token_file(path.to_str().unwrap()).unwrap();
        let _ = std::fs::remove_file(&path);
        assert_eq!(s, "the-token-bytes");
    }

    #[test]
    fn read_token_file_errors_on_missing_path() {
        let err = read_token_file("/no/such/path/please-do-not-exist").expect_err("missing file");
        assert!(err.contains("read"), "msg: {err}");
    }
}
