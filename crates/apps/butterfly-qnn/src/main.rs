use std::process::ExitCode;
fn main() -> ExitCode {
    aria_verify_core::run_main("butterfly_qnn", aria_app_butterfly_qnn::run)
}
