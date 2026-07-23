use std::process::ExitCode;
fn main() -> ExitCode {
    aria_verify_core::run_main("spectra", aria_app_spectra::run)
}
