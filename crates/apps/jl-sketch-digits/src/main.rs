use std::process::ExitCode;
fn main() -> ExitCode {
    aria_verify_core::run_main("jl_sketch_digits", aria_app_jl_sketch_digits::run)
}
