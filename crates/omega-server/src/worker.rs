//! Async worker for circuit execution.

// Worker execution is handled inline in routes for now.
// Future: use tokio::spawn_blocking with a semaphore for concurrent simulations.
