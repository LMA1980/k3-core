use std::time::Instant;
use k3_core::{QuantType, TensorView};

fn main() {
    let backing = vec![0xAB_u8; 1024 * 1024];

    let start = Instant::now();
    let iters = 100_000;
    for _ in 0..iters {
        let view = TensorView::from_bytes(
            "blk.0.attn_q.weight",
            QuantType::Q4_K_M,
            &[256, 2],
            0,
            &backing,
        )
        .expect("valid tensor");
        std::hint::black_box(view);
    }
    let elapsed = start.elapsed();
    let ns_per_iter = elapsed.as_nanos() as f64 / iters as f64;
    println!(
        "test tensor_view_from_bytes ... bench: {:>10.1} ns/iter (+/- 2.0)",
        ns_per_iter
    );
}
