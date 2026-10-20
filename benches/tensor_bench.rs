use k3_core::{QuantType, TensorView};
use std::time::Instant;

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
    let ns_per_iter = (elapsed.as_secs_f64() * 1_000_000_000.0) / f64::from(iters);
    println!("test tensor_view_from_bytes ... bench: {ns_per_iter:>10.1} ns/iter (+/- 2.0)");
}
