//! Empirical Adversarial Challenger Test Suite for Milestone 2 (`TQ1_0` and `TQ2_0` in `k3-core`).
//!
//! Focuses on boundary extremes, overflow bounds, block divisibility, and zero-copy slicing.

#![allow(
    clippy::all,
    clippy::pedantic,
    clippy::nursery,
    clippy::undocumented_unsafe_blocks
)]

use k3_core::{K3Error, QuantType, TensorView};

#[test]
fn test_empirical_arithmetic_overflow_extremes() {
    // 1. usize::MAX elements check
    // On 64-bit platforms, usize::MAX % 256 = 255 != 0 -> fails divisibility
    let err_max_tq1 = QuantType::TQ1_0
        .storage_bytes("overflow_max", usize::MAX)
        .unwrap_err();
    assert!(matches!(err_max_tq1, K3Error::TensorBoundsError { .. }));
    assert!(err_max_tq1.to_string().contains("not divisible"));

    let err_max_tq2 = QuantType::TQ2_0
        .storage_bytes("overflow_max", usize::MAX)
        .unwrap_err();
    assert!(matches!(err_max_tq2, K3Error::TensorBoundsError { .. }));
    assert!(err_max_tq2.to_string().contains("not divisible"));

    // 2. Maximum possible divisible element count: max_divisible = usize::MAX - 255.
    // Mathematical invariant proof:
    // For TQ1_0, block_elements = 256 and block_bytes = 54.
    // Since 54 < 256, (num_elements / 256) * 54 <= num_elements * (54 / 256) < num_elements <= usize::MAX.
    // Hence, compressed storage for TQ1_0 and TQ2_0 (66 < 256) never overflows usize for any num_elements: usize!
    let max_divisible = usize::MAX - (usize::MAX % 256);
    assert_eq!(max_divisible % 256, 0);

    let bytes_tq1 = QuantType::TQ1_0
        .storage_bytes("max_divisible_tq1", max_divisible)
        .expect("TQ1_0 compression fits within usize by invariant (54 < 256)");
    let expected_tq1 = (max_divisible / 256) * 54;
    assert_eq!(bytes_tq1, expected_tq1);
    assert_eq!(bytes_tq1, 3_891_110_078_048_108_490);

    let bytes_tq2 = QuantType::TQ2_0
        .storage_bytes("max_divisible_tq2", max_divisible)
        .expect("TQ2_0 compression fits within usize by invariant (66 < 256)");
    let expected_tq2 = (max_divisible / 256) * 66;
    assert_eq!(bytes_tq2, expected_tq2);
    assert_eq!(bytes_tq2, 4_755_801_206_503_243_710);

    // 3. Contrast with formats where block_bytes > block_elements (e.g. F32: 4 > 1, Q8_0: 34 > 32)
    // where checked_mul triggers the byte size overflow branch:
    let err_f32_overflow = QuantType::F32
        .storage_bytes("f32_overflow", usize::MAX)
        .unwrap_err();
    assert!(matches!(
        err_f32_overflow,
        K3Error::TensorBoundsError { .. }
    ));
    assert!(err_f32_overflow.to_string().contains("byte size overflow"));

    let q8_max_divisible = usize::MAX - (usize::MAX % 32);
    let err_q8_overflow = QuantType::Q8_0
        .storage_bytes("q8_overflow", q8_max_divisible)
        .unwrap_err();
    assert!(matches!(err_q8_overflow, K3Error::TensorBoundsError { .. }));
    assert!(err_q8_overflow.to_string().contains("byte size overflow"));
}

#[test]
fn test_empirical_tensor_view_bounds_offset_overflow() {
    let dummy_buffer = vec![0u8; 1024];

    // 1. offset = usize::MAX - 10 with valid shape [256] (byte_len = 54 or 66)
    // offset + byte_len overflows usize
    let err_offset_tq1 = TensorView::calculate_bounds(
        "offset_overflow_tq1",
        QuantType::TQ1_0,
        &[256],
        usize::MAX - 10,
        usize::MAX,
    )
    .unwrap_err();
    assert!(matches!(err_offset_tq1, K3Error::TensorBoundsError { .. }));
    assert!(err_offset_tq1.to_string().contains("overflowed usize"));

    let err_offset_tq2 = TensorView::calculate_bounds(
        "offset_overflow_tq2",
        QuantType::TQ2_0,
        &[256],
        usize::MAX - 10,
        usize::MAX,
    )
    .unwrap_err();
    assert!(matches!(err_offset_tq2, K3Error::TensorBoundsError { .. }));
    assert!(err_offset_tq2.to_string().contains("overflowed usize"));

    // 2. offset = usize::MAX
    let err_offset_max = TensorView::calculate_bounds(
        "offset_max",
        QuantType::TQ1_0,
        &[256],
        usize::MAX,
        usize::MAX,
    )
    .unwrap_err();
    assert!(matches!(err_offset_max, K3Error::TensorBoundsError { .. }));
    assert!(err_offset_max.to_string().contains("overflowed usize"));

    // 3. from_bytes propagates offset overflow error safely
    let err_from_bytes = TensorView::from_bytes(
        "from_bytes_overflow",
        QuantType::TQ1_0,
        &[256],
        usize::MAX - 10,
        &dummy_buffer,
    )
    .unwrap_err();
    assert!(matches!(err_from_bytes, K3Error::TensorBoundsError { .. }));
    assert!(err_from_bytes.to_string().contains("overflowed usize"));
}

#[test]
fn test_empirical_shape_product_overflow() {
    // 1. Product of shape dimensions overflows usize
    let err_shape_overflow_2d = TensorView::calculate_bounds(
        "shape_overflow_2d",
        QuantType::TQ1_0,
        &[usize::MAX / 2 + 1, 2],
        0,
        1024,
    )
    .unwrap_err();
    assert!(matches!(
        err_shape_overflow_2d,
        K3Error::TensorBoundsError { .. }
    ));
    assert!(
        err_shape_overflow_2d
            .to_string()
            .contains("element product overflowed usize")
    );

    // 2. High-dimensional shape product overflow
    let err_shape_overflow_hd = TensorView::calculate_bounds(
        "shape_overflow_hd",
        QuantType::TQ2_0,
        &[100_000, 100_000, 100_000, 100_000],
        0,
        1024,
    )
    .unwrap_err();
    assert!(matches!(
        err_shape_overflow_hd,
        K3Error::TensorBoundsError { .. }
    ));
    assert!(
        err_shape_overflow_hd
            .to_string()
            .contains("element product overflowed usize")
    );

    // 3. Single dimension = usize::MAX
    let err_single_max =
        TensorView::calculate_bounds("single_max", QuantType::TQ1_0, &[usize::MAX], 0, usize::MAX)
            .unwrap_err();
    // usize::MAX % 256 != 0 -> fails divisibility
    assert!(matches!(err_single_max, K3Error::TensorBoundsError { .. }));
}

#[test]
fn test_empirical_divisibility_corner_cases() {
    // 1. elements = 0
    let err_zero_tq1 = QuantType::TQ1_0.storage_bytes("zero", 0).unwrap_err();
    assert!(matches!(err_zero_tq1, K3Error::TensorBoundsError { .. }));
    assert!(
        err_zero_tq1
            .to_string()
            .contains("element count must be greater than zero")
    );

    let err_zero_tq2 = QuantType::TQ2_0.storage_bytes("zero", 0).unwrap_err();
    assert!(matches!(err_zero_tq2, K3Error::TensorBoundsError { .. }));
    assert!(
        err_zero_tq2
            .to_string()
            .contains("element count must be greater than zero")
    );

    // Empty shape and zero-dimension shapes
    let err_empty_shape =
        TensorView::calculate_bounds("empty", QuantType::TQ1_0, &[], 0, 1024).unwrap_err();
    assert!(matches!(err_empty_shape, K3Error::TensorBoundsError { .. }));
    assert!(
        err_empty_shape
            .to_string()
            .contains("at least one dimension")
    );

    let err_zero_dim =
        TensorView::calculate_bounds("zero_dim", QuantType::TQ1_0, &[256, 0], 0, 1024).unwrap_err();
    assert!(matches!(err_zero_dim, K3Error::TensorBoundsError { .. }));
    assert!(err_zero_dim.to_string().contains("must be non-zero"));

    // 2. elements = 255 (1 element short of 1 block)
    let err_255_tq1 = QuantType::TQ1_0.storage_bytes("255", 255).unwrap_err();
    assert!(matches!(err_255_tq1, K3Error::TensorBoundsError { .. }));
    assert!(
        err_255_tq1
            .to_string()
            .contains("not divisible by TQ1_0 block size 256")
    );

    let err_255_tq2 = QuantType::TQ2_0.storage_bytes("255", 255).unwrap_err();
    assert!(matches!(err_255_tq2, K3Error::TensorBoundsError { .. }));
    assert!(
        err_255_tq2
            .to_string()
            .contains("not divisible by TQ2_0 block size 256")
    );

    // 3. elements = 256 (exact 1 block)
    assert_eq!(QuantType::TQ1_0.storage_bytes("256", 256).unwrap(), 54);
    assert_eq!(QuantType::TQ2_0.storage_bytes("256", 256).unwrap(), 66);

    // 4. elements = 256 * 1024 * 1024 (256M elements = 1,048,576 blocks)
    let count_256m = 256 * 1024 * 1024;
    let expected_tq1_bytes = 1024 * 1024 * 54; // 56,623,104
    let expected_tq2_bytes = 1024 * 1024 * 66; // 69,206,016
    assert_eq!(
        QuantType::TQ1_0.storage_bytes("256m", count_256m).unwrap(),
        expected_tq1_bytes
    );
    assert_eq!(
        QuantType::TQ2_0.storage_bytes("256m", count_256m).unwrap(),
        expected_tq2_bytes
    );

    // 5. Divisibility fuzzing across multiples of 256 with deltas [-10..=10]
    for k in 1..=50 {
        let base = 256 * k;
        for delta in -10..=10 {
            let n = (base as isize + delta) as usize;
            if delta == 0 {
                assert_eq!(QuantType::TQ1_0.storage_bytes("fuzz", n).unwrap(), k * 54);
                assert_eq!(QuantType::TQ2_0.storage_bytes("fuzz", n).unwrap(), k * 66);
            } else {
                assert!(QuantType::TQ1_0.storage_bytes("fuzz", n).is_err());
                assert!(QuantType::TQ2_0.storage_bytes("fuzz", n).is_err());
            }
        }
    }
}

#[test]
fn test_empirical_slice_rows_alignment_and_bounds() {
    // 1. Non-2D shapes rejected by slice_rows
    let backing = vec![0u8; 1000];
    let view_1d = TensorView::from_bytes("view_1d", QuantType::TQ1_0, &[256], 0, &backing).unwrap();
    let err_1d = view_1d.slice_rows(0, 1).unwrap_err();
    assert!(matches!(err_1d, K3Error::TensorBoundsError { .. }));
    assert!(err_1d.to_string().contains("requires exactly 2 dimensions"));

    let view_3d =
        TensorView::from_bytes("view_3d", QuantType::TQ1_0, &[256, 2, 2], 0, &backing).unwrap();
    let err_3d = view_3d.slice_rows(0, 1).unwrap_err();
    assert!(matches!(err_3d, K3Error::TensorBoundsError { .. }));
    assert!(err_3d.to_string().contains("requires exactly 2 dimensions"));

    // 2. Column count unaligned to 256
    // shape [128, 4] has 512 total elements (divisible by 256), so view creation succeeds!
    // But cols = 128 is not divisible by block_elements (256).
    let view_unaligned_128 =
        TensorView::from_bytes("unaligned_128", QuantType::TQ1_0, &[128, 4], 0, &backing).unwrap();
    let err_unaligned_128 = view_unaligned_128.slice_rows(0, 1).unwrap_err();
    assert!(matches!(
        err_unaligned_128,
        K3Error::TensorBoundsError { .. }
    ));
    assert!(
        err_unaligned_128
            .to_string()
            .contains("row length 128 is not divisible by TQ1_0 block size 256")
    );

    // shape [64, 8] also has 512 total elements
    let view_unaligned_64 =
        TensorView::from_bytes("unaligned_64", QuantType::TQ2_0, &[64, 8], 0, &backing).unwrap();
    let err_unaligned_64 = view_unaligned_64.slice_rows(0, 1).unwrap_err();
    assert!(matches!(
        err_unaligned_64,
        K3Error::TensorBoundsError { .. }
    ));
    assert!(
        err_unaligned_64
            .to_string()
            .contains("row length 64 is not divisible by TQ2_0 block size 256")
    );

    // 3. Valid 2D matrix view: [cols = 256, rows = 4]
    // Total elements = 1024 = 4 blocks
    // TQ1_0: 4 blocks * 54 bytes = 216 bytes
    let backing_valid = vec![0u8; 512];
    let view_valid_tq1 =
        TensorView::from_bytes("matrix_tq1", QuantType::TQ1_0, &[256, 4], 0, &backing_valid)
            .unwrap();

    // 3a. row_count = 0 (empty row slice)
    // Slicing 0 rows at row_start = 0
    let empty_slice_start = view_valid_tq1.slice_rows(0, 0).unwrap();
    assert_eq!(empty_slice_start.shape(), &[256, 0]);
    assert_eq!(empty_slice_start.num_elements(), 0);
    assert_eq!(empty_slice_start.byte_len(), 0);
    assert_eq!(empty_slice_start.offset(), 0);

    // Slicing 0 rows at row_start = total_rows (4)
    let empty_slice_end = view_valid_tq1.slice_rows(4, 0).unwrap();
    assert_eq!(empty_slice_end.shape(), &[256, 0]);
    assert_eq!(empty_slice_end.num_elements(), 0);
    assert_eq!(empty_slice_end.byte_len(), 0);
    assert_eq!(empty_slice_end.offset(), 4 * 54);

    // Slicing 0 rows past total_rows (5)
    let err_zero_past = view_valid_tq1.slice_rows(5, 0).unwrap_err();
    assert!(matches!(err_zero_past, K3Error::TensorBoundsError { .. }));
    assert!(
        err_zero_past
            .to_string()
            .contains("exceeds tensor row count 4")
    );

    // 3b. Out-of-bounds row_start
    let err_oob_start = view_valid_tq1.slice_rows(4, 1).unwrap_err();
    assert!(matches!(err_oob_start, K3Error::TensorBoundsError { .. }));
    assert!(
        err_oob_start
            .to_string()
            .contains("exceeds tensor row count 4")
    );

    let err_oob_far = view_valid_tq1.slice_rows(10, 1).unwrap_err();
    assert!(matches!(err_oob_far, K3Error::TensorBoundsError { .. }));

    // 3c. row_start + row_count overflow
    let err_overflow_rows = view_valid_tq1.slice_rows(usize::MAX, 1).unwrap_err();
    assert!(matches!(
        err_overflow_rows,
        K3Error::TensorBoundsError { .. }
    ));
    assert!(
        err_overflow_rows
            .to_string()
            .contains("row_start + row_count overflowed usize")
    );

    // 4. Valid non-empty row slicing
    let slice_0_2 = view_valid_tq1.slice_rows(0, 2).unwrap();
    assert_eq!(slice_0_2.shape(), &[256, 2]);
    assert_eq!(slice_0_2.num_elements(), 512);
    assert_eq!(slice_0_2.byte_len(), 108);
    assert_eq!(slice_0_2.offset(), 0);

    let slice_2_2 = view_valid_tq1.slice_rows(2, 2).unwrap();
    assert_eq!(slice_2_2.shape(), &[256, 2]);
    assert_eq!(slice_2_2.num_elements(), 512);
    assert_eq!(slice_2_2.byte_len(), 108);
    assert_eq!(slice_2_2.offset(), 108);
}
