//! Empirical Challenger Test Suite for Milestone 2: `TQ1_0` & `TQ2_0` QuantType Descriptors,
//! Bounds Verification, and Adversarial Fuzzing.
//!
//! Objectives tested:
//! 1. Exhaustive GGUF type ID mapping roundtrips and boundary probes.
//! 2. Non-divisible element counts (e.g. 1, 255, 257, 511, 1023, 1025) rejecting cleanly with
//!    exact `TensorBoundsError` and informative error messages.
//! 3. Exact mathematical byte calculations across 1..=1000 block multiples and overflow bounds.
//! 4. Zero-scale, empty buffer, and all-0xFF buffer handling.
//! 5. 2D `TensorView::slice_rows` with valid, zero-row, out-of-range, non-2D, and unaligned shapes.
//! 6. GGUF v3 parsing with valid and corrupt `TQ1_0` / `TQ2_0` headers.

#![allow(
    clippy::all,
    clippy::pedantic,
    clippy::nursery,
    clippy::undocumented_unsafe_blocks
)]

use std::collections::{HashMap, HashSet};

use k3_core::test_helpers::create_synthetic_gguf_v3;
use k3_core::{GgufFile, K3Error, ModelWeights, QuantType, TensorView};

// ============================================================================
// 1. Exhaustive GGUF Type ID Mapping and Roundtrip Verification
// ============================================================================

#[test]
fn test_adversarial_gguf_type_id_mapping_exhaustive() {
    // 1.1 All known canonical variants must map bi-directionally
    let canonical_types = [
        (QuantType::F32, 0, "F32", 1, 4),
        (QuantType::F16, 1, "F16", 1, 2),
        (QuantType::Q4_0, 2, "Q4_0", 32, 18),
        (QuantType::Q8_0, 8, "Q8_0", 32, 34),
        (QuantType::Q4_K_M, 12, "Q4_K_M", 256, 144),
        (QuantType::Q5_K, 13, "Q5_K", 256, 176),
        (QuantType::Q6_K, 14, "Q6_K", 256, 210),
        (QuantType::BF16, 30, "BF16", 1, 2),
        (QuantType::TQ1_0, 34, "TQ1_0", 256, 54),
        (QuantType::TQ2_0, 35, "TQ2_0", 256, 66),
    ];

    let mut recognized_ids = HashSet::new();

    for (quant, type_id, name, block_elems, block_bytes) in canonical_types {
        recognized_ids.insert(type_id);

        // Name and Display conformance
        assert_eq!(quant.as_str(), name);
        assert_eq!(quant.type_name(), name);
        assert_eq!(format!("{quant}"), name);

        // Block properties
        assert_eq!(quant.block_elements(), block_elems);
        assert_eq!(quant.block_bytes(), block_bytes);

        // GGUF type ID mapping roundtrip
        assert_eq!(quant.ggml_type_id(), type_id);
        assert_eq!(QuantType::from_gguf_type_id(type_id), Some(quant));
    }

    // 1.2 Exhaustive sweep of all u16 values (0..=65535)
    // ONLY the 9 registered IDs must return Some, all other 65527 IDs must return None.
    for id in 0..=65535_u32 {
        let mapped = QuantType::from_gguf_type_id(id);
        if recognized_ids.contains(&id) {
            assert!(
                mapped.is_some(),
                "Expected registered ID {id} to map to Some(QuantType)"
            );
        } else {
            assert!(
                mapped.is_none(),
                "Unregistered ID {id} unexpectedly mapped to {mapped:?}"
            );
        }
    }

    // 1.3 Boundary extreme u32 probes
    let extreme_probes = [
        65536,
        100_000,
        1_000_000,
        (1 << 31) - 1,
        1 << 31,
        u32::MAX / 2,
        u32::MAX - 1,
        u32::MAX,
    ];
    for &id in &extreme_probes {
        assert_eq!(
            QuantType::from_gguf_type_id(id),
            None,
            "Extreme ID {id} must not map to any QuantType"
        );
    }

    // 1.4 Trait implementations: Clone, Copy, PartialEq, Eq, Hash
    let mut map = HashMap::new();
    map.insert(QuantType::TQ1_0, 34_u32);
    map.insert(QuantType::TQ2_0, 35_u32);
    assert_eq!(map.get(&QuantType::TQ1_0), Some(&34));
    assert_eq!(map.get(&QuantType::TQ2_0), Some(&35));
    assert_ne!(QuantType::TQ1_0, QuantType::TQ2_0);
}

// ============================================================================
// 2. Non-Divisible Element Counts Rejection
// ============================================================================

#[test]
fn test_adversarial_non_divisible_element_counts_rejection() {
    // Specifically requested test counts + primes + near-block boundaries
    let non_divisible_counts = [
        1, 2, 3, 7, 15, 31, 63, 127, 128, 129, 254, 255,  // 1 block - 1
        257,  // 1 block + 1
        263,  // prime
        511,  // 2 blocks - 1
        513,  // 2 blocks + 1
        521,  // prime
        767,  // 3 blocks - 1
        769,  // 3 blocks + 1
        1023, // 4 blocks - 1
        1025, // 4 blocks + 1
        1031, // prime
        2047, // 8 blocks - 1
        2049, // 8 blocks + 1
    ];

    for &count in &non_divisible_counts {
        // TQ1_0
        let err_tq1 = QuantType::TQ1_0
            .storage_bytes("adversarial_tq1", count)
            .expect_err(&format!("TQ1_0 must reject non-divisible count {count}"));
        match err_tq1 {
            K3Error::TensorBoundsError {
                tensor_name,
                offset,
                byte_len,
                buffer_len,
                reason,
            } => {
                assert_eq!(tensor_name, "adversarial_tq1");
                assert_eq!(offset, 0);
                assert_eq!(byte_len, 0);
                assert_eq!(buffer_len, 0);
                assert_eq!(
                    reason,
                    format!("element count {count} is not divisible by TQ1_0 block size 256")
                );
            }
            other => panic!("Expected TensorBoundsError for TQ1_0 with {count}, got: {other:?}"),
        }

        // TQ2_0
        let err_tq2 = QuantType::TQ2_0
            .storage_bytes("adversarial_tq2", count)
            .expect_err(&format!("TQ2_0 must reject non-divisible count {count}"));
        match err_tq2 {
            K3Error::TensorBoundsError {
                tensor_name,
                offset,
                byte_len,
                buffer_len,
                reason,
            } => {
                assert_eq!(tensor_name, "adversarial_tq2");
                assert_eq!(offset, 0);
                assert_eq!(byte_len, 0);
                assert_eq!(buffer_len, 0);
                assert_eq!(
                    reason,
                    format!("element count {count} is not divisible by TQ2_0 block size 256")
                );
            }
            other => panic!("Expected TensorBoundsError for TQ2_0 with {count}, got: {other:?}"),
        }
    }

    // Zero elements must return informative error
    let err_zero = QuantType::TQ1_0
        .storage_bytes("zero_tensor", 0)
        .expect_err("0 elements must be rejected");
    match err_zero {
        K3Error::TensorBoundsError { reason, .. } => {
            assert_eq!(reason, "tensor element count must be greater than zero");
        }
        other => panic!("Expected TensorBoundsError for 0 elements, got {other:?}"),
    }
}

// ============================================================================
// 3. Exact Mathematical Byte Calculations Across Block Multiples (1..=1000)
// ============================================================================

#[test]
fn test_adversarial_mathematical_byte_calculations_sweep_and_overflow() {
    // 3.1 Sweep 1 to 1000 blocks
    for n_blocks in 1..=1000 {
        let num_elements = n_blocks * 256;

        let tq1_bytes = QuantType::TQ1_0
            .storage_bytes("tq1_sweep", num_elements)
            .expect("valid block multiple for TQ1_0");
        let expected_tq1 = n_blocks * 54;
        assert_eq!(
            tq1_bytes, expected_tq1,
            "TQ1_0 byte footprint mismatch at {n_blocks} blocks"
        );

        let tq2_bytes = QuantType::TQ2_0
            .storage_bytes("tq2_sweep", num_elements)
            .expect("valid block multiple for TQ2_0");
        let expected_tq2 = n_blocks * 66;
        assert_eq!(
            tq2_bytes, expected_tq2,
            "TQ2_0 byte footprint mismatch at {n_blocks} blocks"
        );

        // Verification of bits per weight
        let tq1_bpw = (tq1_bytes * 8) as f64 / num_elements as f64;
        assert!(
            (tq1_bpw - 1.6875).abs() < 1e-10,
            "TQ1_0 bpw drift at {n_blocks} blocks"
        );

        let tq2_bpw = (tq2_bytes * 8) as f64 / num_elements as f64;
        assert!(
            (tq2_bpw - 2.0625).abs() < 1e-10,
            "TQ2_0 bpw drift at {n_blocks} blocks"
        );
    }

    // 3.2 Multi-dimensional shapes producing exact block multiples
    let test_shapes = [
        (vec![256], 1),
        (vec![16, 16], 1),
        (vec![256, 4], 4),
        (vec![64, 4, 10], 10),
        (vec![1, 1, 256, 1], 1),
        (vec![128, 2, 8], 8),
    ];

    let buffer = vec![0xAA_u8; 10000];
    for (shape, blocks) in test_shapes {
        let view_tq1 =
            TensorView::from_bytes("md_tq1", QuantType::TQ1_0, &shape, 0, &buffer).unwrap();
        assert_eq!(view_tq1.num_elements(), blocks * 256);
        assert_eq!(view_tq1.byte_len(), blocks * 54);

        let view_tq2 =
            TensorView::from_bytes("md_tq2", QuantType::TQ2_0, &shape, 0, &buffer).unwrap();
        assert_eq!(view_tq2.num_elements(), blocks * 256);
        assert_eq!(view_tq2.byte_len(), blocks * 66);
    }

    // 3.3 Shape element product overflow handling
    let overflow_shape = [usize::MAX / 2 + 1, 2];
    let err_shape_overflow = TensorView::from_bytes(
        "overflow_shape",
        QuantType::TQ1_0,
        &overflow_shape,
        0,
        &buffer,
    )
    .expect_err("product of dimensions exceeds usize::MAX");
    match err_shape_overflow {
        K3Error::TensorBoundsError { reason, .. } => {
            assert!(reason.contains("overflowed usize"));
        }
        other => panic!("Expected TensorBoundsError for shape overflow, got {other:?}"),
    }

    // 3.4 Offset + byte_len overflow handling
    let err_offset_overflow = TensorView::from_bytes(
        "overflow_offset",
        QuantType::TQ1_0,
        &[256],
        usize::MAX - 20,
        &buffer,
    )
    .expect_err("offset + 54 overflows usize");
    match err_offset_overflow {
        K3Error::TensorBoundsError { reason, .. } => {
            assert!(reason.contains("overflowed usize"));
        }
        other => panic!("Expected TensorBoundsError for offset overflow, got {other:?}"),
    }
}

// ============================================================================
// 4. Zero-Scale, Empty Buffer, and Adverse Buffer Handling
// ============================================================================

#[test]
fn test_adversarial_zero_scale_and_empty_buffer_handling() {
    // 4.1 Empty backing buffer (len = 0)
    let empty_buf: [u8; 0] = [];
    let err_empty_buf =
        TensorView::from_bytes("empty_buf", QuantType::TQ1_0, &[256], 0, &empty_buf)
            .expect_err("cannot create view into 0-byte buffer");
    match err_empty_buf {
        K3Error::TensorBoundsError {
            byte_len,
            buffer_len,
            reason,
            ..
        } => {
            assert_eq!(byte_len, 54);
            assert_eq!(buffer_len, 0);
            assert!(reason.contains("exceeds backing buffer length 0"));
        }
        other => panic!("Expected TensorBoundsError, got {other:?}"),
    }

    // 4.2 Empty shape &[ ] with empty buffer
    let err_empty_shape =
        TensorView::from_bytes("empty_shape", QuantType::TQ1_0, &[], 0, &empty_buf)
            .expect_err("cannot create view with empty shape");
    match err_empty_shape {
        K3Error::TensorBoundsError { reason, .. } => {
            assert!(reason.contains("tensor shape must contain at least one dimension"));
        }
        other => panic!("Expected TensorBoundsError, got {other:?}"),
    }

    // 4.3 Zero dimension in shape
    let err_zero_dim =
        TensorView::from_bytes("zero_dim", QuantType::TQ1_0, &[256, 0], 0, &empty_buf)
            .expect_err("cannot create view with zero dimension");
    match err_zero_dim {
        K3Error::TensorBoundsError { reason, .. } => {
            assert!(reason.contains("tensor shape dimensions must be non-zero"));
        }
        other => panic!("Expected TensorBoundsError, got {other:?}"),
    }

    // 4.4 Truncated buffers by exactly 1 byte
    let buf_53 = vec![0x00_u8; 53]; // 53 < 54 (TQ1_0 requires 54)
    let err_tq1_trunc = TensorView::from_bytes("trunc_tq1", QuantType::TQ1_0, &[256], 0, &buf_53)
        .expect_err("53 bytes is insufficient for TQ1_0");
    assert!(matches!(err_tq1_trunc, K3Error::TensorBoundsError { .. }));

    let buf_65 = vec![0x00_u8; 65]; // 65 < 66 (TQ2_0 requires 66)
    let err_tq2_trunc = TensorView::from_bytes("trunc_tq2", QuantType::TQ2_0, &[256], 0, &buf_65)
        .expect_err("65 bytes is insufficient for TQ2_0");
    assert!(matches!(err_tq2_trunc, K3Error::TensorBoundsError { .. }));

    // 4.5 Off-by-one offset boundary
    let exact_tq1_buf = vec![0x77_u8; 54];
    assert!(
        TensorView::from_bytes("exact_tq1", QuantType::TQ1_0, &[256], 0, &exact_tq1_buf).is_ok()
    );
    let err_oob_1b = TensorView::from_bytes("oob_tq1", QuantType::TQ1_0, &[256], 1, &exact_tq1_buf)
        .expect_err("offset 1 exceeds 54-byte buffer");
    assert!(matches!(err_oob_1b, K3Error::TensorBoundsError { .. }));

    // 4.6 Zero-scale buffer (all 0x00 bytes)
    // TQ1_0 & TQ2_0 have FP16 scale at first 2 bytes. 0x0000 = +0.0 scale.
    // Creating views and querying byte data must not panic or divide by zero.
    let zero_scale_buf = vec![0x00_u8; 66];
    let view_zero_tq1 = TensorView::from_bytes(
        "zero_scale_tq1",
        QuantType::TQ1_0,
        &[256],
        0,
        &zero_scale_buf,
    )
    .expect("all-zero scale buffer is valid");
    assert_eq!(view_zero_tq1.byte_len(), 54);
    assert!(view_zero_tq1.as_bytes().iter().all(|&b| b == 0));

    let view_zero_tq2 = TensorView::from_bytes(
        "zero_scale_tq2",
        QuantType::TQ2_0,
        &[256],
        0,
        &zero_scale_buf,
    )
    .expect("all-zero scale buffer is valid");
    assert_eq!(view_zero_tq2.byte_len(), 66);
    assert!(view_zero_tq2.as_bytes().iter().all(|&b| b == 0));

    // 4.7 Max exponent / NaN scale buffer (all 0xFF bytes)
    let all_ff_buf = vec![0xFF_u8; 66];
    let view_ff_tq1 =
        TensorView::from_bytes("all_ff_tq1", QuantType::TQ1_0, &[256], 0, &all_ff_buf)
            .expect("all-0xFF buffer is valid");
    assert_eq!(view_ff_tq1.byte_len(), 54);
    assert!(view_ff_tq1.as_bytes().iter().all(|&b| b == 0xFF));

    let view_ff_tq2 =
        TensorView::from_bytes("all_ff_tq2", QuantType::TQ2_0, &[256], 0, &all_ff_buf)
            .expect("all-0xFF buffer is valid");
    assert_eq!(view_ff_tq2.byte_len(), 66);
    assert!(view_ff_tq2.as_bytes().iter().all(|&b| b == 0xFF));
}

// ============================================================================
// 5. 2D TensorView::slice_rows Slicing Stress and Boundary Testing
// ============================================================================

#[test]
fn test_adversarial_tensor_view_slice_rows_comprehensive() {
    // 5.1 Valid multi-row slicing on TQ1_0 [cols=256, rows=16]
    // 16 rows * 54 bytes = 864 bytes total
    let mut matrix_buf_tq1 = vec![0u8; 864];
    for r in 0..16 {
        matrix_buf_tq1[r * 54] = r as u8;
    }

    let view_tq1 = TensorView::from_bytes(
        "matrix_tq1",
        QuantType::TQ1_0,
        &[256, 16],
        0,
        &matrix_buf_tq1,
    )
    .expect("valid 2D TQ1_0 matrix");

    // Slicing sweeps:
    // (a) Single row slices
    for r in 0..16 {
        let row_slice = view_tq1.slice_rows(r, 1).expect("slice 1 row succeeds");
        assert_eq!(row_slice.shape(), &[256, 1]);
        assert_eq!(row_slice.num_elements(), 256);
        assert_eq!(row_slice.offset(), r * 54);
        assert_eq!(row_slice.byte_len(), 54);
        assert_eq!(row_slice.as_bytes()[0], r as u8);
    }

    // (b) 0-row slice at every boundary
    for r in 0..=16 {
        let zero_slice = view_tq1.slice_rows(r, 0).expect("0-row slice succeeds");
        assert_eq!(zero_slice.shape(), &[256, 0]);
        assert_eq!(zero_slice.num_elements(), 0);
        assert_eq!(zero_slice.offset(), r * 54);
        assert_eq!(zero_slice.byte_len(), 0);
    }

    // (c) Arbitrary windows
    let win_slice = view_tq1.slice_rows(4, 8).expect("slice 8 rows from row 4");
    assert_eq!(win_slice.shape(), &[256, 8]);
    assert_eq!(win_slice.num_elements(), 8 * 256);
    assert_eq!(win_slice.offset(), 4 * 54);
    assert_eq!(win_slice.byte_len(), 8 * 54);
    assert_eq!(win_slice.as_bytes()[0], 4);

    // 5.2 Out-of-range slice rejections on TQ1_0
    let oob_cases = [
        (16, 1),
        (15, 2),
        (10, 7),
        (100, 1),
        (usize::MAX, 1),
        (1, usize::MAX),
    ];
    for (start, count) in oob_cases {
        let err = view_tq1
            .slice_rows(start, count)
            .expect_err(&format!("slice_rows({start}, {count}) must fail"));
        assert!(matches!(err, K3Error::TensorBoundsError { .. }));
    }

    // 5.3 Valid multi-row slicing on TQ2_0 [cols=512, rows=8]
    // cols=512 is 2 blocks = 132 bytes per row.
    // 8 rows * 132 bytes = 1056 bytes total
    let mut matrix_buf_tq2 = vec![0u8; 1056];
    for r in 0..8 {
        matrix_buf_tq2[r * 132] = (r * 10) as u8;
    }
    let view_tq2 = TensorView::from_bytes(
        "matrix_tq2",
        QuantType::TQ2_0,
        &[512, 8],
        0,
        &matrix_buf_tq2,
    )
    .expect("valid 2D TQ2_0 matrix");

    let slice_tq2 = view_tq2.slice_rows(2, 3).expect("slice 3 rows");
    assert_eq!(slice_tq2.shape(), &[512, 3]);
    assert_eq!(slice_tq2.num_elements(), 3 * 512);
    assert_eq!(slice_tq2.offset(), 2 * 132);
    assert_eq!(slice_tq2.byte_len(), 3 * 132);
    assert_eq!(slice_tq2.as_bytes()[0], 20);

    // 5.4 Shape Rank Rejection: slice_rows requires strictly 2D
    let buf_1d = vec![0u8; 54];
    let view_1d = TensorView::from_bytes("v1d", QuantType::TQ1_0, &[256], 0, &buf_1d).unwrap();
    let err_1d = view_1d
        .slice_rows(0, 1)
        .expect_err("1D tensor must reject slice_rows");
    match err_1d {
        K3Error::TensorBoundsError { reason, .. } => {
            assert!(reason.contains("requires exactly 2 dimensions [cols, rows]"));
        }
        other => panic!("Expected TensorBoundsError, got {other:?}"),
    }

    let buf_3d = vec![0u8; 216];
    let view_3d =
        TensorView::from_bytes("v3d", QuantType::TQ1_0, &[256, 2, 2], 0, &buf_3d).unwrap();
    let err_3d = view_3d
        .slice_rows(0, 1)
        .expect_err("3D tensor must reject slice_rows");
    match err_3d {
        K3Error::TensorBoundsError { reason, .. } => {
            assert!(reason.contains("requires exactly 2 dimensions [cols, rows]"));
        }
        other => panic!("Expected TensorBoundsError, got {other:?}"),
    }

    // 5.5 Column block alignment rejection:
    // Shape [128, 4] has 512 total elements (divisible by 256), so view creation succeeds!
    // But row length (cols = 128) is not divisible by 256!
    let buf_unaligned_cols = vec![0u8; 108]; // 512 elements = 2 blocks = 108 bytes
    let view_unaligned = TensorView::from_bytes(
        "unaligned_cols",
        QuantType::TQ1_0,
        &[128, 4],
        0,
        &buf_unaligned_cols,
    )
    .expect("from_bytes succeeds for 512 elements");

    let err_unaligned_cols = view_unaligned
        .slice_rows(0, 1)
        .expect_err("slice_rows must reject row length 128");
    match err_unaligned_cols {
        K3Error::TensorBoundsError { reason, .. } => {
            assert_eq!(
                reason,
                "tensor row length 128 is not divisible by TQ1_0 block size 256"
            );
        }
        other => panic!("Expected TensorBoundsError, got {other:?}"),
    }
}

// ============================================================================
// 6. GGUF v3 Parsing with Valid & Adversarially Corrupt Headers
// ============================================================================

#[test]
fn test_adversarial_gguf_v3_parsing_and_corruptions() {
    // 6.1 Synthesize a valid multi-tensor GGUF v3 file containing TQ1_0 and TQ2_0
    let specs = [
        (
            "model.layers.0.attn_q.weight",
            QuantType::TQ1_0,
            &[256, 4][..],
        ), // 1024 elems = 4 blk = 216B
        (
            "model.layers.0.attn_k.weight",
            QuantType::TQ2_0,
            &[256, 2][..],
        ), // 512 elems = 2 blk = 132B
        (
            "model.layers.0.attn_v.weight",
            QuantType::TQ1_0,
            &[512, 2][..],
        ), // 1024 elems = 4 blk = 216B
        ("model.norm.weight", QuantType::F32, &[32][..]), // 32 elems = 128B
    ];

    let valid_bytes = create_synthetic_gguf_v3("bitnet-b1.58", 32, &specs);
    let gguf = GgufFile::parse(&valid_bytes).expect("Valid synthetic GGUF v3 parsing succeeds");

    assert_eq!(gguf.metadata().architecture(), "bitnet-b1.58");
    assert_eq!(gguf.metadata().tensor_count(), 4);
    assert_eq!(gguf.total_weight_bytes(), 216 + 132 + 216 + 128);

    // Verify retrieval of TQ tensors via ModelWeights trait
    let q_view = gguf
        .tensor("model.layers.0.attn_q.weight")
        .expect("attn_q exists");
    assert_eq!(q_view.quant(), QuantType::TQ1_0);
    assert_eq!(q_view.shape(), &[256, 4]);
    assert_eq!(q_view.byte_len(), 216);

    let k_view = gguf
        .tensor("model.layers.0.attn_k.weight")
        .expect("attn_k exists");
    assert_eq!(k_view.quant(), QuantType::TQ2_0);
    assert_eq!(k_view.shape(), &[256, 2]);
    assert_eq!(k_view.byte_len(), 132);

    assert!(gguf.contains_tensor("model.layers.0.attn_v.weight"));
    assert!(!gguf.contains_tensor("model.nonexistent.weight"));

    // 6.2 Corruption Attack: Invalid Magic Header
    let mut corrupt_magic = valid_bytes.clone();
    corrupt_magic[0..4].copy_from_slice(b"BAD!");
    let err_magic = GgufFile::parse(&corrupt_magic).expect_err("BAD! magic must be rejected");
    match err_magic {
        K3Error::TensorBoundsError { reason, .. } => {
            assert!(reason.contains("invalid GGUF magic"));
        }
        other => panic!("Expected TensorBoundsError for invalid magic, got {other:?}"),
    }

    // 6.3 Corruption Attack: Unsupported GGUF Version (e.g. 4)
    let mut corrupt_ver = valid_bytes.clone();
    corrupt_ver[4..8].copy_from_slice(&4_u32.to_le_bytes());
    let err_ver = GgufFile::parse(&corrupt_ver).expect_err("Version 4 must be rejected");
    match err_ver {
        K3Error::CapabilityMismatch { reason, .. } => {
            assert!(reason.contains("unsupported GGUF version 4"));
        }
        other => panic!("Expected CapabilityMismatch for unsupported version, got {other:?}"),
    }

    // 6.4 Corruption Attack: Corrupt ggml_type ID in tensor descriptor
    // In valid_bytes, find the 34 (TQ1_0) type ID and change it to 999
    let mut corrupt_type = valid_bytes.clone();
    let mut found_type = false;
    for i in 0..corrupt_type.len() - 4 {
        if u32::from_le_bytes(corrupt_type[i..i + 4].try_into().unwrap()) == 34 {
            corrupt_type[i..i + 4].copy_from_slice(&999_u32.to_le_bytes());
            found_type = true;
            break;
        }
    }
    assert!(found_type, "Found TQ1_0 type ID 34 in synthetic buffer");
    let err_type = GgufFile::parse(&corrupt_type).expect_err("ID 999 must be rejected");
    assert!(matches!(err_type, K3Error::CapabilityMismatch { .. }));

    // 6.5 Corruption Attack: Truncate file by 1 byte
    let truncated_1b = &valid_bytes[..valid_bytes.len() - 1];
    let err_trunc = GgufFile::parse(truncated_1b).expect_err("Truncated file must be rejected");
    assert!(matches!(err_trunc, K3Error::TensorBoundsError { .. }));
    assert!(err_trunc.to_string().contains("exceeds buffer length"));

    // 6.6 Corruption Attack: Truncate file into tensor header section
    let truncated_hdr = &valid_bytes[..64];
    let err_trunc_hdr = GgufFile::parse(truncated_hdr).expect_err("Truncated header rejected");
    assert!(matches!(err_trunc_hdr, K3Error::TensorBoundsError { .. }));
    assert!(err_trunc_hdr.to_string().contains("unexpected EOF"));
}
