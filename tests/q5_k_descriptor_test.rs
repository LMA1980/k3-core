//! Integration and Property Test Suite for `QuantType::Q5_K` in `k3-core`.
//!
//! Validates variant existence, GGML type ID mapping (13), block sizing (256 / 176 bytes),
//! 16-byte memory alignment, tensor shape calculations, misaligned dimension rejections,
//! `BlockQ5_K` memory layout, and zero-copy row slicing.

#![allow(
    clippy::all,
    clippy::pedantic,
    clippy::nursery,
    clippy::undocumented_unsafe_blocks
)]

use core::mem::{align_of, offset_of, size_of};
use std::collections::{HashMap, HashSet};

use k3_core::{
    BlockQ5_K as CoreBlockQ5_K, GgufFile, K3Error, ModelWeights, QuantType, TensorView, align_up,
    test_helpers::create_synthetic_gguf_v3,
};

// ============================================================================
// 1. Memory Layout and Struct Alignment Validation (BlockQ5_K)
// ============================================================================

/// Exact repr(C) structure for Q5_K super-block (256 elements, 176 bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlockQ5_K {
    pub d: u16,           // 2 bytes (fp16 scale)
    pub dmin: u16,        // 2 bytes (fp16 min scale)
    pub scales: [u8; 12], // 12 bytes (6-bit packed sub-block scales/mins)
    pub qh: [u8; 32],     // 32 bytes (quants high bit for 256 elements)
    pub qs: [u8; 128],    // 128 bytes (quants low 4 bits for 256 elements)
}

/// 16-byte aligned representation for SIMD / Vector hardware units.
#[repr(C, align(16))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AlignedBlockQ5_K {
    pub d: u16,
    pub dmin: u16,
    pub scales: [u8; 12],
    pub qh: [u8; 32],
    pub qs: [u8; 128],
}

#[test]
fn test_q5_k_block_memory_layout_and_offsets() {
    // 1. Struct size must be exactly 176 bytes
    assert_eq!(
        size_of::<BlockQ5_K>(),
        176,
        "BlockQ5_K size must be 176 bytes"
    );
    assert_eq!(
        size_of::<AlignedBlockQ5_K>(),
        176,
        "AlignedBlockQ5_K size must remain 176 bytes"
    );
    assert_eq!(
        size_of::<CoreBlockQ5_K>(),
        176,
        "CoreBlockQ5_K size must be 176 bytes"
    );

    // 2. Natural alignment is 2 bytes (derived from u16); SIMD alignment is 16 bytes
    assert_eq!(
        align_of::<BlockQ5_K>(),
        2,
        "Natural alignment must be 2 bytes"
    );
    assert_eq!(
        align_of::<AlignedBlockQ5_K>(),
        16,
        "Aligned variant must have 16-byte alignment"
    );
    assert_eq!(
        align_of::<CoreBlockQ5_K>(),
        16,
        "CoreBlockQ5_K must have 16-byte alignment"
    );

    // 3. Exact field byte offsets (zero implicit padding)
    assert_eq!(offset_of!(BlockQ5_K, d), 0, "offset of d must be 0");
    assert_eq!(offset_of!(BlockQ5_K, dmin), 2, "offset of dmin must be 2");
    assert_eq!(
        offset_of!(BlockQ5_K, scales),
        4,
        "offset of scales must be 4"
    );
    assert_eq!(
        offset_of!(BlockQ5_K, qh),
        16,
        "offset of qh must be 16 (16-byte aligned)"
    );
    assert_eq!(
        offset_of!(BlockQ5_K, qs),
        48,
        "offset of qs must be 48 (16-byte aligned)"
    );

    // Offsets on CoreBlockQ5_K
    assert_eq!(offset_of!(CoreBlockQ5_K, d), 0);
    assert_eq!(offset_of!(CoreBlockQ5_K, dmin), 2);
    assert_eq!(offset_of!(CoreBlockQ5_K, scales), 4);
    assert_eq!(offset_of!(CoreBlockQ5_K, qh), 16);
    assert_eq!(offset_of!(CoreBlockQ5_K, qs), 48);

    // 4. Invariant: 176 is divisible by 16 (zero tail padding under align(16))
    assert_eq!(176 % 16, 0, "176 bytes must be an exact multiple of 16");
}

// ============================================================================
// 2. QuantType::Q5_K Enum Variant, Traits, and GGML Mapping
// ============================================================================

#[test]
fn test_q5_k_variant_existence_and_ggml_mapping() {
    let q = QuantType::Q5_K;

    // 1. Canonical string identifiers and Display trait
    assert_eq!(q.as_str(), "Q5_K");
    assert_eq!(q.type_name(), "Q5_K");
    assert_eq!(format!("{q}"), "Q5_K");

    // 2. GGUF type ID mapping (ggml_type = 13)
    assert_eq!(q.ggml_type_id(), 13);
    assert_eq!(q.ggml_type(), 13);
    assert_eq!(QuantType::from_gguf_type_id(13), Some(QuantType::Q5_K));

    // 3. Negative type ID mapping check
    assert_eq!(QuantType::from_gguf_type_id(11), None);
    assert_eq!(QuantType::from_gguf_type_id(15), None);
    assert_eq!(QuantType::from_gguf_type_id(999), None);

    // 4. Trait implementations: Copy, Clone, PartialEq, Eq, Hash
    let q_copy = q;
    assert_eq!(q, q_copy);
    assert_ne!(q, QuantType::Q4_K_M);
    assert_ne!(q, QuantType::Q6_K);

    let mut set = HashSet::new();
    assert!(set.insert(QuantType::Q5_K));
    assert!(!set.insert(QuantType::Q5_K));
    assert_eq!(set.len(), 1);

    let mut map = HashMap::new();
    map.insert(QuantType::Q5_K, 13_u32);
    assert_eq!(map.get(&QuantType::Q5_K), Some(&13));
}

// ============================================================================
// 3. Block Sizing, Bits Per Weight, and Storage Calculations
// ============================================================================

#[test]
fn test_q5_k_block_sizing_and_bpw() {
    let q = QuantType::Q5_K;

    // 1. Elements per block (256)
    assert_eq!(q.block_elements(), 256);
    assert_eq!(q.block_size(), 256);

    // 2. Bytes per block (176)
    assert_eq!(q.block_bytes(), 176);
    assert_eq!(q.type_size(), 176);

    // 3. Bits per weight: 176 * 8 / 256 = 1408 / 256 = 5.50 bpw
    let bpw = (q.block_bytes() * 8) as f64 / q.block_elements() as f64;
    assert!(
        (bpw - 5.50).abs() < 1e-6,
        "Q5_K bpw must equal exactly 5.50"
    );
}

#[test]
fn test_q5_k_tensor_size_calculations() {
    let q = QuantType::Q5_K;

    // Shape [1, 256] -> 256 elements -> 1 block -> 176 bytes
    assert_eq!(q.storage_bytes("t1", 256).unwrap(), 176);

    // Shape [4, 256] -> 1024 elements -> 4 blocks -> 704 bytes
    assert_eq!(q.storage_bytes("t2", 1024).unwrap(), 704);

    // Shape [1, 512] -> 512 elements -> 2 blocks -> 352 bytes
    assert_eq!(q.storage_bytes("t3", 512).unwrap(), 352);

    // Shape [256, 8] -> 2048 elements -> 8 blocks -> 1408 bytes
    assert_eq!(q.storage_bytes("t4", 2048).unwrap(), 1408);

    // Multi-dimensional shape checks via TensorView::calculate_bounds
    let cases = [
        (vec![1, 256], 1, 176),
        (vec![4, 256], 4, 704),
        (vec![1, 512], 2, 352),
        (vec![256, 2], 2, 352),
        (vec![64, 4, 10], 10, 1760),
        (vec![1, 1, 256, 1], 1, 176),
    ];

    for (shape, expected_blocks, expected_bytes) in cases {
        let (num_elems, byte_len) =
            TensorView::calculate_bounds("calc", QuantType::Q5_K, &shape, 0, expected_bytes)
                .expect("valid shape bounds calculation");
        assert_eq!(num_elems, expected_blocks * 256);
        assert_eq!(byte_len, expected_bytes);
    }
}

// ============================================================================
// 4. Alignment Validation (16-Byte Hardware Invariants)
// ============================================================================

#[test]
fn test_q5_k_16byte_alignment_validation() {
    let q = QuantType::Q5_K;

    // 1. Block byte size is a strict multiple of 16
    assert_eq!(q.block_bytes() % 16, 0);

    // 2. All sequential block offsets are 16-byte aligned
    for k in 0..=1000 {
        let block_offset = k * q.block_bytes();
        assert_eq!(
            block_offset % 16,
            0,
            "Block {k} offset {block_offset} must be 16-byte aligned"
        );
        assert_eq!(align_up(block_offset, 16), block_offset);
    }

    // 3. Backing buffer sub-slice alignment
    let buffer = vec![0x55_u8; 1760]; // 10 blocks
    let view = TensorView::from_bytes("aligned_q5k", QuantType::Q5_K, &[256, 10], 0, &buffer)
        .expect("valid 10-block view");
    assert_eq!(view.byte_len(), 1760);

    // Verify sub-block slicing preserves 16-byte alignment
    for r in 0..10 {
        let row_slice = view.slice_rows(r, 1).expect("slice row succeeds");
        assert_eq!(row_slice.offset(), r * 176);
        assert_eq!(row_slice.offset() % 16, 0);
        assert_eq!(row_slice.byte_len(), 176);
    }
}

// ============================================================================
// 5. Misaligned Dimensions and Boundary Error Rejections
// ============================================================================

#[test]
fn test_q5_k_misaligned_dimensions_rejection() {
    let q = QuantType::Q5_K;

    // Non-divisible element counts must reject with TensorBoundsError
    let misaligned_counts = [
        1, 2, 3, 15, 31, 63, 127, 128, 255, 257, 511, 513, 1023, 1025, 2047, 2049,
    ];

    for &count in &misaligned_counts {
        let err = q
            .storage_bytes("unaligned_tensor", count)
            .expect_err(&format!("Q5_K must reject unaligned element count {count}"));

        match err {
            K3Error::TensorBoundsError {
                tensor_name,
                offset,
                byte_len,
                buffer_len,
                reason,
            } => {
                assert_eq!(tensor_name, "unaligned_tensor");
                assert_eq!(offset, 0);
                assert_eq!(byte_len, 0);
                assert_eq!(buffer_len, 0);
                assert_eq!(
                    reason,
                    format!("element count {count} is not divisible by Q5_K block size 256")
                );
            }
            other => panic!("Expected TensorBoundsError for count {count}, got: {other:?}"),
        }
    }

    // Empty tensor (0 elements) rejection
    let err_zero = q
        .storage_bytes("zero_tensor", 0)
        .expect_err("0 elements must be rejected");
    match err_zero {
        K3Error::TensorBoundsError { reason, .. } => {
            assert_eq!(reason, "tensor element count must be greater than zero");
        }
        other => panic!("Expected TensorBoundsError for 0 elements, got: {other:?}"),
    }
}

#[test]
fn test_q5_k_tensor_view_boundary_and_buffer_rejections() {
    let buffer_176 = vec![0x00_u8; 176];

    // 1. Truncated backing buffer by 1 byte (175 < 176)
    let buffer_175 = vec![0x00_u8; 175];
    let err_trunc =
        TensorView::from_bytes("trunc", QuantType::Q5_K, &[256], 0, &buffer_175).unwrap_err();
    match err_trunc {
        K3Error::TensorBoundsError {
            byte_len,
            buffer_len,
            reason,
            ..
        } => {
            assert_eq!(byte_len, 176);
            assert_eq!(buffer_len, 175);
            assert!(reason.contains("exceeds backing buffer length 175"));
        }
        other => panic!("Expected TensorBoundsError for truncated buffer, got {other:?}"),
    }

    // 2. Off-by-one offset out-of-bounds (offset 1 in 176-byte buffer)
    let err_oob_offset =
        TensorView::from_bytes("oob_offset", QuantType::Q5_K, &[256], 1, &buffer_176).unwrap_err();
    match err_oob_offset {
        K3Error::TensorBoundsError {
            offset,
            byte_len,
            buffer_len,
            reason,
            ..
        } => {
            assert_eq!(offset, 1);
            assert_eq!(byte_len, 176);
            assert_eq!(buffer_len, 176);
            assert!(reason.contains("tensor range [1..177] exceeds backing buffer length 176"));
        }
        other => panic!("Expected TensorBoundsError for offset OOB, got {other:?}"),
    }

    // 3. Empty shape rejection
    let err_empty_shape =
        TensorView::from_bytes("empty_shape", QuantType::Q5_K, &[], 0, &buffer_176).unwrap_err();
    match err_empty_shape {
        K3Error::TensorBoundsError { reason, .. } => {
            assert_eq!(reason, "tensor shape must contain at least one dimension");
        }
        other => panic!("Expected TensorBoundsError for empty shape, got {other:?}"),
    }

    // 4. Zero dimension rejection
    let err_zero_dim =
        TensorView::from_bytes("zero_dim", QuantType::Q5_K, &[256, 0], 0, &buffer_176).unwrap_err();
    match err_zero_dim {
        K3Error::TensorBoundsError { reason, .. } => {
            assert_eq!(reason, "tensor shape dimensions must be non-zero");
        }
        other => panic!("Expected TensorBoundsError for zero dim, got {other:?}"),
    }
}

// ============================================================================
// 6. 2D Matrix Zero-Copy Slicing Stress
// ============================================================================

#[test]
fn test_q5_k_slice_rows_zero_copy() {
    // 4 rows of 256 elements = 4 * 176 = 704 bytes
    let mut backing = vec![0xAA_u8; 704];
    for r in 0..4 {
        backing[r * 176] = r as u8;
    }

    let view = TensorView::from_bytes("matrix_q5k", QuantType::Q5_K, &[256, 4], 0, &backing)
        .expect("valid 2D Q5_K matrix");

    // Single row slices
    for r in 0..4 {
        let slice = view.slice_rows(r, 1).expect("valid row slice");
        assert_eq!(slice.shape(), &[256, 1]);
        assert_eq!(slice.num_elements(), 256);
        assert_eq!(slice.offset(), r * 176);
        assert_eq!(slice.byte_len(), 176);
        assert_eq!(slice.as_bytes()[0], r as u8);
    }

    // Window slice [1..3] (2 rows = 352 bytes)
    let win = view.slice_rows(1, 2).expect("valid window slice");
    assert_eq!(win.shape(), &[256, 2]);
    assert_eq!(win.num_elements(), 512);
    assert_eq!(win.offset(), 176);
    assert_eq!(win.byte_len(), 352);
    assert_eq!(win.as_bytes()[0], 1);

    // 0-row slice
    let zero_slice = view.slice_rows(2, 0).expect("valid 0-row slice");
    assert_eq!(zero_slice.shape(), &[256, 0]);
    assert_eq!(zero_slice.num_elements(), 0);
    assert_eq!(zero_slice.byte_len(), 0);
    assert_eq!(zero_slice.offset(), 352);

    // Out-of-bounds row range
    let err_oob_row = view.slice_rows(3, 2).unwrap_err();
    match err_oob_row {
        K3Error::TensorBoundsError { reason, .. } => {
            assert!(reason.contains("exceeds tensor row count 4"));
        }
        other => panic!("Expected TensorBoundsError for row range OOB, got {other:?}"),
    }

    // Unaligned column dimension rejection:
    // Shape [128, 2] has total 256 elements (divisible by 256), but cols=128 is not divisible by 256
    let unaligned_col_buf = vec![0u8; 176];
    let view_unaligned_cols = TensorView::from_bytes(
        "unaligned_cols",
        QuantType::Q5_K,
        &[128, 2],
        0,
        &unaligned_col_buf,
    )
    .expect("from_bytes succeeds for 256 elements");

    let err_align = view_unaligned_cols.slice_rows(0, 1).unwrap_err();
    match err_align {
        K3Error::TensorBoundsError { reason, .. } => {
            assert_eq!(
                reason,
                "tensor row length 128 is not divisible by Q5_K block size 256"
            );
        }
        other => panic!("Expected TensorBoundsError for unaligned cols, got {other:?}"),
    }
}

// ============================================================================
// 7. Property-Based Invariant Tests (Linearity & Monotonicity)
// ============================================================================

#[test]
fn test_q5_k_property_scaling_linearity() {
    let q = QuantType::Q5_K;

    // Property 1: Linearity over 1..=5000 blocks
    for n_blocks in 1..=5000 {
        let num_elements = n_blocks * 256;
        let bytes = q.storage_bytes("prop_linear", num_elements).unwrap();
        let expected = n_blocks * 176;
        assert_eq!(bytes, expected, "Failed linearity at {n_blocks} blocks");
        assert_eq!(
            bytes % 16,
            0,
            "Failed 16-byte alignment at {n_blocks} blocks"
        );
    }

    // Property 2: Rejection of all deltas in [1..255]
    for delta in [1, 2, 7, 15, 31, 63, 127, 255] {
        let unaligned_elems = (10 * 256) + delta;
        assert!(q.storage_bytes("prop_delta", unaligned_elems).is_err());
    }
}

// ============================================================================
// 8. Synthetic GGUF v3 File Deserialization and ModelWeights Access
// ============================================================================

#[test]
fn test_q5_k_synthetic_gguf_v3_roundtrip() {
    let specs = [
        (
            "model.layers.0.attn_q.weight",
            QuantType::Q5_K,
            &[256, 4][..],
        ), // 1024 elems = 4 blk = 704B
        (
            "model.layers.0.attn_k.weight",
            QuantType::Q5_K,
            &[256, 2][..],
        ), // 512 elems = 2 blk = 352B
        ("model.norm.weight", QuantType::F32, &[32][..]), // 32 elems = 128B
    ];

    let file_bytes = create_synthetic_gguf_v3("dory-ngpt-q5k", 32, &specs);
    let gguf = GgufFile::parse(&file_bytes).expect("Valid synthetic GGUF v3 parsing succeeds");

    assert_eq!(gguf.metadata().architecture(), "dory-ngpt-q5k");
    assert_eq!(gguf.metadata().tensor_count(), 3);
    assert_eq!(gguf.total_weight_bytes(), 704 + 352 + 128);

    // Verify Q5_K tensor views via ModelWeights trait
    let q_view = gguf
        .tensor("model.layers.0.attn_q.weight")
        .expect("attn_q exists");
    assert_eq!(q_view.quant(), QuantType::Q5_K);
    assert_eq!(q_view.shape(), &[256, 4]);
    assert_eq!(q_view.num_elements(), 1024);
    assert_eq!(q_view.byte_len(), 704);
    assert_eq!(q_view.byte_len() % 16, 0);

    let k_view = gguf
        .tensor("model.layers.0.attn_k.weight")
        .expect("attn_k exists");
    assert_eq!(k_view.quant(), QuantType::Q5_K);
    assert_eq!(k_view.shape(), &[256, 2]);
    assert_eq!(k_view.num_elements(), 512);
    assert_eq!(k_view.byte_len(), 352);

    assert!(gguf.contains_tensor("model.layers.0.attn_q.weight"));
    assert!(!gguf.contains_tensor("model.nonexistent.weight"));
}
