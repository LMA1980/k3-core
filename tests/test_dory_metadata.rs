//! Integration and Property Test Suite for `dory.*` GGUF Metadata Parsing in `k3-core`.
//!
//! Validates:
//! 1. Golden path parsing of all 11 `dory.*` GGUF metadata keys into `DoryMetadata`.
//! 2. Default value fallback when optional fields are omitted.
//! 3. Architecture mismatch isolation (`general.architecture != "dory"` returns `None`).
//! 4. Alternate KV cache modes (`"per_loop"` vs `"last_loop"`).
//! 5. Adverse and corrupt input rejection (negative counts, type mismatch, missing required keys).
//! 6. Helper query methods (`physical_layer_count`, `effective_layer_count`, segment bounds).
//! 7. Trait implementations (`Display`, `Debug`, `Clone`, `PartialEq`, `Eq`).

#![allow(
    clippy::all,
    clippy::pedantic,
    clippy::nursery,
    clippy::undocumented_unsafe_blocks
)]

use k3_core::test_helpers::create_synthetic_gguf_v3_with_kvs;
use k3_core::{DoryKvCacheMode, DoryMetadata, GgufFile, GgufValue, K3Error};

// ============================================================================
// 1. Golden Path: Complete `dory.*` Metadata Parsing
// ============================================================================

#[test]
fn test_parse_dory_gguf_metadata() {
    let layer_pattern = "S-S-S-*-S-S-S-*-S-S-S-*-S-S-S-*";
    let extra_kvs = [
        ("dory.input_layer_count", GgufValue::Uint32(16)),
        ("dory.recurrent_layer_count", GgufValue::Uint32(40)),
        ("dory.output_layer_count", GgufValue::Uint32(16)),
        ("dory.num_loops", GgufValue::Uint32(2)),
        (
            "dory.recurrent_kv_cache_mode",
            GgufValue::String("per_loop"),
        ),
        ("dory.layer_pattern", GgufValue::String(layer_pattern)),
        ("dory.rope.profile", GgufValue::String("dual_theta")),
        ("dory.rope.freq_base_2", GgufValue::Float32(10000.0)),
        ("dory.init_std", GgufValue::Float32(0.02)),
        ("dory.alpha_init", GgufValue::Float32(1.0)),
        ("dory.gate_init", GgufValue::Float32(0.0)),
    ];

    let bytes = create_synthetic_gguf_v3_with_kvs("dory", 32, &extra_kvs, &[]);
    let gguf = GgufFile::parse(&bytes).expect("GGUF v3 parsing succeeds for dory architecture");

    assert_eq!(gguf.architecture(), "dory");
    assert_eq!(gguf.metadata().architecture(), "dory");

    let dory: &DoryMetadata = gguf
        .metadata()
        .dory()
        .expect("DoryMetadata must be present when architecture is 'dory'");

    // Field-by-field verification against ground truth
    assert_eq!(dory.input_layer_count(), 16);
    assert_eq!(dory.recurrent_layer_count(), 40);
    assert_eq!(dory.output_layer_count(), 16);
    assert_eq!(dory.num_loops(), 2);
    assert_eq!(dory.recurrent_kv_cache_mode(), DoryKvCacheMode::PerLoop);
    assert_eq!(dory.layer_pattern(), layer_pattern);
    assert_eq!(dory.rope_profile(), "dual_theta");
    assert!((dory.rope_freq_base_2() - 10000.0).abs() < f32::EPSILON);
    assert!((dory.init_std() - 0.02).abs() < f32::EPSILON);
    assert!((dory.alpha_init() - 1.0).abs() < f32::EPSILON);
    assert!((dory.gate_init() - 0.0).abs() < f32::EPSILON);

    // Also verify convenience accessor on GgufFile if available
    if let Some(file_dory) = gguf.dory() {
        assert_eq!(file_dory, dory);
    }
}

// ============================================================================
// 2. Optional Fields & Defaults Fallback
// ============================================================================

#[test]
fn test_dory_metadata_optional_defaults() {
    let layer_pattern = "S-S-S-*-";
    // Only supply the required structural keys; omit optional keys
    let extra_kvs = [
        ("dory.input_layer_count", GgufValue::Uint32(16)),
        ("dory.recurrent_layer_count", GgufValue::Uint32(40)),
        ("dory.output_layer_count", GgufValue::Uint32(16)),
        ("dory.layer_pattern", GgufValue::String(layer_pattern)),
    ];

    let bytes = create_synthetic_gguf_v3_with_kvs("dory", 32, &extra_kvs, &[]);
    let gguf = GgufFile::parse(&bytes).expect("parsing succeeds with required keys only");

    let dory = gguf
        .metadata()
        .dory()
        .expect("DoryMetadata must be present");

    // Required fields match
    assert_eq!(dory.input_layer_count(), 16);
    assert_eq!(dory.recurrent_layer_count(), 40);
    assert_eq!(dory.output_layer_count(), 16);
    assert_eq!(dory.layer_pattern(), layer_pattern);

    // Optional fields assume canonical defaults
    assert_eq!(dory.num_loops(), 1, "default num_loops must be 1");
    assert_eq!(
        dory.recurrent_kv_cache_mode(),
        DoryKvCacheMode::PerLoop,
        "default recurrent_kv_cache_mode must be PerLoop"
    );
    assert_eq!(
        dory.rope_profile(),
        "dual_theta",
        "default rope_profile must be 'dual_theta'"
    );
    assert!(
        (dory.rope_freq_base_2() - 10000.0).abs() < f32::EPSILON,
        "default rope_freq_base_2 must be 10000.0"
    );
    assert!(
        (dory.init_std() - 0.02).abs() < 1e-4,
        "default init_std must be 0.02"
    );
    assert!(
        (dory.alpha_init() - 1.0).abs() < f32::EPSILON,
        "default alpha_init must be 1.0"
    );
    assert!(
        (dory.gate_init() - 0.0).abs() < f32::EPSILON,
        "default gate_init must be 0.0"
    );
}

// ============================================================================
// 3. Architecture Mismatch Isolation
// ============================================================================

#[test]
fn test_dory_metadata_architecture_mismatch() {
    let non_dory_archs = ["llama", "bitnet", "synthetic", "qwen2", "mistral"];

    for arch in non_dory_archs {
        let extra_kvs = [
            ("dory.input_layer_count", GgufValue::Uint32(16)),
            ("dory.recurrent_layer_count", GgufValue::Uint32(40)),
            ("dory.output_layer_count", GgufValue::Uint32(16)),
        ];
        let bytes = create_synthetic_gguf_v3_with_kvs(arch, 32, &extra_kvs, &[]);
        let gguf = GgufFile::parse(&bytes).expect("parse non-dory arch");

        assert_eq!(
            gguf.metadata().dory(),
            None,
            "architecture '{arch}' must return None for .dory()"
        );
        assert_eq!(
            gguf.dory(),
            None,
            "architecture '{arch}' must return None for GgufFile::dory()"
        );
    }
}

// ============================================================================
// 4. Alternate KV Cache Mode: last_loop
// ============================================================================

#[test]
fn test_dory_metadata_last_loop_cache_mode() {
    let extra_kvs = [
        ("dory.input_layer_count", GgufValue::Uint32(16)),
        ("dory.recurrent_layer_count", GgufValue::Uint32(40)),
        ("dory.output_layer_count", GgufValue::Uint32(16)),
        ("dory.layer_pattern", GgufValue::String("S-*-")),
        (
            "dory.recurrent_kv_cache_mode",
            GgufValue::String("last_loop"),
        ),
    ];

    let bytes = create_synthetic_gguf_v3_with_kvs("dory", 32, &extra_kvs, &[]);
    let gguf = GgufFile::parse(&bytes).expect("parse succeeds with last_loop cache mode");
    let dory = gguf.metadata().dory().expect("dory metadata exists");

    assert_eq!(dory.recurrent_kv_cache_mode(), DoryKvCacheMode::LastLoop);
    assert_eq!(dory.recurrent_kv_cache_mode().as_str(), "last_loop");
    assert_eq!(format!("{}", dory.recurrent_kv_cache_mode()), "last_loop");
}

// ============================================================================
// 5. Structural Helper Methods & Layer Segment Classification
// ============================================================================

#[test]
fn test_dory_metadata_helper_methods() {
    let extra_kvs = [
        ("dory.input_layer_count", GgufValue::Uint32(16)),
        ("dory.recurrent_layer_count", GgufValue::Uint32(40)),
        ("dory.output_layer_count", GgufValue::Uint32(16)),
        ("dory.num_loops", GgufValue::Uint32(3)),
        ("dory.layer_pattern", GgufValue::String("S-*-")),
    ];

    let bytes = create_synthetic_gguf_v3_with_kvs("dory", 32, &extra_kvs, &[]);
    let gguf = GgufFile::parse(&bytes).expect("parse succeeds");
    let dory = gguf.metadata().dory().expect("dory metadata exists");

    // 16 + 40 + 16 = 72 physical sub-layers
    assert_eq!(dory.physical_layer_count(), 72);
    assert_eq!(dory.total_layer_count(), 72);

    // 16 + (40 * 3) + 16 = 152 effective sub-layers executed per token
    assert_eq!(dory.effective_layer_count(), 152);

    // Segment classification:
    // Input segment: [0..16)
    for idx in 0..16 {
        assert!(dory.is_input_layer(idx), "layer {idx} must be input layer");
        assert!(!dory.is_recurrent_layer(idx));
        assert!(!dory.is_output_layer(idx));
    }

    // Recurrent segment: [16..56)
    for idx in 16..56 {
        assert!(
            dory.is_recurrent_layer(idx),
            "layer {idx} must be recurrent layer"
        );
        assert!(!dory.is_input_layer(idx));
        assert!(!dory.is_output_layer(idx));
    }

    // Output segment: [56..72)
    for idx in 56..72 {
        assert!(
            dory.is_output_layer(idx),
            "layer {idx} must be output layer"
        );
        assert!(!dory.is_input_layer(idx));
        assert!(!dory.is_recurrent_layer(idx));
    }

    // Out-of-bounds layer indices
    assert!(!dory.is_input_layer(72));
    assert!(!dory.is_recurrent_layer(72));
    assert!(!dory.is_output_layer(72));
}

// ============================================================================
// 6. Adverse / Corrupt Values Rejection
// ============================================================================

#[test]
fn test_dory_metadata_adverse_negative_layer_counts() {
    // Negative layer count passed as signed int must be rejected
    let extra_kvs = [
        ("dory.input_layer_count", GgufValue::Int32(-1)),
        ("dory.recurrent_layer_count", GgufValue::Uint32(40)),
        ("dory.output_layer_count", GgufValue::Uint32(16)),
        ("dory.layer_pattern", GgufValue::String("S-*-")),
    ];

    let bytes = create_synthetic_gguf_v3_with_kvs("dory", 32, &extra_kvs, &[]);
    let err = GgufFile::parse(&bytes).expect_err("negative input_layer_count must fail");
    assert!(matches!(err, K3Error::CapabilityMismatch { .. }));
}

#[test]
fn test_dory_metadata_adverse_invalid_kv_types() {
    // Float provided where integer required
    let extra_kvs = [
        ("dory.input_layer_count", GgufValue::Float32(16.5)),
        ("dory.recurrent_layer_count", GgufValue::Uint32(40)),
        ("dory.output_layer_count", GgufValue::Uint32(16)),
        ("dory.layer_pattern", GgufValue::String("S-*-")),
    ];

    let bytes = create_synthetic_gguf_v3_with_kvs("dory", 32, &extra_kvs, &[]);
    let err = GgufFile::parse(&bytes).expect_err("invalid type for input_layer_count must fail");
    assert!(matches!(err, K3Error::CapabilityMismatch { .. }));

    // String provided where float required
    let extra_kvs_float = [
        ("dory.input_layer_count", GgufValue::Uint32(16)),
        ("dory.recurrent_layer_count", GgufValue::Uint32(40)),
        ("dory.output_layer_count", GgufValue::Uint32(16)),
        ("dory.layer_pattern", GgufValue::String("S-*-")),
        (
            "dory.rope.freq_base_2",
            GgufValue::String("not_a_float_string"),
        ),
    ];
    let bytes_float = create_synthetic_gguf_v3_with_kvs("dory", 32, &extra_kvs_float, &[]);
    let err_float =
        GgufFile::parse(&bytes_float).expect_err("invalid string for float freq_base_2 must fail");
    assert!(matches!(err_float, K3Error::CapabilityMismatch { .. }));
}

#[test]
fn test_dory_metadata_adverse_invalid_cache_mode() {
    let extra_kvs = [
        ("dory.input_layer_count", GgufValue::Uint32(16)),
        ("dory.recurrent_layer_count", GgufValue::Uint32(40)),
        ("dory.output_layer_count", GgufValue::Uint32(16)),
        ("dory.layer_pattern", GgufValue::String("S-*-")),
        (
            "dory.recurrent_kv_cache_mode",
            GgufValue::String("invalid_mode"),
        ),
    ];

    let bytes = create_synthetic_gguf_v3_with_kvs("dory", 32, &extra_kvs, &[]);
    let err = GgufFile::parse(&bytes).expect_err("unsupported cache mode must fail");
    assert!(matches!(err, K3Error::CapabilityMismatch { .. }));
    assert!(err.to_string().contains("recurrent_kv_cache_mode"));
}

#[test]
fn test_dory_metadata_adverse_zero_loops() {
    let extra_kvs = [
        ("dory.input_layer_count", GgufValue::Uint32(16)),
        ("dory.recurrent_layer_count", GgufValue::Uint32(40)),
        ("dory.output_layer_count", GgufValue::Uint32(16)),
        ("dory.layer_pattern", GgufValue::String("S-*-")),
        ("dory.num_loops", GgufValue::Uint32(0)),
    ];

    let bytes = create_synthetic_gguf_v3_with_kvs("dory", 32, &extra_kvs, &[]);
    let err = GgufFile::parse(&bytes).expect_err("zero loops must fail");
    assert!(matches!(err, K3Error::CapabilityMismatch { .. }));
}

#[test]
fn test_dory_metadata_adverse_missing_required_keys() {
    // Missing dory.input_layer_count when general.architecture is 'dory'
    let extra_kvs = [
        ("dory.recurrent_layer_count", GgufValue::Uint32(40)),
        ("dory.output_layer_count", GgufValue::Uint32(16)),
        ("dory.layer_pattern", GgufValue::String("S-*-")),
    ];

    let bytes = create_synthetic_gguf_v3_with_kvs("dory", 32, &extra_kvs, &[]);
    let err = GgufFile::parse(&bytes).expect_err("missing required input_layer_count must fail");
    assert!(matches!(err, K3Error::CapabilityMismatch { .. }));
    assert!(err.to_string().contains("input_layer_count"));
}

#[test]
fn test_dory_metadata_adverse_empty_layer_pattern() {
    let extra_kvs = [
        ("dory.input_layer_count", GgufValue::Uint32(16)),
        ("dory.recurrent_layer_count", GgufValue::Uint32(40)),
        ("dory.output_layer_count", GgufValue::Uint32(16)),
        ("dory.layer_pattern", GgufValue::String("")),
    ];

    let bytes = create_synthetic_gguf_v3_with_kvs("dory", 32, &extra_kvs, &[]);
    let err = GgufFile::parse(&bytes).expect_err("empty layer pattern must fail");
    assert!(matches!(err, K3Error::CapabilityMismatch { .. }));
}

// ============================================================================
// 7. Trait & Enum Display Validation
// ============================================================================

#[test]
fn test_dory_kv_cache_mode_enum_traits() {
    assert_eq!(DoryKvCacheMode::default(), DoryKvCacheMode::PerLoop);
    assert_eq!(
        DoryKvCacheMode::parse("per_loop"),
        Some(DoryKvCacheMode::PerLoop)
    );
    assert_eq!(
        DoryKvCacheMode::parse("last_loop"),
        Some(DoryKvCacheMode::LastLoop)
    );
    assert_eq!(DoryKvCacheMode::parse("other"), None);

    assert_eq!(DoryKvCacheMode::PerLoop.as_str(), "per_loop");
    assert_eq!(DoryKvCacheMode::LastLoop.as_str(), "last_loop");
    assert_eq!(format!("{}", DoryKvCacheMode::PerLoop), "per_loop");
    assert_eq!(format!("{}", DoryKvCacheMode::LastLoop), "last_loop");
}
