//! Empirical Adversarial Challenge Test Suite for Milestone 1 of US-07.
//!
//! Stress-tests conservative defaults, invariant bounds, custom `n_ctx`, operator transitions,
//! JSON serialization resilience, schema conformance, and concurrent access in `k3-core`.

use std::thread;

use k3_core::{
    A100_CLUSTER_MASK, CONSERVATIVE_DEFAULT_A100_CORES, CONSERVATIVE_DEFAULT_N_CTX,
    CONSERVATIVE_DEFAULT_X100_CORES, CONSERVATIVE_TUNING_ADVISORY, ComputePool, CostProfile,
    DEFAULT_ACTIVE_CONTROL_MASK, DEFAULT_SPARE_RECOVERY_MASK, DEFAULT_X100_COMPUTE_CAP_MASK,
    K3Error, ModelEntry, ModelIdentity, ModelListResponse, ModelLoadAuditEvent, ModelProfile,
    PriorityClass, ProfileStatus, TuningProfile,
};

/// 1. Comprehensive verification of all conservative default invariants.
#[test]
fn test_challenge_conservative_invariants_exhaustive() {
    let profile = TuningProfile::conservative_default();

    // Invariant 1: General-purpose X100 pool only, A100 strictly isolated (0 cores)
    assert_eq!(profile.compute_pool, ComputePool::X100);
    assert_eq!(profile.cluster, "x100");
    assert_eq!(profile.a100_cores, 0);
    assert_eq!(profile.a100_cores, CONSERVATIVE_DEFAULT_A100_CORES);

    // Invariant 2: Exactly 6 compute cores on X100, bounded by 0x003F and disjoint from control (0x00C0)
    assert_eq!(profile.x100_cores, 6);
    assert_eq!(profile.x100_cores, CONSERVATIVE_DEFAULT_X100_CORES);
    assert_eq!(
        profile.x100_cores,
        DEFAULT_X100_COMPUTE_CAP_MASK.count_ones() as usize
    );
    assert_eq!(
        DEFAULT_X100_COMPUTE_CAP_MASK & DEFAULT_ACTIVE_CONTROL_MASK,
        0,
        "Compute cap mask MUST NOT overlap active control cores"
    );
    assert_eq!(
        DEFAULT_ACTIVE_CONTROL_MASK & DEFAULT_SPARE_RECOVERY_MASK,
        0,
        "Active control cores MUST NOT overlap spare recovery core"
    );
    assert_eq!(
        profile.compute_pool.core_mask() & A100_CLUSTER_MASK,
        0,
        "X100 compute pool mask MUST NOT overlap A100 cluster mask"
    );

    // Invariant 3: Standard Throughput priority class
    assert_eq!(profile.priority_class, PriorityClass::Throughput);
    assert_eq!(profile.priority_class.as_str(), "throughput");

    // Invariant 4: Conservative default n_ctx = 2048
    assert_eq!(profile.n_ctx, 2048);
    assert_eq!(profile.n_ctx, CONSERVATIVE_DEFAULT_N_CTX);

    // Invariant 5: Core spillover lending and borrowing disabled
    assert!(!profile.lend_idle_cores);
    assert!(!profile.borrow_idle_cores);

    // Invariant 6: Status is explicitly unprofiled with canonical advisory notice
    assert!(profile.unprofiled);
    assert!(profile.is_unprofiled());
    assert_eq!(profile.profile_status, ProfileStatus::Unprofiled);
    assert_eq!(
        profile.advisory_notice.as_deref(),
        Some(CONSERVATIVE_TUNING_ADVISORY)
    );

    // Default trait conformance
    assert_eq!(TuningProfile::default(), profile);
}

/// 2. Stress-test custom `n_ctx` overrides across extreme boundary values.
#[test]
fn test_challenge_custom_n_ctx_boundary_values() {
    let test_cases = [
        0_usize,
        1_usize,
        64_usize, // CONSTRAINED_TEST_N_CTX
        512_usize,
        2048_usize, // CONSERVATIVE_DEFAULT_N_CTX
        8192_usize,
        32768_usize,
        131_072_usize,
        usize::MAX,
    ];

    for &n_ctx in &test_cases {
        let profile = TuningProfile::conservative_default_with_n_ctx(n_ctx);

        // Verify n_ctx is faithfully set
        assert_eq!(profile.n_ctx, n_ctx);

        // Verify all other conservative defaults are strictly preserved
        assert_eq!(profile.compute_pool, ComputePool::X100);
        assert_eq!(profile.cluster, "x100");
        assert_eq!(profile.x100_cores, 6);
        assert_eq!(profile.a100_cores, 0);
        assert_eq!(profile.priority_class, PriorityClass::Throughput);
        assert!(!profile.lend_idle_cores);
        assert!(!profile.borrow_idle_cores);
        assert!(profile.unprofiled);
        assert_eq!(profile.profile_status, ProfileStatus::Unprofiled);
        assert_eq!(
            profile.advisory_notice.as_deref(),
            Some(CONSERVATIVE_TUNING_ADVISORY)
        );

        // ModelProfile constructor with custom n_ctx
        let model = ModelProfile::unprofiled_with_n_ctx("test-boundary-model", n_ctx);
        assert_eq!(model.model_id, "test-boundary-model");
        assert_eq!(model.tuning.n_ctx, n_ctx);
        assert!(model.is_unprofiled());
        let entry = model.to_model_entry(12345);
        assert_eq!(entry.n_ctx, n_ctx);
        assert!(entry.unprofiled);
    }
}

/// 3. Stress-test operator tuning transition across all permutations of `ComputePool`, cores, and priority.
#[test]
fn test_challenge_operator_tuning_transitions_and_sanitization() {
    let pools = [ComputePool::X100, ComputePool::A100, ComputePool::Both];
    let priorities = [PriorityClass::Throughput, PriorityClass::Latency];

    for pool in pools {
        for prio in priorities {
            for &lend in &[false, true] {
                for &borrow in &[false, true] {
                    let mut profile = TuningProfile::conservative_default();
                    assert!(profile.is_unprofiled());
                    assert!(profile.advisory_notice.is_some());

                    let x100_c = if pool == ComputePool::A100 { 0 } else { 4 };
                    let a100_c = if pool == ComputePool::X100 { 0 } else { 8 };
                    let n_ctx = 4096;

                    profile.apply_operator_tuning(pool, x100_c, a100_c, prio, n_ctx, lend, borrow);

                    // Assert tuned invariants
                    assert!(!profile.is_unprofiled());
                    assert!(!profile.unprofiled);
                    assert_eq!(profile.profile_status, ProfileStatus::Profiled);
                    assert_eq!(profile.compute_pool, pool);
                    assert_eq!(profile.cluster, pool.as_str());
                    assert_eq!(profile.x100_cores, x100_c);
                    assert_eq!(profile.a100_cores, a100_c);
                    assert_eq!(profile.priority_class, prio);
                    assert_eq!(profile.n_ctx, n_ctx);
                    assert_eq!(profile.lend_idle_cores, lend);
                    assert_eq!(profile.borrow_idle_cores, borrow);
                    assert_eq!(profile.advisory_notice, None);

                    // Round-trip to JSON and back
                    let json = profile.to_json_pretty().expect("serialize tuned profile");
                    assert!(!json.contains("advisory_notice"));
                    assert!(json.contains("\"unprofiled\": false"));
                    assert!(json.contains("\"profile_status\": \"profiled\""));

                    let deserialized =
                        TuningProfile::from_json_str(&json).expect("deserialize tuned profile");
                    assert_eq!(profile, deserialized);
                }
            }
        }
    }
}

/// 4. Adversarial challenge on `ModelProfile::apply_tuning`:
///
/// If a caller passes a `TuningProfile` that still has `unprofiled: true` or `advisory_notice: Some(...)`,
/// `apply_tuning` MUST sanitize and force `unprofiled = false`, `profile_status = Profiled`, `advisory_notice = None`.
#[test]
fn test_challenge_model_profile_sanitizes_unprofiled_flag_on_tuning() {
    let mut model = ModelProfile::unprofiled_default("qwen-test");
    assert!(model.is_unprofiled());

    // Adversarial: tuning profile claims to be unprofiled with advisory notice
    let malicious_tuning = TuningProfile {
        model_id: None,
        compute_pool: ComputePool::Both,
        cluster: "both".to_owned(),
        x100_cores: 6,
        a100_cores: 8,
        priority_class: PriorityClass::Latency,
        n_ctx: 8192,
        lend_idle_cores: true,
        borrow_idle_cores: true,
        profile_status: ProfileStatus::Unprofiled, // Sneaky unprofiled status!
        unprofiled: true,                          // Sneaky unprofiled flag!
        advisory_notice: Some("Malicious advisory".to_owned()),
    };

    model.apply_tuning(malicious_tuning);

    // Assert that ModelProfile::apply_tuning sanitized these fields
    assert_eq!(model.tuning.model_id, Some("qwen-test".to_owned()));
    assert!(!model.is_unprofiled());
    assert!(!model.tuning.unprofiled);
    assert_eq!(model.tuning.profile_status, ProfileStatus::Profiled);
    assert_eq!(model.tuning.advisory_notice, None);

    let entry = model.to_model_entry(100);
    assert!(!entry.unprofiled);
    assert_eq!(entry.profile_status, "profiled");
    assert_eq!(entry.advisory_notice, None);
}

/// 5. JSON serialization resilience: edge cases, unicode, large payloads, and missing fields.
#[test]
fn test_challenge_json_serialization_adversarial_edge_cases() {
    let repeated = "a".repeat(1024);
    let weird_ids: [&str; 5] = [
        "🤖-deepseek-r1:7b-instruct_v1.0",
        "org/sub-team/model@v2.1#prod",
        "  spaces  around  ",
        "模型-7b-中文-測試",
        &repeated,
    ];

    for &raw_id in &weird_ids {
        let model = ModelProfile::unprofiled_default(raw_id);
        let entry = model.to_model_entry(1_700_000_000);
        let json = serde_json::to_string(&entry).expect("serialize weird model entry");
        let parsed: ModelEntry =
            serde_json::from_str(&json).expect("deserialize weird model entry");
        assert_eq!(parsed.id, raw_id);
        assert!(parsed.unprofiled);
        assert_eq!(
            parsed.advisory_notice.as_deref(),
            Some(CONSERVATIVE_TUNING_ADVISORY)
        );

        let audit = ModelLoadAuditEvent::from_tuning(raw_id, &model.tuning);
        let audit_json = serde_json::to_string(&audit).expect("serialize weird audit event");
        let parsed_audit: ModelLoadAuditEvent =
            serde_json::from_str(&audit_json).expect("deserialize weird audit event");
        assert_eq!(parsed_audit.model_id, raw_id);
    }

    // B. Deserialization with unknown/extra fields in JSON should not panic or fail
    let json_with_extra = format!(
        r#"{{
            "id": "extra-fields-model",
            "object": "model",
            "created": 1700000000,
            "owned_by": "k3",
            "unprofiled": true,
            "profile_status": "unprofiled",
            "priority_class": "throughput",
            "cluster": "x100",
            "a100_cores": 0,
            "n_ctx": 2048,
            "lend_idle_cores": false,
            "borrow_idle_cores": false,
            "advisory_notice": "{CONSERVATIVE_TUNING_ADVISORY}",
            "unknown_future_field_foo": "bar",
            "telemetry_metric": 42
        }}"#
    );

    let parsed_extra: ModelEntry = serde_json::from_str(&json_with_extra)
        .expect("should deserialize JSON with extra unknown fields");
    assert_eq!(parsed_extra.id, "extra-fields-model");
    assert!(parsed_extra.unprofiled);

    // C. Malformed / corrupted JSON should cleanly return Error, not panic
    let corruptions = [
        "",
        "{",
        "null",
        "{\"unprofiled\": \"not_a_bool\"}",
        "{\"priority_class\": \"hyperspeed\"}",
        "{\"n_ctx\": -42}",
        "{\"cluster\": 123}",
    ];

    for corrupt in corruptions {
        let res: Result<ModelEntry, _> = serde_json::from_str(corrupt);
        assert!(res.is_err(), "corrupt input should fail: {corrupt}");
    }
}

/// 6. String parsing error diagnostics and case insensitivity.
#[test]
fn test_challenge_parsing_diagnostics_and_case_handling() {
    // PriorityClass
    for valid in &[
        "throughput",
        "Throughput",
        "THROUGHPUT",
        "latency",
        "Latency",
        "LATENCY",
    ] {
        assert!(PriorityClass::parse(valid).is_ok(), "valid: {valid}");
    }
    for invalid in &["", "realtime", "batch", "urgent", "123"] {
        let err = PriorityClass::parse(invalid).unwrap_err();
        assert!(matches!(err, K3Error::CapabilityMismatch { .. }));
    }

    // ComputePool
    for valid in &["x100", "X100", "a100", "A100", "both", "Both", "BOTH"] {
        assert!(ComputePool::parse(valid).is_ok(), "valid: {valid}");
    }
    for invalid in &["", "gpu", "npu", "spacemit", "cluster0"] {
        let err = ComputePool::parse(invalid).unwrap_err();
        assert!(matches!(err, K3Error::CapabilityMismatch { .. }));
    }

    // ProfileStatus
    for valid in &[
        "unprofiled",
        "Unprofiled",
        "UNPROFILED",
        "profiled",
        "Profiled",
        "PROFILED",
    ] {
        assert!(ProfileStatus::parse(valid).is_ok(), "valid: {valid}");
    }
    for invalid in &["", "unknown", "in_progress", "failed"] {
        let err = ProfileStatus::parse(invalid).unwrap_err();
        assert!(matches!(err, K3Error::CapabilityMismatch { .. }));
    }
}

/// 7. `OpenAI` `/v1/models` collection response with mixed models.
#[test]
fn test_challenge_openai_model_list_collection_response() {
    let mut entries = Vec::new();

    // 10 unprofiled models
    for i in 0_u64..10_u64 {
        let m = ModelProfile::unprofiled_default(format!("unprofiled-model-{i}"));
        entries.push(m.to_model_entry(1_700_000_000 + i));
    }

    // 10 profiled models with various configurations
    for i in 0_u64..10_u64 {
        let mut m = ModelProfile::unprofiled_default(format!("profiled-model-{i}"));
        let mut tuning = TuningProfile::conservative_default();
        tuning.apply_operator_tuning(
            if i % 2 == 0 {
                ComputePool::A100
            } else {
                ComputePool::Both
            },
            4,
            8,
            PriorityClass::Latency,
            8192,
            true,
            false,
        );
        m.apply_tuning(tuning);
        entries.push(m.to_model_entry(1_700_000_100 + i));
    }

    let response = ModelListResponse::new(entries);
    assert_eq!(response.object, "list");
    assert_eq!(response.data.len(), 20);

    let json = serde_json::to_string_pretty(&response).expect("serialize list response");
    let deserialized: ModelListResponse =
        serde_json::from_str(&json).expect("deserialize list response");

    assert_eq!(deserialized.object, "list");
    assert_eq!(deserialized.data.len(), 20);

    // Verify unprofiled vs profiled attributes in collection
    for i in 0..10 {
        let entry = &deserialized.data[i];
        assert!(entry.unprofiled);
        assert_eq!(entry.profile_status, "unprofiled");
        assert_eq!(entry.cluster, "x100");
        assert_eq!(entry.a100_cores, 0);
        assert!(entry.advisory_notice.is_some());
    }

    for i in 10..20 {
        let entry = &deserialized.data[i];
        assert!(!entry.unprofiled);
        assert_eq!(entry.profile_status, "profiled");
        assert_eq!(entry.priority_class, "latency");
        assert_eq!(entry.a100_cores, 8);
        assert!(entry.advisory_notice.is_none());
    }
}

/// 8. Multithreaded concurrent generation, serialization, and roundtrip stress.
#[test]
fn test_challenge_concurrency_stress() {
    let threads = 16;
    let iterations_per_thread = 100;

    let handles: Vec<_> = (0..threads)
        .map(|tid| {
            thread::spawn(move || {
                for i in 0..iterations_per_thread {
                    let model_name = format!("concurrent-model-{tid}-{i}");

                    // Conservative default
                    let default_prof = ModelProfile::unprofiled_default(&model_name);
                    assert!(default_prof.is_unprofiled());
                    let entry = default_prof.to_model_entry(1_700_000_000);
                    assert!(entry.unprofiled);

                    let entry_json = serde_json::to_string(&entry).unwrap();
                    let entry_deser: ModelEntry = serde_json::from_str(&entry_json).unwrap();
                    assert_eq!(entry_deser.id, model_name);

                    // Audit event
                    let audit = ModelLoadAuditEvent::from_tuning(&model_name, &default_prof.tuning);
                    let audit_json = serde_json::to_string(&audit).unwrap();
                    let audit_deser: ModelLoadAuditEvent =
                        serde_json::from_str(&audit_json).unwrap();
                    assert_eq!(audit_deser.model_id, model_name);

                    // Operator tuning
                    let mut tuned_prof = default_prof;
                    let mut tuned = TuningProfile::conservative_default();
                    tuned.apply_operator_tuning(
                        ComputePool::Both,
                        6,
                        8,
                        PriorityClass::Latency,
                        4096,
                        false,
                        false,
                    );
                    tuned_prof.apply_tuning(tuned);
                    assert!(!tuned_prof.is_unprofiled());

                    let tuned_entry = tuned_prof.to_model_entry(1_700_000_001);
                    assert!(!tuned_entry.unprofiled);
                    assert!(tuned_entry.advisory_notice.is_none());
                }
            })
        })
        .collect();

    for h in handles {
        h.join().expect("thread join");
    }
}

/// 9. Baseline `CostProfile::unprofiled_default` invariants.
#[test]
fn test_challenge_cost_profile_unprofiled_baseline() {
    let identity = ModelIdentity {
        name: "test-model".to_owned(),
        architecture: "test-arch".to_owned(),
        file_path: "/tmp/model.gguf".to_owned(),
        file_digest: "1234abcd".to_owned(),
    };

    let cost = CostProfile::unprofiled_default(identity.clone());
    assert_eq!(cost.model_identity, identity);
    assert_eq!(cost.cluster_placement.cluster, "x100");
    assert_eq!(cost.cluster_placement.x100_cores, 6);
    assert_eq!(cost.cluster_placement.a100_cores, 0);
    assert_eq!(cost.allocated_cores.x100_cores, 6);
    assert_eq!(cost.allocated_cores.a100_cores, 0);
    assert_eq!(cost.quantization_level.primary, "unprofiled");
    assert_eq!(cost.tuning_advisory.recommended_cluster, "x100");
    assert!(!cost.tuning_advisory.recommended_ime);

    // Serialization roundtrip
    let json = cost.to_json_pretty().expect("serialize cost profile");
    let deser = CostProfile::from_json_str(&json).expect("deserialize cost profile");
    assert_eq!(cost, deser);
}
