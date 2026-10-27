//! Integration tests for US-07: Safe Conservative Defaults for Unprofiled Models (k3-core).
//!
//! Validates:
//! - Scenario 1: Conservative placement on initial model load (X100 only, `a100_cores` == 0,
//!   throughput priority class, bounded `n_ctx`, lending/borrowing disabled).
//! - Scenario 2: Explicit unprofiled status surfacing (`/v1/models` representation, advisory notice,
//!   operational audit event generation, and operator profile transition).
//! - Crate-boundary contracts: JSON serialization, parsing, error diagnostics, and topology bounds.

use k3_core::{
    A100_CLUSTER_MASK, CONSERVATIVE_DEFAULT_A100_CORES, CONSERVATIVE_DEFAULT_N_CTX,
    CONSERVATIVE_DEFAULT_X100_CORES, CONSERVATIVE_TUNING_ADVISORY, CONSTRAINED_TEST_N_CTX,
    ComputePool, DEFAULT_ACTIVE_CONTROL_MASK, DEFAULT_X100_COMPUTE_CAP_MASK, K3Error, ModelEntry,
    ModelListResponse, ModelLoadAuditEvent, ModelProfile, PriorityClass, ProfileStatus,
    TuningProfile, X100_CLUSTER_MASK,
};

#[test]
fn test_scenario1_conservative_defaults_invariants() {
    let profile = TuningProfile::conservative_default();

    // 1. Compute cluster placement: General-purpose X100 pool only
    assert_eq!(profile.compute_pool, ComputePool::X100);
    assert_eq!(profile.cluster, "x100");
    assert_eq!(profile.x100_cores, CONSERVATIVE_DEFAULT_X100_CORES);
    assert_eq!(profile.x100_cores, 6);
    assert_eq!(profile.a100_cores, CONSERVATIVE_DEFAULT_A100_CORES);
    assert_eq!(profile.a100_cores, 0);

    // 2. Hardware mask alignment: X100 compute cap mask (0x003F) does not overlap active control (0x00C0)
    let compute_cap = DEFAULT_X100_COMPUTE_CAP_MASK;
    assert_eq!(compute_cap, 0x003F);
    assert_eq!(compute_cap & DEFAULT_ACTIVE_CONTROL_MASK, 0);
    assert_eq!(profile.compute_pool.core_mask(), X100_CLUSTER_MASK);

    // 3. Priority class: Standard throughput priority
    assert_eq!(profile.priority_class, PriorityClass::Throughput);
    assert_eq!(profile.priority_class.as_str(), "throughput");

    // 4. Context window bound: Conservative default n_ctx = 2048
    assert_eq!(profile.n_ctx, CONSERVATIVE_DEFAULT_N_CTX);
    assert_eq!(profile.n_ctx, 2048);

    // 5. Core spillover: Dynamic idle core lending and borrowing disabled
    assert!(!profile.lend_idle_cores);
    assert!(!profile.borrow_idle_cores);

    // 6. Status and advisory notice
    assert!(profile.unprofiled);
    assert!(profile.is_unprofiled());
    assert_eq!(profile.profile_status, ProfileStatus::Unprofiled);
    assert_eq!(
        profile.advisory_notice.as_deref(),
        Some(CONSERVATIVE_TUNING_ADVISORY)
    );
}

#[test]
fn test_scenario1_constrained_test_harness_override() {
    // In constrained test harnesses (AGENTS.md Rule 5), n_ctx can be overridden to 64
    let profile = TuningProfile::conservative_default_with_n_ctx(CONSTRAINED_TEST_N_CTX);

    assert_eq!(profile.n_ctx, 64);
    assert_eq!(profile.n_ctx, CONSTRAINED_TEST_N_CTX);
    assert_eq!(profile.compute_pool, ComputePool::X100);
    assert_eq!(profile.x100_cores, 6);
    assert_eq!(profile.a100_cores, 0);
    assert_eq!(profile.priority_class, PriorityClass::Throughput);
    assert!(!profile.lend_idle_cores);
    assert!(!profile.borrow_idle_cores);
    assert!(profile.unprofiled);

    let model_prof = ModelProfile::unprofiled_with_n_ctx("qwen2.5-0.5b", CONSTRAINED_TEST_N_CTX);
    assert_eq!(model_prof.model_id, "qwen2.5-0.5b");
    assert_eq!(model_prof.tuning.n_ctx, 64);
    assert!(model_prof.is_unprofiled());
}

#[test]
fn test_scenario2_explicit_unprofiled_status_surfacing() {
    let model = ModelProfile::unprofiled_default("qwen2.5-7b-instruct");
    assert!(model.is_unprofiled());

    let timestamp = 1_700_000_123_u64;
    let entry: ModelEntry = model.to_model_entry(timestamp);

    // Validate fields surfaced in /v1/models listing
    assert_eq!(entry.id, "qwen2.5-7b-instruct");
    assert_eq!(entry.object, "model");
    assert_eq!(entry.created, timestamp);
    assert_eq!(entry.owned_by, "k3");
    assert!(entry.unprofiled);
    assert_eq!(entry.profile_status, "unprofiled");
    assert_eq!(entry.priority_class, "throughput");
    assert_eq!(entry.cluster, "x100");
    assert_eq!(entry.a100_cores, 0);
    assert_eq!(entry.n_ctx, 2048);
    assert!(!entry.lend_idle_cores);
    assert!(!entry.borrow_idle_cores);
    assert_eq!(
        entry.advisory_notice.as_deref(),
        Some(CONSERVATIVE_TUNING_ADVISORY)
    );

    // Validate JSON representation
    let json = serde_json::to_string_pretty(&entry).expect("serialize ModelEntry");
    assert!(json.contains("\"unprofiled\": true"));
    assert!(json.contains("\"profile_status\": \"unprofiled\""));
    assert!(json.contains("\"priority_class\": \"throughput\""));
    assert!(json.contains("\"cluster\": \"x100\""));
    assert!(json.contains("\"a100_cores\": 0"));
    assert!(json.contains("\"n_ctx\": 2048"));
    assert!(json.contains("\"lend_idle_cores\": false"));
    assert!(json.contains("\"borrow_idle_cores\": false"));
    assert!(json.contains(CONSERVATIVE_TUNING_ADVISORY));

    // Validate ModelListResponse collection
    let list_response = ModelListResponse::new(vec![entry]);
    let list_json = serde_json::to_string_pretty(&list_response).expect("serialize list");
    assert!(list_json.contains("\"object\": \"list\""));
    assert!(list_json.contains("\"id\": \"qwen2.5-7b-instruct\""));
}

#[test]
fn test_scenario2_structured_audit_event_generation() {
    let unprofiled_tuning = TuningProfile::conservative_default();
    let audit_event = ModelLoadAuditEvent::from_tuning("deepseek-r1-7b", &unprofiled_tuning);

    assert_eq!(audit_event.model_id, "deepseek-r1-7b");
    assert!(audit_event.unprofiled);
    assert_eq!(audit_event.profile_status, "unprofiled");
    assert_eq!(audit_event.cluster, "x100");
    assert_eq!(audit_event.a100_cores, 0);
    assert_eq!(audit_event.priority_class, "throughput");
    assert_eq!(audit_event.n_ctx, 2048);
    assert_eq!(
        audit_event.advisory_notice.as_deref(),
        Some(CONSERVATIVE_TUNING_ADVISORY)
    );

    let event_json = serde_json::to_string(&audit_event).expect("serialize audit event");
    assert!(event_json.contains("\"unprofiled\":true"));
    assert!(event_json.contains("\"profile_status\":\"unprofiled\""));
    assert!(event_json.contains("\"model_id\":\"deepseek-r1-7b\""));
}

#[test]
fn test_scenario2_operator_tuning_transition_clears_unprofiled() {
    let mut model = ModelProfile::unprofiled_default("qwen2.5-14b");
    assert!(model.is_unprofiled());

    // Operator applies explicit tuning profile
    let mut tuned = TuningProfile::conservative_default();
    tuned.apply_operator_tuning(
        ComputePool::Both,
        6,
        8,
        PriorityClass::Latency,
        8192,
        true,
        true,
    );

    model.apply_tuning(tuned);

    // Verify unprofiled status is cleared
    assert!(!model.is_unprofiled());
    assert!(!model.tuning.unprofiled);
    assert_eq!(model.tuning.profile_status, ProfileStatus::Profiled);
    assert_eq!(model.tuning.compute_pool, ComputePool::Both);
    assert_eq!(model.tuning.cluster, "both");
    assert_eq!(model.tuning.x100_cores, 6);
    assert_eq!(model.tuning.a100_cores, 8);
    assert_eq!(model.tuning.priority_class, PriorityClass::Latency);
    assert_eq!(model.tuning.n_ctx, 8192);
    assert!(model.tuning.lend_idle_cores);
    assert!(model.tuning.borrow_idle_cores);
    assert!(model.tuning.advisory_notice.is_none());

    // Verify surfaced ModelEntry reflects tuned state with advisory removed
    let entry = model.to_model_entry(1_700_000_500);
    assert!(!entry.unprofiled);
    assert_eq!(entry.profile_status, "profiled");
    assert_eq!(entry.priority_class, "latency");
    assert_eq!(entry.cluster, "both");
    assert_eq!(entry.a100_cores, 8);
    assert_eq!(entry.n_ctx, 8192);
    assert!(entry.lend_idle_cores);
    assert!(entry.borrow_idle_cores);
    assert!(entry.advisory_notice.is_none());

    let entry_json = serde_json::to_string(&entry).expect("serialize tuned entry");
    assert!(entry_json.contains("\"unprofiled\":false"));
    assert!(entry_json.contains("\"profile_status\":\"profiled\""));
    assert!(!entry_json.contains("advisory_notice"));

    // Verify audit event reflects tuned profile
    let tuned_audit = ModelLoadAuditEvent::from_tuning("qwen2.5-14b", &model.tuning);
    assert!(!tuned_audit.unprofiled);
    assert_eq!(tuned_audit.profile_status, "profiled");
    assert!(tuned_audit.advisory_notice.is_none());
}

#[test]
fn test_parsing_and_error_diagnostics() {
    // PriorityClass parsing
    assert_eq!(
        PriorityClass::parse("throughput").unwrap(),
        PriorityClass::Throughput
    );
    assert_eq!(
        PriorityClass::parse("latency").unwrap(),
        PriorityClass::Latency
    );
    let bad_priority = PriorityClass::parse("realtime");
    assert!(matches!(
        bad_priority,
        Err(K3Error::CapabilityMismatch { .. })
    ));

    // ComputePool parsing
    assert_eq!(ComputePool::parse("x100").unwrap(), ComputePool::X100);
    assert_eq!(ComputePool::parse("a100").unwrap(), ComputePool::A100);
    assert_eq!(ComputePool::parse("both").unwrap(), ComputePool::Both);
    let bad_pool = ComputePool::parse("gpu");
    assert!(matches!(bad_pool, Err(K3Error::CapabilityMismatch { .. })));

    // ProfileStatus parsing
    assert_eq!(
        ProfileStatus::parse("unprofiled").unwrap(),
        ProfileStatus::Unprofiled
    );
    assert_eq!(
        ProfileStatus::parse("profiled").unwrap(),
        ProfileStatus::Profiled
    );
    let bad_status = ProfileStatus::parse("unknown");
    assert!(matches!(
        bad_status,
        Err(K3Error::CapabilityMismatch { .. })
    ));
}

#[test]
fn test_topology_mask_and_pool_capacity() {
    assert_eq!(ComputePool::X100.core_mask(), X100_CLUSTER_MASK);
    assert_eq!(ComputePool::A100.core_mask(), A100_CLUSTER_MASK);
    assert_eq!(
        ComputePool::Both.core_mask(),
        X100_CLUSTER_MASK | A100_CLUSTER_MASK
    );

    assert_eq!(ComputePool::X100.default_max_compute_cores(), 6);
    assert_eq!(ComputePool::A100.default_max_compute_cores(), 8);
    assert_eq!(ComputePool::Both.default_max_compute_cores(), 14);
}
