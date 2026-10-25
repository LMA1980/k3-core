//! Empirical Challenge Fuzzing Suite for Milestone M1 (`k3-core` Capacity & Resolver)
//!
//! Conducts exhaustive state-space verification across all 256 8-bit topology permutations,
//! full parameter sweeps across active/spare configurations, and stress-tests edge-case boundaries
//! for panics, integer overflows, and invariant violations.

#![allow(clippy::doc_markdown, clippy::uninlined_format_args)]

use std::sync::Arc;
use std::thread;

use k3_core::affinity::*;
use k3_core::error::K3Error;

/// Verification 1: Exhaustive iteration over all 256 8-bit online masks with default configuration.
#[test]
fn test_exhaustive_256_online_masks_default_config() {
    let cfg = ControlPlaneCapacityConfig::default();

    for mask_val in 0u64..=255u64 {
        let res = cfg.resolve_with_online_mask(mask_val);
        let online_count = mask_val.count_ones() as usize;

        if online_count < 3 {
            // Default config requests 2 active + 1 spare = 3 cores.
            // When fewer than 3 cores are online (or 0 cores online), it must fail with K3Error::AffinityError.
            assert!(
                res.is_err(),
                "mask 0x{:02X} has {} online cores, but resolve succeeded unexpectedly",
                mask_val,
                online_count
            );
            match res.unwrap_err() {
                K3Error::AffinityError {
                    requested_mask,
                    requested_threads,
                    reason,
                } => {
                    assert_eq!(requested_mask, mask_val);
                    if mask_val == 0 {
                        assert_eq!(requested_threads, 0);
                        assert!(reason.contains("no online X100 general-purpose cores available"));
                    } else {
                        assert_eq!(requested_threads, 3);
                        assert!(reason.contains("exceeds"));
                    }
                }
                other => panic!(
                    "expected AffinityError for mask 0x{:02X}, got {:?}",
                    mask_val, other
                ),
            }
        } else {
            // online_count >= 3: resolution must succeed and satisfy all invariants.
            let resolved = res.unwrap_or_else(|e| {
                panic!(
                    "mask 0x{:02X} ({} cores online) failed resolution: {:?}",
                    mask_val, online_count, e
                )
            });

            // Invariant 1: Exactly 2 active cores and 1 spare core
            assert_eq!(resolved.active_control_cores.len(), 2);
            assert_eq!(resolved.active_control_mask.count_ones(), 2);
            assert_eq!(resolved.spare_recovery_cores.len(), 1);
            assert_eq!(resolved.spare_recovery_mask.count_ones(), 1);

            // Invariant 2: Active cores and spare cores are disjoint
            assert_eq!(
                resolved.active_control_mask & resolved.spare_recovery_mask,
                0,
                "active and spare masks must be disjoint for mask 0x{:02X}",
                mask_val
            );

            // Invariant 3: Active cores and compute cap are strictly disjoint
            assert_eq!(
                resolved.active_control_mask & resolved.x100_compute_cap_mask,
                0,
                "active and compute cap masks must be disjoint for mask 0x{:02X}",
                mask_val
            );

            // Invariant 4: Spare recovery core is lendable (contained within compute cap)
            assert_eq!(
                resolved.spare_recovery_mask & resolved.x100_compute_cap_mask,
                resolved.spare_recovery_mask,
                "spare recovery core must be contained within compute cap for mask 0x{:02X}",
                mask_val
            );

            // Invariant 5: Total compute cores equals online cores minus dedicated active control cores
            assert_eq!(
                resolved.x100_max_compute_cores,
                online_count - 2,
                "max compute cores mismatch for mask 0x{:02X}",
                mask_val
            );
            assert_eq!(
                resolved.x100_compute_cap_mask.count_ones() as usize,
                resolved.x100_max_compute_cores
            );

            // Invariant 6: A100 compute capacity is always preserved at 8
            assert_eq!(resolved.a100_max_compute_cores, 8);

            // Invariant 7: spare_recovery_core() returns Some
            assert!(resolved.spare_recovery_core().is_some());
            let spare_core = resolved.spare_recovery_core().unwrap();
            assert_eq!(resolved.spare_recovery_cores[0], spare_core);
            assert!(resolved.is_spare_recovery_core(spare_core));
            assert!(resolved.is_x100_compute_allowed(spare_core));
            assert!(!resolved.is_active_control_core(spare_core));

            // Invariant 8: Round-robin selection never panics and rotates across active cores
            let c0 = resolved.next_active_control_core();
            let c1 = resolved.next_active_control_core();
            let c2 = resolved.next_active_control_core();
            assert!(resolved.is_active_control_core(c0));
            assert!(resolved.is_active_control_core(c1));
            assert_eq!(c0, c2);
            assert_ne!(c0, c1);

            // Invariant 9: Compute mask validation
            assert!(resolved.validate_compute_mask(resolved.x100_compute_cap_mask).is_ok());
            assert!(resolved.validate_compute_mask(resolved.active_control_mask).is_err());
        }
    }
}

/// Verification 2: Exhaustive iteration over all 256 8-bit online masks with minimal configuration (1 active, 0 spare).
#[test]
fn test_exhaustive_256_online_masks_minimal_config() {
    let cfg = ControlPlaneCapacityConfig::minimal();

    for mask_val in 0u64..=255u64 {
        let res = cfg.resolve_with_online_mask(mask_val);
        let online_count = mask_val.count_ones() as usize;

        if online_count == 0 {
            assert!(res.is_err(), "mask 0x00 must fail resolution");
            match res.unwrap_err() {
                K3Error::AffinityError { reason, .. } => {
                    assert!(reason.contains("no online X100 general-purpose cores available"));
                }
                other => panic!("expected AffinityError, got {:?}", other),
            }
        } else {
            let resolved = res.unwrap_or_else(|e| {
                panic!(
                    "minimal config failed on mask 0x{:02X}: {:?}",
                    mask_val, e
                )
            });

            // Invariant 1: Exactly 1 active core and 0 spare cores
            assert_eq!(resolved.active_control_cores.len(), 1);
            assert_eq!(resolved.active_control_mask.count_ones(), 1);
            assert_eq!(resolved.spare_recovery_cores.len(), 0);
            assert_eq!(resolved.spare_recovery_mask, 0);
            assert_eq!(resolved.spare_recovery_core(), None);

            // Invariant 2: Active core and compute cap are disjoint
            assert_eq!(
                resolved.active_control_mask & resolved.x100_compute_cap_mask,
                0
            );

            // Invariant 3: Compute cores count
            assert_eq!(
                resolved.x100_max_compute_cores,
                online_count - 1
            );

            // Invariant 4: Round-robin returns the sole active core
            let sole_active = resolved.active_control_cores[0];
            assert_eq!(resolved.next_active_control_core(), sole_active);
            assert_eq!(resolved.next_active_control_core(), sole_active);
            assert_eq!(resolved.round_robin_active_core(42), sole_active);
        }
    }
}

/// Verification 3: Full 2D Parameter Sweep (active: 0..=8, spare: 0..=8) across all 256 masks.
/// 9 x 9 x 256 = 20,736 total resolution invocations.
#[test]
fn test_exhaustive_parameter_sweep_20736_cases() {
    for active in 0..=8 {
        for spare in 0..=8 {
            let cfg = ControlPlaneCapacityConfig::new()
                .with_active_cores(active)
                .with_spare_cores(spare);

            for mask_val in 0u64..=255u64 {
                let res = cfg.resolve_with_online_mask(mask_val);
                let online_count = mask_val.count_ones() as usize;

                if online_count == 0 {
                    assert!(res.is_err());
                } else if active == 0 {
                    // Requirement R3: 0 active control cores must be rejected
                    assert!(
                        res.is_err(),
                        "active=0, spare={}, mask=0x{:02X} must be rejected",
                        spare,
                        mask_val
                    );
                    let err = res.unwrap_err();
                    assert_eq!(err.diagnostic_code(), "affinity_error");
                } else if active + spare > online_count {
                    // Requested cores exceed online cores: must be rejected
                    assert!(
                        res.is_err(),
                        "active={}, spare={}, online={} must be rejected",
                        active,
                        spare,
                        online_count
                    );
                    let err = res.unwrap_err();
                    assert_eq!(err.diagnostic_code(), "affinity_error");
                } else {
                    // Valid configuration: must succeed and satisfy all invariants
                    let resolved = res.unwrap_or_else(|e| {
                        panic!(
                            "failed for active={}, spare={}, mask=0x{:02X}: {:?}",
                            active, spare, mask_val, e
                        )
                    });

                    assert_eq!(resolved.active_control_cores.len(), active);
                    assert_eq!(resolved.active_control_mask.count_ones() as usize, active);
                    assert_eq!(resolved.spare_recovery_cores.len(), spare);
                    assert_eq!(resolved.spare_recovery_mask.count_ones() as usize, spare);

                    assert_eq!(
                        resolved.active_control_mask & resolved.spare_recovery_mask,
                        0
                    );
                    assert_eq!(
                        resolved.active_control_mask & resolved.x100_compute_cap_mask,
                        0
                    );

                    // Ensure round-robin never panics
                    let next = resolved.next_active_control_core();
                    assert!(resolved.is_active_control_core(next));
                }
            }
        }
    }
}

/// Verification 4: Regression Test — active control cores auto-allocated when custom_spare_recovery_mask provided alone.
/// When custom_spare_recovery_mask is set without custom_active_control_mask,
/// active control cores are auto-allocated from remaining online cores, preventing zero active cores and panic.
/// When insufficient cores exist to allocate active cores, resolution rejects with K3Error::AffinityError.
#[test]
fn test_demonstrate_bug_zero_active_cores_resolved_and_panic() {
    test_regression_zero_active_cores_resolved_and_panic();
}

#[test]
fn test_regression_zero_active_cores_resolved_and_panic() {
    let cfg = ControlPlaneCapacityConfig::default()
        .with_custom_spare_mask(0x0020);

    let res = cfg.resolve_default();
    assert!(res.is_ok(), "resolve_default should succeed with auto-allocated active cores");

    let resolved = res.unwrap();
    assert_eq!(resolved.active_control_mask, 0x00C0);
    assert_eq!(resolved.active_control_cores, vec![6, 7]);
    assert_eq!(resolved.spare_recovery_mask, 0x0020);
    assert_eq!(resolved.spare_recovery_core(), Some(5));

    // Calling next_active_control_core() succeeds safely without panic
    let c1 = resolved.next_active_control_core();
    let c2 = resolved.next_active_control_core();
    assert!(resolved.is_active_control_core(c1));
    assert!(resolved.is_active_control_core(c2));

    // Calling round_robin_active_core succeeds safely
    assert_eq!(resolved.round_robin_active_core(0), 6);
    assert_eq!(resolved.round_robin_active_core(1), 7);

    // Rotator next_core succeeds safely
    let rot = resolved.rotator();
    assert_eq!(rot.next_core(), 6);
    assert_eq!(rot.next_core(), 7);

    // When custom spare mask exhausts available online cores, resolution returns Err(K3Error::AffinityError)
    let constrained = ControlPlaneCapacityConfig::minimal()
        .with_custom_spare_mask(0x0020);
    let err = constrained.resolve_with_online_mask(0x0020).unwrap_err();
    match err {
        K3Error::AffinityError { requested_mask, requested_threads, reason } => {
            assert_eq!(requested_mask, 0x0020);
            assert_eq!(requested_threads, 1);
            assert!(reason.contains("available online X100 cores") || reason.contains("active control core"));
        }
        other => panic!("expected AffinityError, got {:?}", other),
    }
}

/// Verification 5: Regression Test — Integer overflow on active + spare returns K3Error::AffinityError without panicking.
#[test]
fn test_demonstrate_bug_integer_overflow_panic() {
    test_regression_integer_overflow_returns_affinity_error();
}

#[test]
fn test_regression_integer_overflow_returns_affinity_error() {
    let cfg = ControlPlaneCapacityConfig::new()
        .with_active_cores(usize::MAX)
        .with_spare_cores(1);

    let panic_res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        cfg.resolve_default()
    }));
    assert!(panic_res.is_ok(), "resolve_default must not panic on integer overflow");

    let res = panic_res.unwrap();
    assert!(res.is_err(), "resolve_default must return structured error on integer overflow");
    match res.unwrap_err() {
        K3Error::AffinityError { requested_threads, reason, .. } => {
            assert_eq!(requested_threads, usize::MAX);
            assert!(reason.contains("overflow"));
        }
        other => panic!("expected AffinityError, got {:?}", other),
    }
}

/// Verification 6: Regression Test — custom_x100_compute_cap_mask rejects offline cores.
#[test]
fn test_demonstrate_bug_custom_compute_cap_accepts_offline_cores() {
    test_regression_custom_compute_cap_rejects_offline_cores();
}

#[test]
fn test_regression_custom_compute_cap_rejects_offline_cores() {
    let cfg = ControlPlaneCapacityConfig::default()
        .with_custom_active_mask(0x00C0)
        .with_custom_compute_cap_mask(0x000F); // cores 0..3 are offline

    let res = cfg.resolve_with_online_mask(0x00C0);
    assert!(res.is_err(), "custom compute cap specifying offline cores must be rejected");
    match res.unwrap_err() {
        K3Error::AffinityError { requested_mask, requested_threads, reason } => {
            assert_eq!(requested_mask, 0x000F);
            assert_eq!(requested_threads, 4);
            assert!(reason.contains("specifies offline cores"));
        }
        other => panic!("expected AffinityError, got {:?}", other),
    }
}

/// Verification 7: Concurrency stress testing on next_active_control_core() across 16 threads.
#[test]
fn test_concurrent_rotator_stress_16_threads() {
    let cfg = ControlPlaneCapacityConfig::default();
    let res = Arc::new(cfg.resolve_default().expect("resolve default"));

    let handles: Vec<_> = (0..16)
        .map(|_| {
            let res_clone = Arc::clone(&res);
            thread::spawn(move || {
                let mut sum = 0usize;
                for _ in 0..10_000 {
                    let core = res_clone.next_active_control_core();
                    assert!(core == 6 || core == 7);
                    sum += core;
                }
                sum
            })
        })
        .collect();

    for h in handles {
        h.join().unwrap();
    }
}

/// Verification 8: Upper bit masking — online masks containing bits > 7 (e.g., A100 or higher bits).
#[test]
fn test_upper_bit_masking_cluster_isolation() {
    let cfg = ControlPlaneCapacityConfig::default();

    // 0x01FF has 0x00FF on X100 and bit 8 on A100.
    // The resolver must mask online_mask with X100_CLUSTER_MASK (0x00FF),
    // producing the exact same resolution as 0x00FF.
    let res_mixed = cfg.resolve_with_online_mask(0x01FF).expect("resolve 0x01FF");
    let res_pure = cfg.resolve_with_online_mask(0x00FF).expect("resolve 0x00FF");

    assert_eq!(res_mixed, res_pure);
    assert_eq!(res_mixed.active_control_mask, 0x00C0);
    assert_eq!(res_mixed.spare_recovery_mask, 0x0020);
    assert_eq!(res_mixed.x100_compute_cap_mask, 0x003F);
}

/// Verification 9: Round-robin counter wrapping near usize::MAX.
#[test]
fn test_round_robin_counter_wrapping_at_usize_max() {
    let cfg = ControlPlaneCapacityConfig::default();
    let res = cfg.resolve_default().expect("resolve default");

    res.round_robin_counter.store(usize::MAX - 1, std::sync::atomic::Ordering::Relaxed);

    let c0 = res.next_active_control_core();
    let c1 = res.next_active_control_core();
    let c2 = res.next_active_control_core();

    assert!(c0 == 6 || c0 == 7);
    assert!(c1 == 6 || c1 == 7);
    assert!(c2 == 6 || c2 == 7);
    assert_ne!(c0, c1);
    assert_eq!(c0, c2);
}
