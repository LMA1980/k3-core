//! Empirical Stress & Adversarial Challenge Harness for Milestone M1 Iteration 2
//!
//! Specifically tests:
//! 1. Arithmetic overflow & extreme core count inputs across all resolution branches
//! 2. Offline core rejection across active, spare, and custom compute cap masks
//! 3. Bitmask boundary conditions (u64::MAX, SoC mask violation, Cluster 1 isolation)
//! 4. Invariant preservation fuzzing under random combinatorial inputs

#![allow(
    clippy::all,
    clippy::pedantic,
    clippy::unreadable_literal,
    unused_parens,
    unused_imports
)]

use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::thread;

use k3_core::affinity::*;
use k3_core::error::K3Error;

#[test]
fn test_stress_arithmetic_overflow_extremes() {
    let extreme_pairs = [
        (usize::MAX, 1),
        (1, usize::MAX),
        (usize::MAX, usize::MAX),
        (usize::MAX, 0),
        (0, usize::MAX),
        (usize::MAX - 1, 1),
        (usize::MAX - 1, 2),
        (usize::MAX / 2 + 1, usize::MAX / 2 + 1),
        (usize::MAX / 2, usize::MAX / 2 + 2),
        (usize::MAX - 100, 200),
    ];

    for &(active, spare) in &extreme_pairs {
        let cfg = ControlPlaneCapacityConfig::new()
            .with_active_cores(active)
            .with_spare_cores(spare);

        // Test on standard online mask 0x00FF
        let res = catch_unwind(AssertUnwindSafe(|| cfg.resolve_default()));
        assert!(res.is_ok(), "resolve_default panicked on active={}, spare={}", active, spare);
        let outcome = res.unwrap();
        assert!(
            outcome.is_err(),
            "expected Err on extreme active={}, spare={}, got {:?}",
            active,
            spare,
            outcome
        );
        match outcome.unwrap_err() {
            K3Error::AffinityError { requested_threads, reason, .. } => {
                if active == 0 {
                    assert_eq!(requested_threads, 0);
                    assert!(reason.contains("at least 1 active control core"));
                } else {
                    assert!(
                        reason.contains("overflow") || reason.contains("exceeds"),
                        "unexpected reason for active={}, spare={}: {}",
                        active,
                        spare,
                        reason
                    );
                }
            }
            other => panic!("expected AffinityError, got {:?}", other),
        }

        // Test on degraded online mask 0x0001
        let res_deg = catch_unwind(AssertUnwindSafe(|| cfg.resolve_with_online_mask(0x0001)));
        assert!(res_deg.is_ok(), "resolve_with_online_mask(0x0001) panicked on active={}, spare={}", active, spare);
        assert!(res_deg.unwrap().is_err());

        // Test on empty online mask 0x0000
        let res_empty = catch_unwind(AssertUnwindSafe(|| cfg.resolve_with_online_mask(0x0000)));
        assert!(res_empty.is_ok(), "resolve_with_online_mask(0x0000) panicked on active={}, spare={}", active, spare);
        assert!(res_empty.unwrap().is_err());
    }
}

#[test]
fn test_stress_arithmetic_overflow_with_custom_spare_mask() {
    let extreme_actives = [
        usize::MAX,
        usize::MAX - 1,
        usize::MAX / 2 + 1,
        9,
        100,
        1_000_000,
    ];

    for &act in &extreme_actives {
        let cfg = ControlPlaneCapacityConfig::new()
            .with_active_cores(act)
            .with_custom_spare_mask(0x0020);

        let res = catch_unwind(AssertUnwindSafe(|| cfg.resolve_default()));
        assert!(res.is_ok(), "panicked with custom spare and active={}", act);
        let outcome = res.unwrap();
        assert!(outcome.is_err(), "expected error for active={}, got {:?}", act, outcome);
        match outcome.unwrap_err() {
            K3Error::AffinityError { requested_threads, reason, .. } => {
                assert_eq!(requested_threads, act);
                assert!(reason.contains("exceeds") && reason.contains("available online X100 cores"));
            }
            other => panic!("expected AffinityError, got {:?}", other),
        }
    }
}

#[test]
fn test_stress_offline_core_rejection_all_8_single_bits() {
    // For each core 0..8, test when that single core is offline while being requested in masks.
    for core in 0..8 {
        let offline_bit = 1u64 << core;
        let online_mask = X100_CLUSTER_MASK & !offline_bit;

        // 1. Custom active mask containing the offline core
        let cfg_act = ControlPlaneCapacityConfig::default().with_custom_active_mask(offline_bit);
        let res_act = cfg_act.resolve_with_online_mask(online_mask);
        assert!(res_act.is_err(), "core {} offline was accepted in active mask", core);
        match res_act.unwrap_err() {
            K3Error::AffinityError { requested_mask, reason, .. } => {
                assert_eq!(requested_mask, offline_bit);
                assert!(reason.contains("specifies offline cores"));
            }
            other => panic!("unexpected error {:?}", other),
        }

        // 2. Custom spare mask containing the offline core
        let cfg_spr = ControlPlaneCapacityConfig::default()
            .with_custom_active_mask(if core == 7 { 0x0040 } else { 0x0080 })
            .with_custom_spare_mask(offline_bit);
        let res_spr = cfg_spr.resolve_with_online_mask(online_mask);
        assert!(res_spr.is_err(), "core {} offline was accepted in spare mask", core);
        match res_spr.unwrap_err() {
            K3Error::AffinityError { requested_mask, reason, .. } => {
                assert_eq!(requested_mask, offline_bit);
                assert!(reason.contains("specifies offline cores"));
            }
            other => panic!("unexpected error {:?}", other),
        }

        // 3. Custom compute cap mask containing the offline core
        let active_core = if core == 7 { 6 } else { 7 };
        let cfg_cap = ControlPlaneCapacityConfig::default()
            .with_custom_active_mask(1u64 << active_core)
            .with_custom_compute_cap_mask(offline_bit);
        let res_cap = cfg_cap.resolve_with_online_mask(online_mask);
        assert!(res_cap.is_err(), "core {} offline was accepted in compute cap mask", core);
        match res_cap.unwrap_err() {
            K3Error::AffinityError { requested_mask, reason, .. } => {
                assert_eq!(requested_mask, offline_bit);
                assert!(reason.contains("specifies offline cores"));
            }
            other => panic!("unexpected error {:?}", other),
        }
    }
}

#[test]
fn test_stress_offline_cores_combinatorial_patterns() {
    let degraded_topologies = [
        (0x000F, 0x00F0), // bottom 4 online, top 4 offline
        (0x00F0, 0x000F), // top 4 online, bottom 4 offline
        (0x0055, 0x00AA), // alternating online/offline
        (0x00AA, 0x0055), // alternating offline/online
        (0x0001, 0x00FE), // 1 core online, 7 offline
        (0x0080, 0x007F), // 1 core online, 7 offline
    ];

    for &(online_mask, offline_mask) in &degraded_topologies {
        // Requesting compute cap specifying all offline cores
        let cfg = ControlPlaneCapacityConfig::minimal()
            .with_custom_compute_cap_mask(offline_mask);
        let res = cfg.resolve_with_online_mask(online_mask);
        assert!(res.is_err());
        assert!(res.unwrap_err().to_string().contains("specifies offline cores"));

        // Requesting compute cap specifying 1 online core + 1 offline core
        let lowest_online = 1u64 << online_mask.trailing_zeros();
        let lowest_offline = 1u64 << offline_mask.trailing_zeros();
        let mixed_cap = lowest_online | lowest_offline;

        let cfg_mixed = ControlPlaneCapacityConfig::minimal()
            .with_custom_compute_cap_mask(mixed_cap);
        let res_mixed = cfg_mixed.resolve_with_online_mask(online_mask);
        assert!(res_mixed.is_err());
        assert!(res_mixed.unwrap_err().to_string().contains("specifies offline cores"));
    }
}

#[test]
fn test_stress_validate_compute_mask_boundaries() {
    let cfg = ControlPlaneCapacityConfig::default();
    let res = cfg.resolve_default().expect("resolve default");

    // Valid compute mask: subset of default compute cap (cores 0..5, 0x003F)
    assert!(res.validate_compute_mask(0x0001).is_ok());
    assert!(res.validate_compute_mask(0x003F).is_ok());
    assert!(res.validate_compute_mask(0x0020).is_ok()); // Core 5 is lendable spare

    // Valid on A100: Cluster 1 cores (0xFF00)
    assert!(res.validate_compute_mask(0xFF00).is_ok());
    assert!(res.validate_compute_mask(0x003F | 0xFF00).is_ok());

    // Invalid: Overlaps active control cores (6, 7 -> 0x0040, 0x0080, 0x00C0)
    assert!(res.validate_compute_mask(0x0040).is_err());
    assert!(res.validate_compute_mask(0x0080).is_err());
    assert!(res.validate_compute_mask(0x00C0).is_err());
    assert!(res.validate_compute_mask(0x0001 | 0x0080).is_err());

    // Invalid: Out of SoC range (outside 0xFFFF)
    let oob_masks = [
        1u64 << 16,
        1u64 << 17,
        1u64 << 63,
        u64::MAX,
        0xFFFF_0000,
        0x0001_0000,
    ];
    for &oob in &oob_masks {
        let err = res.validate_compute_mask(oob);
        assert!(err.is_err(), "validate_compute_mask accepted out of bounds mask 0x{:X}", oob);
        let msg = err.unwrap_err().to_string();
        assert!(msg.contains("outside valid 16-core SoC mask") || msg.contains("invalid cores"));
    }
}

#[test]
fn test_stress_concurrent_rotator_near_usize_max_wrap() {
    let cfg = ControlPlaneCapacityConfig::default();
    let res = Arc::new(cfg.resolve_default().expect("resolve default"));

    // Set counter near usize::MAX to provoke wraparound under high concurrency
    res.round_robin_counter.store(usize::MAX - 50_000, Ordering::SeqCst);

    let threads: Vec<_> = (0..16)
        .map(|_| {
            let res_clone = Arc::clone(&res);
            thread::spawn(move || {
                for _ in 0..10_000 {
                    let core = res_clone.next_active_control_core();
                    assert!(core == 6 || core == 7, "invalid core returned: {}", core);
                }
            })
        })
        .collect();

    for t in threads {
        t.join().unwrap();
    }
}

#[test]
fn test_stress_fuzz_random_configurations_and_invariants() {
    // Deterministic pseudo-random sequence (Linear Congruential Generator)
    let mut state = 0xDEAD_BEEF_u64;
    let mut next_u64 = || {
        state = state.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
        state
    };

    for _ in 0..5_000 {
        let active = (next_u64() % 12) as usize; // 0..11
        let spare = (next_u64() % 12) as usize;  // 0..11
        let online_mask = next_u64() & 0x03FF;   // 10-bit mask (some in X100, some in A100)

        let mut cfg = ControlPlaneCapacityConfig::new()
            .with_active_cores(active)
            .with_spare_cores(spare);

        // Randomly set custom masks
        if (next_u64() % 3) == 0 {
            let custom_act = next_u64() & 0x01FF;
            cfg = cfg.with_custom_active_mask(custom_act);
        }
        if (next_u64() % 3) == 0 {
            let custom_spr = next_u64() & 0x01FF;
            cfg = cfg.with_custom_spare_mask(custom_spr);
        }
        if (next_u64() % 3) == 0 {
            let custom_cap = next_u64() & 0x01FF;
            cfg = cfg.with_custom_compute_cap_mask(custom_cap);
        }

        let res = catch_unwind(AssertUnwindSafe(|| {
            cfg.resolve_with_online_mask(online_mask)
        }));

        assert!(res.is_ok(), "fuzzer triggered panic on cfg: {:?}", cfg);

        if let Ok(Ok(resolved)) = res {
            let valid_online = online_mask & X100_CLUSTER_MASK;
            // Assert all fundamental invariants
            assert!(!resolved.active_control_cores.is_empty(), "active cores must not be empty");
            assert_ne!(resolved.active_control_mask, 0, "active mask must not be 0");
            assert_eq!(
                resolved.active_control_mask & resolved.spare_recovery_mask,
                0,
                "active and spare must be disjoint"
            );
            assert_eq!(
                resolved.active_control_mask & resolved.x100_compute_cap_mask,
                0,
                "active and compute cap must be disjoint"
            );
            assert_eq!(
                resolved.active_control_mask & !valid_online,
                0,
                "active cores must be online"
            );
            assert_eq!(
                resolved.spare_recovery_mask & !valid_online,
                0,
                "spare cores must be online"
            );
            assert_eq!(
                resolved.x100_compute_cap_mask & !valid_online,
                0,
                "compute cap cores must be online"
            );

            // Test rotator methods do not panic
            let next_c = resolved.next_active_control_core();
            assert!(resolved.is_active_control_core(next_c));
            assert_eq!(resolved.round_robin_active_core(0), resolved.active_control_cores[0]);
        }
    }
}
