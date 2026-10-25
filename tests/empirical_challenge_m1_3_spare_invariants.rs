//! Empirical Challenger Test Suite for Milestone M1 Iteration 2
//! Focus: Invariants of custom spare allocation without custom active mask across topologies.

#![allow(
    clippy::all,
    clippy::pedantic,
    clippy::nursery,
    clippy::undocumented_unsafe_blocks
)]

use std::sync::Arc;
use std::thread;

use k3_core::affinity::*;

/// Invariant Suite 1: Custom spare mask across ALL 256 online topologies with default active cores (2).
///
/// For every possible online mask in `0..=255`, and for every core `c` that is online in `online_mask`:
/// - We configure `ControlPlaneCapacityConfig::default().with_custom_spare_mask(1 << c)`.
/// - If `online_count < 3`: Must fail with `K3Error::AffinityError`.
/// - If `online_count >= 3`: Must succeed and strictly verify:
///   1. `res.active_control_cores.len() == 2`
///   2. `res.active_control_mask.count_ones() == 2`
///   3. `res.spare_recovery_cores == vec![c]`
///   4. `res.spare_recovery_mask == (1 << c)`
///   5. `(res.active_control_mask & res.spare_recovery_mask) == 0`
///   6. `(res.active_control_mask & res.x100_compute_cap_mask) == 0`
///   7. `(res.spare_recovery_mask & res.x100_compute_cap_mask) == res.spare_recovery_mask` (lendable spare)
///   8. `res.active_control_mask | res.x100_compute_cap_mask == online_mask`
///   9. Active control cores are chosen descending from highest available cores in `online_mask & !(1 << c)`
///   10. `next_active_control_core()` rotates deterministically between the 2 active cores.
#[test]
fn test_exhaustive_custom_spare_without_custom_active_default_2cores() {
    let mut total_tested = 0;
    let mut total_success = 0;
    let mut total_rejected = 0;

    for online_mask in 0u64..=255u64 {
        let online_count = online_mask.count_ones() as usize;

        for c in 0..8 {
            if (online_mask & (1 << c)) == 0 {
                continue;
            }
            total_tested += 1;
            let spare_bit = 1u64 << c;
            let cfg = ControlPlaneCapacityConfig::default().with_custom_spare_mask(spare_bit);

            match cfg.resolve_with_online_mask(online_mask) {
                Ok(res) => {
                    total_success += 1;
                    assert!(
                        online_count >= 3,
                        "Expected failure when online_count < 3, but succeeded for mask 0x{online_mask:02X}, spare core {c}"
                    );

                    // 1. Core count invariants
                    assert_eq!(res.active_control_cores.len(), 2);
                    assert_eq!(res.active_control_mask.count_ones(), 2);
                    assert_eq!(res.spare_recovery_cores, vec![c]);
                    assert_eq!(res.spare_recovery_mask, spare_bit);
                    assert_eq!(res.spare_recovery_core(), Some(c));

                    // 2. Disjointness invariants
                    assert_eq!(
                        res.active_control_mask & res.spare_recovery_mask,
                        0,
                        "Active mask 0x{:04X} overlaps spare mask 0x{:04X}",
                        res.active_control_mask,
                        res.spare_recovery_mask
                    );
                    assert_eq!(
                        res.active_control_mask & res.x100_compute_cap_mask,
                        0,
                        "Active mask 0x{:04X} overlaps compute cap 0x{:04X}",
                        res.active_control_mask,
                        res.x100_compute_cap_mask
                    );

                    // 3. Lendable spare invariant
                    assert_eq!(
                        res.spare_recovery_mask & res.x100_compute_cap_mask,
                        res.spare_recovery_mask,
                        "Spare core {c} is not lendable to compute"
                    );

                    // 4. Partition invariant
                    assert_eq!(
                        res.active_control_mask | res.x100_compute_cap_mask,
                        online_mask,
                        "Active + compute cap must partition online cores"
                    );

                    // 5. Descending allocation order check:
                    // Active cores must be the top 2 available cores in online_mask & !spare_bit
                    let available = online_mask & !spare_bit;
                    let mut expected_active = Vec::new();
                    for bit in (0..8).rev() {
                        if (available & (1 << bit)) != 0 {
                            expected_active.push(bit);
                            if expected_active.len() == 2 {
                                break;
                            }
                        }
                    }
                    expected_active.sort_unstable();
                    assert_eq!(
                        res.active_control_cores, expected_active,
                        "Active cores must be top available cores descending"
                    );

                    // 6. Rotator behavior
                    let next1 = res.next_active_control_core();
                    let next2 = res.next_active_control_core();
                    assert_eq!(next1, expected_active[0]);
                    assert_eq!(next2, expected_active[1]);
                }
                Err(err) => {
                    total_rejected += 1;
                    assert!(
                        online_count < 3,
                        "Unexpected failure when online_count = {online_count} >= 3: {err}"
                    );
                    assert_eq!(err.diagnostic_code(), "affinity_error");
                }
            }
        }
    }

    assert!(total_tested > 0);
    assert!(total_success > 0);
    assert!(total_rejected > 0);
    println!("Exhaustive default config with custom spare tested {total_tested} combinations ({total_success} success, {total_rejected} rejected).");
}

/// Invariant Suite 2: Custom spare mask across ALL 256 online topologies with minimal active cores (1).
///
/// For every possible online mask in `0..=255`, and for every core `c` that is online in `online_mask`:
/// - We configure `ControlPlaneCapacityConfig::minimal().with_custom_spare_mask(1 << c)`.
/// - If `online_count < 2`: Must fail with `K3Error::AffinityError`.
/// - If `online_count >= 2`: Must succeed and strictly verify:
///   1. `res.active_control_cores.len() == 1`
///   2. `res.active_control_mask.count_ones() == 1`
///   3. `res.spare_recovery_cores == vec![c]`
///   4. `res.active_control_mask & res.spare_recovery_mask == 0`
///   5. `res.active_control_mask & res.x100_compute_cap_mask == 0`
///   6. `res.spare_recovery_mask & res.x100_compute_cap_mask == res.spare_recovery_mask`
///   7. `res.next_active_control_core()` always yields the single active core.
#[test]
fn test_exhaustive_custom_spare_without_custom_active_minimal_1core() {
    let mut total_tested = 0;
    let mut total_success = 0;
    let mut total_rejected = 0;

    for online_mask in 0u64..=255u64 {
        let online_count = online_mask.count_ones() as usize;

        for c in 0..8 {
            if (online_mask & (1 << c)) == 0 {
                continue;
            }
            total_tested += 1;
            let spare_bit = 1u64 << c;
            let cfg = ControlPlaneCapacityConfig::minimal().with_custom_spare_mask(spare_bit);

            match cfg.resolve_with_online_mask(online_mask) {
                Ok(res) => {
                    total_success += 1;
                    assert!(
                        online_count >= 2,
                        "Expected failure when online_count < 2, but succeeded for mask 0x{online_mask:02X}, spare core {c}"
                    );

                    assert_eq!(res.active_control_cores.len(), 1);
                    assert_eq!(res.active_control_mask.count_ones(), 1);
                    assert_eq!(res.spare_recovery_cores, vec![c]);
                    assert_eq!(res.spare_recovery_core(), Some(c));

                    assert_eq!(res.active_control_mask & res.spare_recovery_mask, 0);
                    assert_eq!(res.active_control_mask & res.x100_compute_cap_mask, 0);
                    assert_eq!(
                        res.spare_recovery_mask & res.x100_compute_cap_mask,
                        res.spare_recovery_mask
                    );
                    assert_eq!(
                        res.active_control_mask | res.x100_compute_cap_mask,
                        online_mask
                    );

                    let sole_active = res.active_control_cores[0];
                    assert_eq!(res.next_active_control_core(), sole_active);
                    assert_eq!(res.next_active_control_core(), sole_active);
                }
                Err(err) => {
                    total_rejected += 1;
                    assert!(
                        online_count < 2,
                        "Unexpected failure when online_count = {online_count} >= 2: {err}"
                    );
                    assert_eq!(err.diagnostic_code(), "affinity_error");
                }
            }
        }
    }

    assert!(total_tested > 0);
    assert!(total_success > 0);
    assert!(total_rejected > 0);
    println!("Exhaustive minimal config with custom spare tested {total_tested} combinations ({total_success} success, {total_rejected} rejected).");
}

/// Invariant Suite 3: Multi-core custom spare mask without custom active mask across topologies.
///
/// Tests configurations requesting multiple spare cores (e.g. 2 spare cores `0x0030`, cores 4 and 5).
#[test]
fn test_multi_core_custom_spare_without_custom_active() {
    let spare_mask = 0x0030; // cores 4 and 5
    let cfg = ControlPlaneCapacityConfig::default().with_custom_spare_mask(spare_mask);

    // 1. Baseline on 0x00FF: 8 cores online.
    // Requests 2 active cores + 2 spare cores.
    let res = cfg.resolve_default().expect("resolve default on 0x00FF");
    assert_eq!(res.spare_recovery_cores, vec![4, 5]);
    assert_eq!(res.spare_recovery_mask, 0x0030);
    // Active cores auto-allocated from remaining: top 2 are cores 6 and 7 (0x00C0)
    assert_eq!(res.active_control_cores, vec![6, 7]);
    assert_eq!(res.active_control_mask, 0x00C0);
    // Compute cap: valid_online & !0x00C0 = 0x003F
    assert_eq!(res.x100_compute_cap_mask, 0x003F);
    assert_eq!(
        res.spare_recovery_mask & res.x100_compute_cap_mask,
        res.spare_recovery_mask
    );

    // 2. Degraded topology where cores 6 and 7 are OFFLINE, online = 0x003F (cores 0..5)
    // Cores 4 and 5 are spare, remaining available for active = 0..3 (count = 4).
    // Active cores requested = 2. Auto-allocation should select top 2: cores 2 and 3!
    let res_degraded = cfg
        .resolve_with_online_mask(0x003F)
        .expect("resolve on 0x003F");
    assert_eq!(res_degraded.spare_recovery_cores, vec![4, 5]);
    assert_eq!(res_degraded.active_control_cores, vec![2, 3]);
    assert_eq!(res_degraded.active_control_mask, 0x000C);
    assert_eq!(res_degraded.active_control_mask & res_degraded.spare_recovery_mask, 0);

    // 3. Degraded topology where one of the spare cores is OFFLINE:
    // E.g., online = 0x00EF (core 4 is offline, core 5 is online)
    let err = cfg.resolve_with_online_mask(0x00EF).unwrap_err();
    assert_eq!(err.diagnostic_code(), "affinity_error");
    assert!(err.to_string().contains("specifies offline cores"));

    // 4. Insufficient cores for active: online = 0x0031 (cores 0, 4, 5)
    // spare takes cores 4, 5. Only core 0 remains for active. But cfg requests 2 active cores.
    let err_insuff = cfg.resolve_with_online_mask(0x0031).unwrap_err();
    assert_eq!(err_insuff.diagnostic_code(), "affinity_error");
    assert!(err_insuff.to_string().contains("exceeds 1 available online X100 cores"));
}

/// Invariant Suite 4: Adversarial rejection of invalid custom spare masks.
#[test]
fn test_adversarial_custom_spare_rejections() {
    // 1. Spare mask entirely on A100 (Cluster 1, 0xFF00)
    let cfg_a100 = ControlPlaneCapacityConfig::default().with_custom_spare_mask(0x0100);
    let err_a100 = cfg_a100.resolve_default().unwrap_err();
    assert_eq!(err_a100.diagnostic_code(), "affinity_error");
    assert!(err_a100.to_string().contains("encroaches on Cluster 1"));

    // 2. Spare mask outside 16-core SoC
    let cfg_out = ControlPlaneCapacityConfig::default().with_custom_spare_mask(1u64 << 20);
    let err_out = cfg_out.resolve_default().unwrap_err();
    assert_eq!(err_out.diagnostic_code(), "affinity_error");
    assert!(err_out.to_string().contains("encroaches on Cluster 1") || err_out.to_string().contains("outside X100"));

    // 3. Empty online mask with custom spare mask
    let cfg_empty = ControlPlaneCapacityConfig::default().with_custom_spare_mask(0x0020);
    let err_empty = cfg_empty.resolve_with_online_mask(0x0000).unwrap_err();
    assert_eq!(err_empty.diagnostic_code(), "affinity_error");
    assert!(err_empty.to_string().contains("no online X100"));
}

/// Invariant Suite 5: Concurrent stress test on capacity resolved with custom spare mask.
#[test]
fn test_concurrency_stress_custom_spare_resolved_rotator() {
    let cfg = ControlPlaneCapacityConfig::default().with_custom_spare_mask(0x0010); // core 4 is spare
    let res = Arc::new(cfg.resolve_default().expect("resolve"));

    // Active cores must be 6 and 7
    assert_eq!(res.active_control_cores, vec![6, 7]);
    assert_eq!(res.spare_recovery_cores, vec![4]);

    let num_threads = 32;
    let ops_per_thread = 10_000;
    let handles: Vec<_> = (0..num_threads)
        .map(|_| {
            let res_clone = Arc::clone(&res);
            thread::spawn(move || {
                let mut c6 = 0;
                let mut c7 = 0;
                for _ in 0..ops_per_thread {
                    let core = res_clone.next_active_control_core();
                    if core == 6 {
                        c6 += 1;
                    } else if core == 7 {
                        c7 += 1;
                    } else {
                        panic!("Unexpected core: {core}");
                    }
                }
                (c6, c7)
            })
        })
        .collect();

    let mut total_c6 = 0;
    let mut total_c7 = 0;
    for h in handles {
        let (c6, c7) = h.join().unwrap();
        total_c6 += c6;
        total_c7 += c7;
    }

    assert_eq!(total_c6 + total_c7, num_threads * ops_per_thread);
    assert_eq!(total_c6, total_c7);
}
