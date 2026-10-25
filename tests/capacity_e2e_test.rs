//! Opaque-Box E2E Integration Test Suite for US-06.1.1: ControlPlaneCapacityConfig & X100 Core Resolver.
//!
//! Validates:
//! - Tier 1: Feature Coverage (Features 1–15, >=5 assertions/cases per feature)
//! - Tier 2: Boundary & Corner Cases (empty mask, 1-core online, non-contiguous masks, atomic overflow, error format)
//! - Tier 3: Cross-Feature Combinations (lendable core 5 x compute, atomic rotator concurrency, custom mask fallback)
//! - Tier 4: Real-World Scenarios (Full SoC 16-core, runaway preemption, degraded mode, high-throughput tokenization, survival mode)
//!
//! Specifications: ADR-007, ADR-008, ADR-041, `features/core-affinity.md`, Issue #69.

#![allow(
    clippy::pedantic,
    clippy::cast_possible_truncation,
    clippy::undocumented_unsafe_blocks,
    clippy::too_many_lines,
    clippy::doc_markdown
)]

use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::thread;

use k3_core::ipc::protocol::StatusCode;

use k3_core::affinity::{
    A100_CLUSTER_MASK, ControlPlaneCapacityConfig, DEFAULT_A100_MAX_COMPUTE_CORES,
    DEFAULT_ACTIVE_CONTROL_MASK, DEFAULT_SPARE_RECOVERY_MASK, DEFAULT_X100_COMPUTE_CAP_MASK,
    DEFAULT_X100_MAX_COMPUTE_CORES, X100_CLUSTER_MASK,
};

// ============================================================================
// Tier 1: Feature Coverage Suite (Features 1–15, >=5 assertions per feature)
// ============================================================================

#[test]
fn test_tier1_f01_hardware_mask_topology_constants() {
    // 1. X100 cluster mask covers cores 0..7 (bits 0..7)
    assert_eq!(X100_CLUSTER_MASK, 0x00FF);
    // 2. A100 cluster mask covers cores 8..15 (bits 8..15)
    assert_eq!(A100_CLUSTER_MASK, 0xFF00);
    // 3. Clusters are strictly mutually disjoint
    assert_eq!(X100_CLUSTER_MASK & A100_CLUSTER_MASK, 0);
    // 4. Cluster union covers all 16 cores of SpacemiT K3 SoC
    assert_eq!(X100_CLUSTER_MASK | A100_CLUSTER_MASK, 0xFFFF);
    // 5. Each cluster has exactly 8 physical cores
    assert_eq!(X100_CLUSTER_MASK.count_ones(), 8);
    assert_eq!(A100_CLUSTER_MASK.count_ones(), 8);
}

#[test]
fn test_tier1_f02_default_capacity_constants() {
    // 1. Default active control mask is cores 6 and 7 (0x00C0)
    assert_eq!(DEFAULT_ACTIVE_CONTROL_MASK, 0x00C0);
    assert_eq!(DEFAULT_ACTIVE_CONTROL_MASK.count_ones(), 2);
    // 2. Default spare recovery mask is core 5 (0x0020)
    assert_eq!(DEFAULT_SPARE_RECOVERY_MASK, 0x0020);
    assert_eq!(DEFAULT_SPARE_RECOVERY_MASK.count_ones(), 1);
    // 3. Default X100 compute cap mask is cores 0..5 (0x003F)
    assert_eq!(DEFAULT_X100_COMPUTE_CAP_MASK, 0x003F);
    assert_eq!(DEFAULT_X100_COMPUTE_CAP_MASK.count_ones(), 6);
    // 4. Max compute cores on X100 is 6
    assert_eq!(DEFAULT_X100_MAX_COMPUTE_CORES, 6);
    // 5. Max compute cores on A100 is 8
    assert_eq!(DEFAULT_A100_MAX_COMPUTE_CORES, 8);
}

#[test]
fn test_tier1_f03_control_plane_capacity_config_builder() {
    // 1. Default configuration initializes expected active and spare core counts
    let cfg = ControlPlaneCapacityConfig::default();
    assert_eq!(cfg.active_control_cores, 2);
    assert_eq!(cfg.spare_recovery_cores, 1);
    assert_eq!(cfg.custom_active_control_mask, None);
    assert_eq!(cfg.custom_spare_recovery_mask, None);
    assert_eq!(cfg.custom_x100_compute_cap_mask, None);

    // 2. Builder method with_active_cores modifies active count
    let cfg2 = cfg.clone().with_active_cores(3);
    assert_eq!(cfg2.active_control_cores, 3);

    // 3. Builder method with_spare_cores modifies spare count
    let cfg3 = cfg2.with_spare_cores(2);
    assert_eq!(cfg3.spare_recovery_cores, 2);

    // 4. Builder method with_custom_active_mask sets custom active mask
    let cfg4 = cfg3.with_custom_active_mask(0x0080);
    assert_eq!(cfg4.custom_active_control_mask, Some(0x0080));

    // 5. Builder methods with_custom_spare_mask and with_custom_compute_cap_mask
    let cfg5 = cfg4
        .with_custom_spare_mask(0x0040)
        .with_custom_compute_cap_mask(0x003F);
    assert_eq!(cfg5.custom_spare_recovery_mask, Some(0x0040));
    assert_eq!(cfg5.custom_x100_compute_cap_mask, Some(0x003F));
}

#[test]
fn test_tier1_f04_minimal_1core_config_constructor() {
    // 1. Minimal constructor requests 1 active core
    let min = ControlPlaneCapacityConfig::minimal();
    assert_eq!(min.active_control_cores, 1);
    // 2. Minimal constructor requests 0 spare recovery cores
    assert_eq!(min.spare_recovery_cores, 0);
    // 3. Custom masks remain None
    assert_eq!(min.custom_active_control_mask, None);
    // 4. Resolves on 8-core X100 to single highest core 7 (0x0080)
    let res = min.resolve_default().expect("resolve minimal");
    assert_eq!(res.active_control_mask, 0x0080);
    assert_eq!(res.active_control_cores, vec![7]);
    // 5. Spare mask is 0 (None) and compute cap expands to 7 cores (0x007F)
    assert_eq!(res.spare_recovery_mask, 0x0000);
    assert_eq!(res.spare_recovery_core(), None);
    assert_eq!(res.x100_compute_cap_mask, 0x007F);
    assert_eq!(res.x100_max_compute_cores, 7);
}

#[test]
fn test_tier1_f05_default_8core_capacity_resolution() {
    let cfg = ControlPlaneCapacityConfig::default();
    let res = cfg.resolve_default().expect("resolve default 8-core");

    // 1. Active control mask matches 0x00C0 (cores 6 and 7)
    assert_eq!(res.active_control_mask, DEFAULT_ACTIVE_CONTROL_MASK);
    assert_eq!(res.active_control_cores, vec![6, 7]);
    // 2. Spare recovery mask matches 0x0020 (core 5)
    assert_eq!(res.spare_recovery_mask, DEFAULT_SPARE_RECOVERY_MASK);
    assert_eq!(res.spare_recovery_cores, vec![5]);
    // 3. Compute cap mask matches 0x003F (cores 0..5)
    assert_eq!(res.x100_compute_cap_mask, DEFAULT_X100_COMPUTE_CAP_MASK);
    assert_eq!(res.x100_max_compute_cores, 6);
    // 4. A100 compute capacity is preserved at 8 cores
    assert_eq!(res.a100_max_compute_cores, 8);
    // 5. Active control cores are recognized by predicate
    assert!(res.is_active_control_core(6));
    assert!(res.is_active_control_core(7));
    assert!(!res.is_active_control_core(5));
}

#[test]
fn test_tier1_f06_constrained_topology_resolution() {
    let cfg = ControlPlaneCapacityConfig::default();

    // 1. 4-core online (0x000F): active cores 2, 3 (0x000C), spare core 1 (0x0002), compute cap cores 0, 1 (0x0003)
    let res4 = cfg.resolve_with_online_mask(0x000F).expect("resolve 4-core");
    assert_eq!(res4.active_control_mask, 0x000C);
    assert_eq!(res4.active_control_cores, vec![2, 3]);
    assert_eq!(res4.spare_recovery_mask, 0x0002);
    assert_eq!(res4.x100_compute_cap_mask, 0x0003);
    assert_eq!(res4.x100_max_compute_cores, 2);

    // 2. 4-core minimal config: active core 3 (0x0008), 0 spare, compute cap cores 0..2 (0x0007, max 3)
    let min = ControlPlaneCapacityConfig::minimal();
    let res4_min = min.resolve_with_online_mask(0x000F).expect("resolve 4-core minimal");
    assert_eq!(res4_min.active_control_mask, 0x0008);
    assert_eq!(res4_min.spare_recovery_mask, 0x0000);
    assert_eq!(res4_min.x100_compute_cap_mask, 0x0007);
    assert_eq!(res4_min.x100_max_compute_cores, 3);

    // 3. 2-core minimal config (0x0003): active core 1 (0x0002), 0 spare, compute cap core 0 (0x0001)
    let res2 = min.resolve_with_online_mask(0x0003).expect("resolve 2-core");
    assert_eq!(res2.active_control_mask, 0x0002);
    assert_eq!(res2.spare_recovery_mask, 0x0000);
    assert_eq!(res2.x100_compute_cap_mask, 0x0001);
    assert_eq!(res2.x100_max_compute_cores, 1);

    // 4. 1-core minimal config (0x0001): active core 0 (0x0001), 0 spare, compute cap 0x0000
    let res1 = min.resolve_with_online_mask(0x0001).expect("resolve 1-core");
    assert_eq!(res1.active_control_mask, 0x0001);
    assert_eq!(res1.spare_recovery_mask, 0x0000);
    assert_eq!(res1.x100_compute_cap_mask, 0x0000);
    assert_eq!(res1.x100_max_compute_cores, 0);

    // 5. Requesting default (3 cores) on 2-core online returns error
    let err2 = cfg.resolve_with_online_mask(0x0003);
    assert!(err2.is_err());
}

#[test]
fn test_tier1_f07_lendable_spare_capacity_semantics() {
    let cfg = ControlPlaneCapacityConfig::default();
    let res = cfg.resolve_default().expect("resolve default");

    // 1. Core 5 is designated as the spare recovery core
    assert_eq!(res.spare_recovery_core(), Some(5));
    assert!(res.is_spare_recovery_core(5));

    // 2. Core 5 is simultaneously in the X100 compute cap mask (lendable)
    assert!(res.is_x100_compute_allowed(5));
    assert_eq!(res.spare_recovery_mask & res.x100_compute_cap_mask, res.spare_recovery_mask);

    // 3. Active control cores 6 and 7 are strictly NOT allowed for compute
    assert!(!res.is_x100_compute_allowed(6));
    assert!(!res.is_x100_compute_allowed(7));

    // 4. Active control mask and compute cap mask are strictly disjoint
    assert_eq!(res.active_control_mask & res.x100_compute_cap_mask, 0);

    // 5. Compute reservation within cap (0x003F) validates successfully
    assert!(res.validate_compute_mask(0x003F).is_ok());
    assert!(res.validate_compute_mask(0x0020).is_ok());
    // Encroaching on active core 6 or 7 is rejected
    assert!(res.validate_compute_mask(0x0040).is_err());
}

#[test]
fn test_tier1_f08_a100_compute_capacity_invariant() {
    let cfg = ControlPlaneCapacityConfig::default();
    let res = cfg.resolve_default().expect("resolve default");

    // 1. A100 maximum compute cores equals 8
    assert_eq!(res.a100_max_compute_cores, 8);
    // 2. A100 cluster mask is disjoint from X100 active control mask
    assert_eq!(A100_CLUSTER_MASK & res.active_control_mask, 0);
    // 3. A100 cluster mask is disjoint from X100 spare recovery mask
    assert_eq!(A100_CLUSTER_MASK & res.spare_recovery_mask, 0);
    // 4. A100 cluster mask is disjoint from X100 compute cap mask
    assert_eq!(A100_CLUSTER_MASK & res.x100_compute_cap_mask, 0);
    // 5. Changing X100 capacity settings does not alter A100 compute capacity
    let min_res = ControlPlaneCapacityConfig::minimal().resolve_default().unwrap();
    assert_eq!(min_res.a100_max_compute_cores, 8);
}

#[test]
fn test_tier1_f09_deterministic_active_control_round_robin() {
    let cfg = ControlPlaneCapacityConfig::default();
    let res = cfg.resolve_default().expect("resolve default");

    // 1. round_robin_active_core rotates deterministically across cores [6, 7]
    assert_eq!(res.round_robin_active_core(0), 6);
    assert_eq!(res.round_robin_active_core(1), 7);
    assert_eq!(res.round_robin_active_core(2), 6);
    assert_eq!(res.round_robin_active_core(3), 7);

    // 2. next_active_control_core yields monotonic sequential progression
    assert_eq!(res.next_active_control_core(), 6);
    assert_eq!(res.next_active_control_core(), 7);
    assert_eq!(res.next_active_control_core(), 6);
    assert_eq!(res.next_active_control_core(), 7);

    // 3. Minimal 1-core resolution cycles purely on single core 7
    let min_res = ControlPlaneCapacityConfig::minimal().resolve_default().unwrap();
    assert_eq!(min_res.next_active_control_core(), 7);
    assert_eq!(min_res.next_active_control_core(), 7);
    assert_eq!(min_res.next_active_control_core(), 7);

    // 4. Large index rotation wraps smoothly
    assert_eq!(res.round_robin_active_core(100), 6);
    assert_eq!(res.round_robin_active_core(101), 7);

    // 5. Total active cores list length is 2
    assert_eq!(res.active_control_cores.len(), 2);
}

#[test]
fn test_tier1_f10_dedicated_spare_recovery_core_assignment() {
    let cfg = ControlPlaneCapacityConfig::default();
    let res = cfg.resolve_default().expect("resolve default");

    // 1. Default spare recovery core is Some(5)
    assert_eq!(res.spare_recovery_core(), Some(5));
    // 2. Core 5 is verified as spare
    assert!(res.is_spare_recovery_core(5));
    // 3. Dedicated active cores are NOT spare cores
    assert!(!res.is_spare_recovery_core(6));
    assert!(!res.is_spare_recovery_core(7));
    // 4. Minimal configuration returns None for spare recovery core
    let min_res = ControlPlaneCapacityConfig::minimal().resolve_default().unwrap();
    assert_eq!(min_res.spare_recovery_core(), None);
    // 5. Config with custom spare mask 0x0010 returns Some(4)
    let custom_cfg = ControlPlaneCapacityConfig::default()
        .with_custom_active_mask(0x00C0)
        .with_custom_spare_mask(0x0010);
    let custom_res = custom_cfg.resolve_default().expect("custom spare");
    assert_eq!(custom_res.spare_recovery_core(), Some(4));
}

#[test]
fn test_tier1_f11_validation_zero_control_cores_rejection() {
    // 1. Setting active_control_cores = 0 returns error
    let cfg = ControlPlaneCapacityConfig::default().with_active_cores(0);
    let res = cfg.resolve_default();
    assert!(res.is_err());

    // 2. Error is K3Error::AffinityError
    let err = res.unwrap_err();
    assert_eq!(err.diagnostic_code(), "affinity_error");

    // 3. Error message specifies at least 1 active control core required
    let msg = err.to_string();
    assert!(msg.contains("at least 1 active control core"));

    // 4. Custom active mask set to 0 returns error
    let cfg_zero_mask = ControlPlaneCapacityConfig::default().with_custom_active_mask(0);
    let res_zero = cfg_zero_mask.resolve_default();
    assert!(res_zero.is_err());

    // 5. Error message indicates active mask cannot be zero
    let msg_zero = res_zero.unwrap_err().to_string();
    assert!(msg_zero.contains("active control mask cannot be zero"));
}

#[test]
fn test_tier1_f12_validation_outside_x100_rejection() {
    // 1. Custom active mask with bit 8 (0x0100 on A100) is rejected
    let cfg_a100_act = ControlPlaneCapacityConfig::default().with_custom_active_mask(0x0100);
    let res1 = cfg_a100_act.resolve_default();
    assert!(res1.is_err());
    assert!(res1.unwrap_err().to_string().contains("encroaches on Cluster 1 (A100"));

    // 2. Custom spare mask with bit 9 (0x0200 on A100) is rejected
    let cfg_a100_spr = ControlPlaneCapacityConfig::default().with_custom_spare_mask(0x0200);
    let res2 = cfg_a100_spr.resolve_default();
    assert!(res2.is_err());
    assert!(res2.unwrap_err().to_string().contains("encroaches on Cluster 1 (A100"));

    // 3. Full A100 mask (0xFF00) as active control mask is rejected
    let cfg_full_a100 = ControlPlaneCapacityConfig::default().with_custom_active_mask(A100_CLUSTER_MASK);
    assert!(cfg_full_a100.resolve_default().is_err());

    // 4. Bit 16 (0x10000 outside 16-core SoC) is rejected
    let cfg_oob = ControlPlaneCapacityConfig::default().with_custom_active_mask(0x10000);
    assert!(cfg_oob.resolve_default().is_err());

    // 5. Valid mask strictly inside 0x00FF (e.g. 0x00C0) is accepted
    let cfg_valid = ControlPlaneCapacityConfig::default().with_custom_active_mask(0x00C0);
    assert!(cfg_valid.resolve_default().is_ok());
}

#[test]
fn test_tier1_f13_validation_active_vs_spare_overlap_rejection() {
    // 1. Overlapping active mask 0x00C0 and spare mask 0x0040 (overlap on core 6)
    let cfg = ControlPlaneCapacityConfig::default()
        .with_custom_active_mask(0x00C0)
        .with_custom_spare_mask(0x0040);
    let res = cfg.resolve_default();
    assert!(res.is_err());

    // 2. Error message indicates overlap
    let err = res.unwrap_err();
    assert!(err.to_string().contains("overlaps spare recovery mask"));

    // 3. Complete overlap (active 0x0020, spare 0x0020) is rejected
    let cfg2 = ControlPlaneCapacityConfig::default()
        .with_custom_active_mask(0x0020)
        .with_custom_spare_mask(0x0020);
    assert!(cfg2.resolve_default().is_err());

    // 4. Disjoint active (0x00C0) and spare (0x0020) is accepted
    let cfg3 = ControlPlaneCapacityConfig::default()
        .with_custom_active_mask(0x00C0)
        .with_custom_spare_mask(0x0020);
    assert!(cfg3.resolve_default().is_ok());

    // 5. Disjoint single-core active (0x0080) and single-core spare (0x0010) is accepted
    let cfg4 = ControlPlaneCapacityConfig::default()
        .with_custom_active_mask(0x0080)
        .with_custom_spare_mask(0x0010);
    assert!(cfg4.resolve_default().is_ok());
}

#[test]
fn test_tier1_f14_validation_compute_cap_vs_active_overlap_rejection() {
    // 1. Custom compute cap mask overlapping dedicated active mask (0x007F overlaps 0x00C0 on core 6)
    let cfg = ControlPlaneCapacityConfig::default()
        .with_custom_active_mask(0x00C0)
        .with_custom_compute_cap_mask(0x007F);
    let res = cfg.resolve_default();
    assert!(res.is_err());

    // 2. Error message indicates compute cap overlaps active control mask
    let err = res.unwrap_err();
    assert!(err.to_string().contains("compute cap mask"));
    assert!(err.to_string().contains("overlaps dedicated active control mask"));

    // 3. Compute cap covering entire X100 (0x00FF) is rejected
    let cfg_all = ControlPlaneCapacityConfig::default().with_custom_compute_cap_mask(0x00FF);
    assert!(cfg_all.resolve_default().is_err());

    // 4. Compute cap disjoint from active mask (0x003F disjoint from 0x00C0) is accepted
    let cfg_ok = ControlPlaneCapacityConfig::default()
        .with_custom_active_mask(0x00C0)
        .with_custom_compute_cap_mask(0x003F);
    assert!(cfg_ok.resolve_default().is_ok());

    // 5. Compute cap containing lendable spare core 5 (0x0020) is accepted
    let cfg_lendable = ControlPlaneCapacityConfig::default()
        .with_custom_active_mask(0x00C0)
        .with_custom_spare_mask(0x0020)
        .with_custom_compute_cap_mask(0x003F);
    let resolved = cfg_lendable.resolve_default().expect("valid lendable");
    assert_eq!(resolved.x100_compute_cap_mask & resolved.spare_recovery_mask, 0x0020);
}

#[test]
fn test_tier1_f15_validation_online_core_exhaustion_rejection() {
    // 1. Requesting 3 active + 2 spare (5 cores) on a 4-core mask (0x000F)
    let cfg = ControlPlaneCapacityConfig::default()
        .with_active_cores(3)
        .with_spare_cores(2);
    let res = cfg.resolve_with_online_mask(0x000F);
    assert!(res.is_err());

    // 2. Error message states requested cores exceed online cores
    let err = res.unwrap_err();
    assert!(err.to_string().contains("exceeds 4 online X100 cores"));

    // 3. Requesting 2 active + 1 spare (3 cores) on 2-core online mask (0x0003) is rejected
    let cfg_default = ControlPlaneCapacityConfig::default();
    let res2 = cfg_default.resolve_with_online_mask(0x0003);
    assert!(res2.is_err());
    assert!(res2.unwrap_err().to_string().contains("exceeds 2 online X100 cores"));

    // 4. Requesting exact count (2 active + 2 spare = 4 on 4 online cores) is accepted
    let cfg_exact = ControlPlaneCapacityConfig::default()
        .with_active_cores(2)
        .with_spare_cores(2);
    assert!(cfg_exact.resolve_with_online_mask(0x000F).is_ok());

    // 5. Requesting 1 active on 1 online core is accepted
    let cfg_single = ControlPlaneCapacityConfig::minimal();
    assert!(cfg_single.resolve_with_online_mask(0x0001).is_ok());
}

// ============================================================================
// Tier 2: Boundary, Limit & Corner Cases (8 comprehensive tests)
// ============================================================================

#[test]
fn test_tier2_boundary_empty_online_mask_rejection() {
    let cfg = ControlPlaneCapacityConfig::default();
    // 1. Online mask = 0 (0 cores online)
    let res = cfg.resolve_with_online_mask(0x0000);
    assert!(res.is_err());
    let err = res.unwrap_err();
    assert_eq!(err.diagnostic_code(), "affinity_error");
    assert!(err.to_string().contains("no online X100 general-purpose cores available"));
}

#[test]
fn test_tier2_boundary_a100_only_online_rejection() {
    let cfg = ControlPlaneCapacityConfig::default();
    // 2. Online mask = 0xFF00 (only A100 cores online, 0 cores online in X100)
    let res = cfg.resolve_with_online_mask(A100_CLUSTER_MASK);
    assert!(res.is_err());
    assert!(res.unwrap_err().to_string().contains("no online X100 general-purpose cores available"));
}

#[test]
fn test_tier2_boundary_single_online_core_with_default_config_rejection() {
    let cfg = ControlPlaneCapacityConfig::default(); // requests 2 active + 1 spare = 3
    // 3. Online mask = 0x0001 (1 core online)
    let res = cfg.resolve_with_online_mask(0x0001);
    assert!(res.is_err());
    assert!(res.unwrap_err().to_string().contains("exceeds 1 online X100 cores"));
}

#[test]
fn test_tier2_boundary_non_contiguous_online_masks() {
    // 4. Alternating even cores online: 0x0055 = cores 0, 2, 4, 6 online (4 cores)
    let cfg = ControlPlaneCapacityConfig::default(); // 2 active, 1 spare
    let res = cfg.resolve_with_online_mask(0x0055).expect("resolve non-contiguous even");
    // Highest online cores for active control: 6, 4 -> mask (1<<6) | (1<<4) = 0x0040 | 0x0010 = 0x0050
    assert_eq!(res.active_control_mask, 0x0050);
    assert_eq!(res.active_control_cores, vec![4, 6]);
    // Next online core for spare recovery: 2 -> mask (1<<2) = 0x0004
    assert_eq!(res.spare_recovery_mask, 0x0004);
    assert_eq!(res.spare_recovery_core(), Some(2));
    // Remaining online cores for compute cap: cores 0, 2 -> mask 0x0005
    assert_eq!(res.x100_compute_cap_mask, 0x0005);
    assert_eq!(res.x100_max_compute_cores, 2);

    // 5. Alternating odd cores online: 0x00AA = cores 1, 3, 5, 7 online (4 cores)
    let res_odd = cfg.resolve_with_online_mask(0x00AA).expect("resolve non-contiguous odd");
    // Highest online cores: 7, 5 -> mask 0x0080 | 0x0020 = 0x00A0
    assert_eq!(res_odd.active_control_mask, 0x00A0);
    assert_eq!(res_odd.active_control_cores, vec![5, 7]);
    // Next online core: 3 -> mask 0x0008
    assert_eq!(res_odd.spare_recovery_mask, 0x0008);
    assert_eq!(res_odd.spare_recovery_core(), Some(3));
    // Compute cap: cores 1, 3 -> mask 0x000A
    assert_eq!(res_odd.x100_compute_cap_mask, 0x000A);
}

#[test]
fn test_tier2_boundary_atomic_round_robin_counter_overflow() {
    let cfg = ControlPlaneCapacityConfig::default();
    let res = cfg.resolve_default().expect("resolve default");

    // Pre-set round-robin counter near usize::MAX
    res.round_robin_counter.store(usize::MAX - 1, Ordering::SeqCst);

    // Call 4 times, ensuring safe wrapping over usize::MAX without panic or arithmetic overflow
    let c1 = res.next_active_control_core(); // usize::MAX - 1
    let c2 = res.next_active_control_core(); // usize::MAX
    let c3 = res.next_active_control_core(); // 0
    let c4 = res.next_active_control_core(); // 1

    assert!(c1 == 6 || c1 == 7);
    assert!(c2 == 6 || c2 == 7);
    assert_ne!(c1, c2);
    assert_eq!(c3, 6);
    assert_eq!(c4, 7);
}

#[test]
fn test_tier2_boundary_custom_mask_limits_and_extremes() {
    // Single highest core 7 as active, single lowest core 0 as spare, cores 1..6 as compute
    let cfg = ControlPlaneCapacityConfig::default()
        .with_custom_active_mask(0x0080)
        .with_custom_spare_mask(0x0001)
        .with_custom_compute_cap_mask(0x007E);
    let res = cfg.resolve_default().expect("resolve extreme masks");

    assert_eq!(res.active_control_mask, 0x0080);
    assert_eq!(res.active_control_cores, vec![7]);
    assert_eq!(res.spare_recovery_mask, 0x0001);
    assert_eq!(res.spare_recovery_core(), Some(0));
    assert_eq!(res.x100_compute_cap_mask, 0x007E);
    assert_eq!(res.x100_max_compute_cores, 6);
}

#[test]
fn test_tier2_boundary_diagnostic_error_formatting_and_status_code() {
    let cfg = ControlPlaneCapacityConfig::default().with_active_cores(0);
    let err = cfg.resolve_default().unwrap_err();

    // 1. Diagnostic code is "affinity_error"
    assert_eq!(err.diagnostic_code(), "affinity_error");

    // 2. Exit code / status code is 13 (StatusCode::AffinityError)
    assert_eq!(StatusCode::AffinityError as i32, 13);

    // 3. Display string matches canonical format
    let display = format!("{err}");
    assert!(display.starts_with("[affinity_error] invalid affinity configuration"));
}

#[test]
fn test_tier2_boundary_offline_core_in_custom_mask_rejection() {
    // Custom active mask specifies core 7 (0x0080), but core 7 is offline in online_mask (0x007F)
    let cfg = ControlPlaneCapacityConfig::default().with_custom_active_mask(0x0080);
    let res = cfg.resolve_with_online_mask(0x007F);
    assert!(res.is_err());
    let err_msg = res.unwrap_err().to_string();
    assert!(err_msg.contains("specifies offline cores"));
    assert!(err_msg.contains("0x0080"));

    // Custom spare mask specifies core 5 (0x0020), but core 5 is offline (0x00DF)
    let cfg_spr = ControlPlaneCapacityConfig::default()
        .with_custom_active_mask(0x00C0)
        .with_custom_spare_mask(0x0020);
    let res_spr = cfg_spr.resolve_with_online_mask(0x00DF);
    assert!(res_spr.is_err());
    assert!(res_spr.unwrap_err().to_string().contains("specifies offline cores"));
}

// ============================================================================
// Tier 3: Cross-Feature Combinations (Pairwise Coverage)
// ============================================================================

#[test]
fn test_tier3_pairwise_lendable_spare_x_compute_placement() {
    let cfg = ControlPlaneCapacityConfig::default();
    let res = cfg.resolve_default().expect("resolve default");

    // Pair 1: Core 5 is simultaneously permitted for compute AND spare recovery
    let spare_core = res.spare_recovery_core().expect("spare core");
    assert_eq!(spare_core, 5);
    assert!(res.is_x100_compute_allowed(spare_core));
    assert!(res.is_spare_recovery_core(spare_core));

    // Dedicated active control cores are strictly rejected for compute
    for &act in &res.active_control_cores {
        assert!(!res.is_x100_compute_allowed(act));
    }

    // Full compute cap (0x003F) validates cleanly
    assert!(res.validate_compute_mask(0x003F).is_ok());

    // Single compute worker pinned to lendable core 5 (0x0020) validates cleanly
    assert!(res.validate_compute_mask(0x0020).is_ok());

    // Compute worker attempting to pin to core 6 is rejected
    assert!(res.validate_compute_mask(0x0040).is_err());
}

#[test]
fn test_tier3_pairwise_atomic_rotator_x_multithread_concurrency() {
    let cfg = ControlPlaneCapacityConfig::default();
    let res = Arc::new(cfg.resolve_default().expect("resolve default"));

    // Pair 2: 16 concurrent threads each dispatching 1,000 requests via next_active_control_core
    let mut handles = Vec::new();
    for _ in 0..16 {
        let res_clone = Arc::clone(&res);
        handles.push(thread::spawn(move || {
            let mut count_6 = 0_usize;
            let mut count_7 = 0_usize;
            for _ in 0..1000 {
                let core = res_clone.next_active_control_core();
                if core == 6 {
                    count_6 += 1;
                } else if core == 7 {
                    count_7 += 1;
                } else {
                    panic!("unexpected core: {core}");
                }
            }
            (count_6, count_7)
        }));
    }

    let mut total_6 = 0;
    let mut total_7 = 0;
    for h in handles {
        let (c6, c7) = h.join().expect("thread join");
        total_6 += c6;
        total_7 += c7;
    }

    // Total operations = 16,000
    assert_eq!(total_6 + total_7, 16_000);
    // Exact balanced distribution across active cores 6 and 7
    assert_eq!(total_6, 8_000);
    assert_eq!(total_7, 8_000);
}

#[test]
fn test_tier3_pairwise_custom_mask_overrides_x_offline_topology() {
    // Pair 3: Custom masks interacting with partially offline cores
    let cfg = ControlPlaneCapacityConfig::default()
        .with_custom_active_mask(0x00C0)
        .with_custom_spare_mask(0x0020);

    // Online mask has core 6 offline (0x00BF): custom active mask (0x00C0) must fail
    assert!(cfg.resolve_with_online_mask(0x00BF).is_err());

    // Online mask has core 5 offline (0x00DF): custom spare mask (0x0020) must fail
    assert!(cfg.resolve_with_online_mask(0x00DF).is_err());

    // Online mask has cores 0..3 offline (0x00F0), but cores 5, 6, 7 are online: succeeds!
    let res = cfg.resolve_with_online_mask(0x00F0).expect("online 5,6,7");
    assert_eq!(res.active_control_mask, 0x00C0);
    assert_eq!(res.spare_recovery_mask, 0x0020);
    assert_eq!(res.x100_compute_cap_mask, 0x0030); // cores 4, 5
}

#[test]
fn test_tier3_pairwise_minimal_1core_x_multi_model_isolation() {
    // Pair 4: Minimal 1-core configuration on standard 8-core X100
    let min_cfg = ControlPlaneCapacityConfig::minimal();
    let res = min_cfg.resolve_default().expect("minimal resolve");

    // Only core 7 reserved for control
    assert_eq!(res.active_control_mask, 0x0080);
    assert_eq!(res.spare_recovery_core(), None);

    // Cores 0..6 (0x007F, 7 cores) available for compute
    assert_eq!(res.x100_compute_cap_mask, 0x007F);
    assert_eq!(res.x100_max_compute_cores, 7);

    // Simulate multi-model placement:
    // Model A: cores 0..1 (mask 0x0003)
    assert!(res.validate_compute_mask(0x0003).is_ok());
    // Model B: cores 2..3 (mask 0x000C)
    assert!(res.validate_compute_mask(0x000C).is_ok());
    // Model C: cores 4..6 (mask 0x0070)
    assert!(res.validate_compute_mask(0x0070).is_ok());

    // Disjoint compute partitions do not overlap active control core 7
    assert_eq!((0x0003 | 0x000C | 0x0070) & res.active_control_mask, 0);
}

#[test]
fn test_tier3_pairwise_cross_cluster_isolation_x100_vs_a100() {
    // Pair 5: X100 control reservation vs A100 compute domain
    let cfg = ControlPlaneCapacityConfig::default();
    let res = cfg.resolve_default().expect("resolve default");

    // Control reservation total mask: active (0x00C0) | spare (0x0020) = 0x00E0
    let total_ctrl_mask = res.active_control_mask | res.spare_recovery_mask;
    assert_eq!(total_ctrl_mask, 0x00E0);

    // Total control mask is strictly disjoint from A100 cluster mask (0xFF00)
    assert_eq!(total_ctrl_mask & A100_CLUSTER_MASK, 0);

    // A100 compute mask is strictly preserved
    assert_eq!(res.a100_max_compute_cores, 8);
}

// ============================================================================
// Tier 4: Real-World Application Scenarios (5 production workloads)
// ============================================================================

#[test]
fn test_tier4_scenario1_full_soc_spacemit_k3_16core_baseline() {
    // Scenario 1: Standard boot on SpacemiT K3 (pey-wail-deu) with 16 physical cores online (0xFFFF)
    let cfg = ControlPlaneCapacityConfig::default();
    let res = cfg.resolve_with_online_mask(0xFFFF).expect("16-core SoC");

    // 1. Cluster 0 (X100) control plane reservation
    assert_eq!(res.active_control_mask, 0x00C0); // cores 6, 7
    assert_eq!(res.spare_recovery_mask, 0x0020); // core 5
    assert_eq!(res.x100_compute_cap_mask, 0x003F); // cores 0..5
    assert_eq!(res.x100_max_compute_cores, 6);

    // 2. Cluster 1 (A100) dedicated compute
    assert_eq!(res.a100_max_compute_cores, 8);

    // 3. Cluster isolation verified
    assert_eq!(res.active_control_mask & A100_CLUSTER_MASK, 0);
    assert_eq!(res.spare_recovery_mask & A100_CLUSTER_MASK, 0);
    assert_eq!(res.x100_compute_cap_mask & A100_CLUSTER_MASK, 0);
}

#[test]
fn test_tier4_scenario2_runaway_compute_preemption_via_watchdog() {
    // Scenario 2: Runaway compute workload intervention per US-06 Scenario 2
    let cfg = ControlPlaneCapacityConfig::default();
    let res = cfg.resolve_default().expect("resolve default");

    // Compute workers saturate cores 0..5 (0x003F)
    assert!(res.validate_compute_mask(0x003F).is_ok());

    // Watchdog / health supervisor is bound to spare recovery core 5
    let watchdog_core = res.spare_recovery_core().expect("watchdog core");
    assert_eq!(watchdog_core, 5);

    // Active control plane tasks run on cores 6 and 7, remaining 100% responsive
    let t1_core = res.next_active_control_core();
    let t2_core = res.next_active_control_core();
    assert!(t1_core == 6 || t1_core == 7);
    assert!(t2_core == 6 || t2_core == 7);
    assert_ne!(t1_core, t2_core);

    // Even if core 5 was running compute, active cores 6 and 7 are never starved
    assert!(!res.is_active_control_core(watchdog_core));
    assert_eq!(res.active_control_mask & (1 << watchdog_core), 0);
}

#[test]
fn test_tier4_scenario3_degraded_board_mode_partial_cores() {
    // Scenario 3: Thermally throttled or battery-degraded mode where cores 4..7 are offline
    let online_mask = 0x000F; // only cores 0..3 online

    // Resolver automatically downscales to 2 active cores (2, 3), 1 spare core (1), compute cap cores 0, 1
    let cfg = ControlPlaneCapacityConfig::default();
    let res = cfg.resolve_with_online_mask(online_mask).expect("degraded 4-core");

    assert_eq!(res.active_control_mask, 0x000C);
    assert_eq!(res.spare_recovery_mask, 0x0002);
    assert_eq!(res.x100_compute_cap_mask, 0x0003);
    assert_eq!(res.x100_max_compute_cores, 2);

    // Round-robin balances across degraded active cores 2 and 3
    assert_eq!(res.next_active_control_core(), 2);
    assert_eq!(res.next_active_control_core(), 3);
    assert_eq!(res.next_active_control_core(), 2);
}

#[test]
fn test_tier4_scenario4_high_throughput_tokenization_load_balancing() {
    // Scenario 4: High-throughput prompt tokenization and request admission distribution
    let cfg = ControlPlaneCapacityConfig::default();
    let res = cfg.resolve_default().expect("resolve default");

    // Simulate 50,000 consecutive client requests admitted into control plane
    let mut core_histogram = [0_usize; 8];
    for _ in 0..50_000 {
        let assigned_core = res.next_active_control_core();
        core_histogram[assigned_core] += 1;
    }

    // Exactly 25,000 on core 6 and 25,000 on core 7
    assert_eq!(core_histogram[6], 25_000);
    assert_eq!(core_histogram[7], 25_000);

    // Zero requests assigned to spare core 5 or compute cores 0..4
    assert_eq!(core_histogram[5], 0);
    assert_eq!(core_histogram[0], 0);
}

#[test]
fn test_tier4_scenario5_single_core_disaster_recovery_mode() {
    // Scenario 5: Emergency single-core recovery boot mode (only core 0 online)
    let min_cfg = ControlPlaneCapacityConfig::minimal();
    let res = min_cfg.resolve_with_online_mask(0x0001).expect("disaster mode");

    // Core 0 reserved for active control plane
    assert_eq!(res.active_control_mask, 0x0001);
    assert_eq!(res.active_control_cores, vec![0]);
    assert_eq!(res.spare_recovery_core(), None);

    // Compute cap set to 0 to prevent starvation of system control plane
    assert_eq!(res.x100_compute_cap_mask, 0x0000);
    assert_eq!(res.x100_max_compute_cores, 0);

    // Compute worker placement request is strictly refused
    assert!(res.validate_compute_mask(0x0001).is_err());
}
