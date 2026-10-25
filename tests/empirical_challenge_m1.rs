//! Empirical Challenger Test Suite for Milestone M1: `k3-core` Capacity & Resolver.
//!
//! Stress-tests:
//! 1. Multi-threaded concurrency and round-robin fairness on `next_active_control_core()` and `ActiveControlRotator`.
//! 2. Exhaustive invariant verification across all online masks:
//!    - `(active_control_mask & spare_recovery_mask) == 0`
//!    - `(active_control_mask & x100_compute_cap_mask) == 0`
//!    - `active_control_mask != 0`
//! 3. Adversarial boundary conditions:
//!    - Rotator counter near `usize::MAX` overflow.
//!    - Custom spare mask without custom active mask.
//!    - Minimal and constrained topologies.

#![allow(
    clippy::all,
    clippy::pedantic,
    clippy::nursery,
    clippy::undocumented_unsafe_blocks
)]

use std::sync::atomic::Ordering;
use std::sync::{Arc, Barrier};
use std::thread;

use k3_core::affinity::{ActiveControlRotator, ControlPlaneCapacityConfig};
use k3_core::error::K3Error;

/// Challenge 1: High-concurrency round-robin fairness on `next_active_control_core()`.
///
/// Spawns 64 threads, each executing 10,000 iterations against a shared `ResolvedControlCapacity`.
/// Total operations = 640,000.
/// Default config has 2 active cores (`[6, 7]`).
/// Invariant: Because 640,000 is evenly divisible by 2, every core must be selected exactly 320,000 times.
#[test]
fn test_concurrency_stress_next_active_control_core_fairness() {
    let cfg = ControlPlaneCapacityConfig::default();
    let res = Arc::new(cfg.resolve_default().expect("resolve default"));

    assert_eq!(res.active_control_cores, vec![6, 7]);

    let num_threads = 64;
    let ops_per_thread = 10_000;
    let total_ops = num_threads * ops_per_thread;

    let barrier = Arc::new(Barrier::new(num_threads));
    let mut handles = Vec::with_capacity(num_threads);

    for _ in 0..num_threads {
        let res_clone = Arc::clone(&res);
        let bar_clone = Arc::clone(&barrier);

        handles.push(thread::spawn(move || {
            bar_clone.wait();
            let mut local_counts = [0usize; 8];
            for _ in 0..ops_per_thread {
                let core = res_clone.next_active_control_core();
                local_counts[core] += 1;
            }
            local_counts
        }));
    }

    let mut global_counts = [0usize; 8];
    for handle in handles {
        let local = handle.join().expect("thread did not panic");
        for (i, count) in local.iter().enumerate() {
            global_counts[i] += count;
        }
    }

    assert_eq!(
        global_counts[6] + global_counts[7],
        total_ops,
        "Total operations must equal sum of active core counts"
    );
    assert_eq!(
        global_counts[6],
        total_ops / 2,
        "Core 6 must receive exactly half of total operations"
    );
    assert_eq!(
        global_counts[7],
        total_ops / 2,
        "Core 7 must receive exactly half of total operations"
    );

    for core in [0, 1, 2, 3, 4, 5] {
        assert_eq!(
            global_counts[core], 0,
            "Inactive core {core} must never be selected by rotator"
        );
    }
}

/// Challenge 2: Multi-threaded fairness on `ActiveControlRotator` with odd core set.
///
/// Tests rotator with 3 active cores `[1, 3, 5]`.
/// 60 threads each performing 10,000 calls = 600,000 total calls.
/// Invariant: Each of the 3 cores must be selected exactly 200,000 times.
#[test]
fn test_concurrency_stress_active_control_rotator_odd_core_set() {
    let cores = vec![1, 3, 5];
    let rotator = Arc::new(ActiveControlRotator::new(cores.clone()));

    let num_threads = 60;
    let ops_per_thread = 10_000;
    let total_ops = num_threads * ops_per_thread;

    let barrier = Arc::new(Barrier::new(num_threads));
    let mut handles = Vec::with_capacity(num_threads);

    for _ in 0..num_threads {
        let rot = Arc::clone(&rotator);
        let bar = Arc::clone(&barrier);

        handles.push(thread::spawn(move || {
            bar.wait();
            let mut local_counts = [0usize; 8];
            for _ in 0..ops_per_thread {
                let core = rot.next_core();
                local_counts[core] += 1;
            }
            local_counts
        }));
    }

    let mut global_counts = [0usize; 8];
    for handle in handles {
        let local = handle.join().expect("thread did not panic");
        for (i, count) in local.iter().enumerate() {
            global_counts[i] += count;
        }
    }

    assert_eq!(global_counts[1], total_ops / 3);
    assert_eq!(global_counts[3], total_ops / 3);
    assert_eq!(global_counts[5], total_ops / 3);

    for core in [0, 2, 4, 6, 7] {
        assert_eq!(global_counts[core], 0);
    }
}

/// Challenge 3: Single-core active set under heavy concurrency.
///
/// On minimal 1-core configurations, `next_active_control_core()` must reliably
/// yield the single core 100% of the time across all threads without contention failure.
#[test]
fn test_concurrency_stress_single_core_minimal_config() {
    let cfg = ControlPlaneCapacityConfig::minimal();
    let res = Arc::new(cfg.resolve_default().expect("resolve minimal"));
    assert_eq!(res.active_control_cores, vec![7]);

    let num_threads = 32;
    let ops_per_thread = 10_000;

    let barrier = Arc::new(Barrier::new(num_threads));
    let mut handles = Vec::with_capacity(num_threads);

    for _ in 0..num_threads {
        let res_clone = Arc::clone(&res);
        let bar_clone = Arc::clone(&barrier);

        handles.push(thread::spawn(move || {
            bar_clone.wait();
            for _ in 0..ops_per_thread {
                let core = res_clone.next_active_control_core();
                assert_eq!(core, 7);
            }
        }));
    }

    for handle in handles {
        handle.join().expect("thread passed");
    }
}

/// Challenge 4: Concurrent `reset()` interleaved with `next_core()`.
///
/// Tests that concurrently calling `reset()` while reader threads are actively advancing
/// the rotator does not produce data races, out-of-bounds indices, or crashes.
#[test]
fn test_concurrency_rotator_interleaved_reset() {
    let cores = vec![2, 4, 6];
    let rotator = Arc::new(ActiveControlRotator::new(cores));

    let num_readers = 16;
    let ops_per_reader = 20_000;

    let barrier = Arc::new(Barrier::new(num_readers + 1));
    let mut handles = Vec::with_capacity(num_readers + 1);

    // Reader threads
    for _ in 0..num_readers {
        let rot = Arc::clone(&rotator);
        let bar = Arc::clone(&barrier);

        handles.push(thread::spawn(move || {
            bar.wait();
            for _ in 0..ops_per_reader {
                let core = rot.next_core();
                assert!(core == 2 || core == 4 || core == 6);
            }
        }));
    }

    // Reset thread
    let rot_reset = Arc::clone(&rotator);
    let bar_reset = Arc::clone(&barrier);
    handles.push(thread::spawn(move || {
        bar_reset.wait();
        for _ in 0..1_000 {
            rot_reset.reset();
            thread::yield_now();
        }
    }));

    for handle in handles {
        handle.join().expect("thread finished cleanly");
    }
}

/// Challenge 5: Atomic cursor wraparound behavior near `usize::MAX`.
///
/// Verifies behavior when `round_robin_counter` approaches and overflows `usize::MAX`.
#[test]
fn test_rotator_atomic_counter_near_max_usize_wraparound() {
    let capacity = ControlPlaneCapacityConfig::default()
        .resolve_default()
        .expect("resolve");

    // Cores are [6, 7]
    assert_eq!(capacity.active_control_cores, vec![6, 7]);

    // Position cursor at usize::MAX - 3
    capacity
        .round_robin_counter
        .store(usize::MAX - 3, Ordering::SeqCst);

    // Sequence of indices:
    // usize::MAX - 3 -> odd / even depending on value.
    // usize::MAX is 2^64 - 1 (odd).
    // usize::MAX - 3 is even -> idx % 2 == 0 -> core 6
    // usize::MAX - 2 is odd  -> idx % 2 == 1 -> core 7
    // usize::MAX - 1 is even -> idx % 2 == 0 -> core 6
    // usize::MAX     is odd  -> idx % 2 == 1 -> core 7
    // Wraps to 0     is even -> idx % 2 == 0 -> core 6
    // 1              is odd  -> idx % 2 == 1 -> core 7
    assert_eq!(capacity.next_active_control_core(), 6);
    assert_eq!(capacity.next_active_control_core(), 7);
    assert_eq!(capacity.next_active_control_core(), 6);
    assert_eq!(capacity.next_active_control_core(), 7);
    // After wrapping to 0:
    assert_eq!(capacity.next_active_control_core(), 6);
    assert_eq!(capacity.next_active_control_core(), 7);
}

/// Challenge 6: Property Test — Exhaustive verification of disjoint masks across ALL 256 online masks.
///
/// For EVERY possible online mask in `0..=0x00FF` (256 subsets):
/// 1. Test `default()` config (2 active, 1 spare)
/// 2. Test `minimal()` config (1 active, 0 spare)
///
/// Invariant: Whenever resolution succeeds:
/// - `(active_control_mask & spare_recovery_mask) == 0`
/// - `(active_control_mask & x100_compute_cap_mask) == 0`
/// - `active_control_mask != 0`
/// - `!active_control_cores.is_empty()`
/// - `(active_control_mask | spare_recovery_mask | x100_compute_cap_mask) == valid_online`
/// - `(spare_recovery_mask & x100_compute_cap_mask) == spare_recovery_mask` (lendable spare)
#[test]
fn test_property_exhaustive_online_masks_disjoint_invariants() {
    let configs = [
        ("default", ControlPlaneCapacityConfig::default()),
        ("minimal", ControlPlaneCapacityConfig::minimal()),
    ];

    for (name, cfg) in &configs {
        for online_mask in 0_u64..=255_u64 {
            let online_cores_count = online_mask.count_ones() as usize;

            match cfg.resolve_with_online_mask(online_mask) {
                Ok(res) => {
                    // 1. Mandatory disjointness
                    assert_eq!(
                        res.active_control_mask & res.spare_recovery_mask,
                        0,
                        "[{name}, mask=0x{online_mask:02X}] active and spare masks MUST be strictly disjoint"
                    );
                    assert_eq!(
                        res.active_control_mask & res.x100_compute_cap_mask,
                        0,
                        "[{name}, mask=0x{online_mask:02X}] active and compute cap masks MUST be strictly disjoint"
                    );

                    // 2. Active control mask must be non-zero
                    assert_ne!(
                        res.active_control_mask, 0,
                        "[{name}, mask=0x{online_mask:02X}] active_control_mask cannot be zero on success"
                    );
                    assert!(
                        !res.active_control_cores.is_empty(),
                        "[{name}, mask=0x{online_mask:02X}] active_control_cores cannot be empty on success"
                    );

                    // 3. Exact core count matches mask bits
                    assert_eq!(
                        res.active_control_mask.count_ones() as usize,
                        res.active_control_cores.len()
                    );
                    assert_eq!(
                        res.spare_recovery_mask.count_ones() as usize,
                        res.spare_recovery_cores.len()
                    );
                    assert_eq!(
                        res.x100_compute_cap_mask.count_ones() as usize,
                        res.x100_max_compute_cores
                    );

                    // 4. Lendable spare capacity invariant
                    assert_eq!(
                        res.spare_recovery_mask & res.x100_compute_cap_mask,
                        res.spare_recovery_mask,
                        "[{name}, mask=0x{online_mask:02X}] spare recovery cores must be lendable to compute"
                    );

                    // 5. Union of active + compute cap covers all valid online cores
                    assert_eq!(
                        res.active_control_mask | res.x100_compute_cap_mask,
                        online_mask,
                        "[{name}, mask=0x{online_mask:02X}] active + compute cap must partition valid online cores"
                    );
                }
                Err(err) => {
                    // Must fail with AffinityError
                    assert_eq!(err.diagnostic_code(), "affinity_error");

                    // Verify failure reason:
                    // Either online_cores_count < active + spare, or online_cores_count == 0
                    let needed = cfg.active_control_cores + cfg.spare_recovery_cores;
                    assert!(
                        online_cores_count < needed || online_cores_count == 0,
                        "[{name}, mask=0x{online_mask:02X}] Unexpected resolution error when cores={online_cores_count} >= needed={needed}: {err}"
                    );
                }
            }
        }
    }
}

/// Challenge 7: Property Test — Combinatorial Grid Search across `(active, spare)` and online topologies.
///
/// Explores 8 active core counts (1..=8) x 9 spare core counts (0..=8) x 256 online masks
/// = 18,432 combinations!
/// Verifies that every single successful resolution strictly obeys:
/// - `(active_control_mask & spare_recovery_mask) == 0`
/// - `(active_control_mask & x100_compute_cap_mask) == 0`
/// - `!active_control_cores.is_empty()`
#[test]
fn test_property_combinatorial_grid_search_all_capacity_dimensions() {
    let mut total_successes = 0;
    let mut total_failures = 0;

    for active in 1..=8 {
        for spare in 0..=8 {
            let cfg = ControlPlaneCapacityConfig::new()
                .with_active_cores(active)
                .with_spare_cores(spare);

            for online_mask in 0_u64..=255_u64 {
                match cfg.resolve_with_online_mask(online_mask) {
                    Ok(res) => {
                        total_successes += 1;
                        assert_eq!(
                            res.active_control_mask & res.spare_recovery_mask,
                            0,
                            "Intersection between active and spare must be 0"
                        );
                        assert_eq!(
                            res.active_control_mask & res.x100_compute_cap_mask,
                            0,
                            "Intersection between active and compute cap must be 0"
                        );
                        assert_ne!(res.active_control_mask, 0);
                        assert!(!res.active_control_cores.is_empty());
                        assert_eq!(res.active_control_cores.len(), active);
                        assert_eq!(res.spare_recovery_cores.len(), spare);
                    }
                    Err(err) => {
                        total_failures += 1;
                        assert_eq!(err.diagnostic_code(), "affinity_error");
                    }
                }
            }
        }
    }

    assert!(total_successes > 0);
    assert!(total_failures > 0);
    println!(
        "Combinatorial grid search complete: {total_successes} valid configurations, {total_failures} rejected configurations."
    );
}

/// Challenge 8: Adversarial Test — Custom spare mask without custom active mask.
///
/// Investigates behavior when `custom_spare_recovery_mask` is provided but `custom_active_control_mask` is omitted.
/// Per R3, resolving with 0 active control cores must be rejected or active control cores must be populated.
#[test]
fn test_adversarial_custom_spare_without_custom_active_mask() {
    let cfg = ControlPlaneCapacityConfig::default().with_custom_spare_mask(0x0020);

    let res = cfg.resolve_default();
    match res {
        Ok(resolved) => {
            println!(
                "OBSERVED: resolved with active_control_mask=0x{:04X}, cores={:?}",
                resolved.active_control_mask, resolved.active_control_cores
            );
            // If it succeeded, does it violate the non-empty active cores invariant?
            assert!(
                !resolved.active_control_cores.is_empty(),
                "BUG CONFIRMED: resolve_default() returned Ok with empty active_control_cores! Calling next_active_control_core() will panic!"
            );
            assert_ne!(
                resolved.active_control_mask, 0,
                "BUG CONFIRMED: resolve_default() returned Ok with active_control_mask == 0!"
            );
        }
        Err(K3Error::AffinityError { reason, .. }) => {
            // If rejected, that is compliant with R3 ("0 active control cores requested or resolved")
            println!("Configuration rejected as expected: {reason}");
        }
        Err(other) => panic!("Unexpected error type: {other:?}"),
    }
}

/// Challenge 9: Adversarial Test — Arithmetic overflow on total requested cores.
///
/// If requested `active_control_cores` and `spare_recovery_cores` sum to greater than `usize::MAX`,
/// `affinity.rs` should reject with `K3Error::AffinityError` rather than crashing via arithmetic panic.
#[test]
fn test_adversarial_arithmetic_overflow_requested_cores() {
    let cfg = ControlPlaneCapacityConfig::new()
        .with_active_cores(usize::MAX)
        .with_spare_cores(1);

    // If resolve_default panics with overflow, that's a crash vulnerability
    let result = std::panic::catch_unwind(|| {
        let _ = cfg.resolve_default();
    });

    assert!(
        result.is_ok(),
        "CRASH VULNERABILITY: resolve_default() panicked with arithmetic overflow on total_needed!"
    );
}

/// Challenge 10: Adversarial Test — Custom compute cap mask with offline cores.
///
/// `custom_active_control_mask` and `custom_spare_recovery_mask` both validate `custom_mask & !valid_online == 0`.
/// However, `custom_x100_compute_cap_mask` does not check against `valid_online`, allowing offline cores
/// to be returned as valid compute reservation caps.
#[test]
fn test_adversarial_custom_compute_cap_specifying_offline_cores() {
    // Only cores 6 and 7 are online
    let online_mask = 0x00C0;
    // Request compute cap on cores 0..3 (which are completely offline)
    let cfg = ControlPlaneCapacityConfig::minimal().with_custom_compute_cap_mask(0x000F);

    let res = cfg.resolve_with_online_mask(online_mask);
    match res {
        Ok(resolved) => {
            println!(
                "OBSERVED: resolved with x100_compute_cap_mask=0x{:04X} on online_mask=0x{:04X}",
                resolved.x100_compute_cap_mask, online_mask
            );
            assert_eq!(
                resolved.x100_compute_cap_mask & !online_mask,
                0,
                "BUG CONFIRMED: custom_x100_compute_cap_mask allowed offline cores in compute cap!"
            );
        }
        Err(K3Error::AffinityError { reason, .. }) => {
            println!("Rejected offline compute cap cores as expected: {reason}");
        }
        Err(other) => panic!("Unexpected error type: {other:?}"),
    }
}
