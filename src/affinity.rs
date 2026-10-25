//! `SpacemiT` K3 Cluster 0 (`X100`) control-plane capacity model, core resolution, and compute bounding.
//!
//! Enforces dedicated control-plane reservation, spare recovery core allocation, compute capping,
//! and thread-safe round-robin task distribution per ADR-007, ADR-008, ADR-041, and `features/core-affinity.md`.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use serde::{Deserialize, Serialize};

use crate::error::K3Error;

/// Physical core bitmask for Cluster 0 (`X100` Out-of-Order cores 0–7, `zvl256b`).
pub const X100_CLUSTER_MASK: u64 = 0x00FF;

/// Physical core bitmask for Cluster 1 (`A100` In-Order cores 8–15, `zvl1024b` + IME).
pub const A100_CLUSTER_MASK: u64 = 0xFF00;

/// Physical core bitmask for all 16 cores across Cluster 0 and Cluster 1 (`0xFFFF`).
pub const SOC_CLUSTER_MASK: u64 = X100_CLUSTER_MASK | A100_CLUSTER_MASK;

/// Default active control core bitmask on standard 8-core `X100` (cores 6–7, ADR-007).
pub const DEFAULT_ACTIVE_CONTROL_MASK: u64 = 0x00C0;

/// Default spare recovery core bitmask on standard 8-core `X100` (core 5, ADR-007).
pub const DEFAULT_SPARE_RECOVERY_MASK: u64 = 0x0020;

/// Default compute reservation cap bitmask on `X100` (cores 0–5, ADR-007).
pub const DEFAULT_X100_COMPUTE_CAP_MASK: u64 = 0x003F;

/// Default maximum compute thread reservation on `X100` (cores 0–5).
pub const DEFAULT_X100_MAX_COMPUTE_CORES: usize = 6;

/// Default maximum compute thread reservation on `A100` (all 8 cores 8–15).
pub const DEFAULT_A100_MAX_COMPUTE_CORES: usize = 8;

/// Alias for [`DEFAULT_X100_MAX_COMPUTE_CORES`] for engine naming compatibility.
pub const X100_DEFAULT_MAX_COMPUTE_THREADS: usize = DEFAULT_X100_MAX_COMPUTE_CORES;

/// Alias for [`DEFAULT_A100_MAX_COMPUTE_CORES`] for engine naming compatibility.
pub const A100_DEFAULT_MAX_COMPUTE_THREADS: usize = DEFAULT_A100_MAX_COMPUTE_CORES;

/// Alias for [`DEFAULT_X100_MAX_COMPUTE_CORES`] for manager naming compatibility.
pub const DEFAULT_MAX_X100_COMPUTE_CORES: usize = DEFAULT_X100_MAX_COMPUTE_CORES;

/// Alias for [`DEFAULT_A100_MAX_COMPUTE_CORES`] for manager naming compatibility.
pub const DEFAULT_MAX_A100_COMPUTE_CORES: usize = DEFAULT_A100_MAX_COMPUTE_CORES;

/// Configuration schema for control-plane capacity reservation on `X100`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ControlPlaneCapacityConfig {
    /// Number of active control cores requested (min: 1).
    pub active_control_cores: usize,
    /// Number of spare recovery cores requested (default: 1, min: 0).
    pub spare_recovery_cores: usize,
    /// Optional explicit bitmask for active control cores (must be a subset of `X100` `0x00FF`).
    pub custom_active_control_mask: Option<u64>,
    /// Optional explicit bitmask for spare recovery cores (must be a subset of `X100` `0x00FF`).
    pub custom_spare_recovery_mask: Option<u64>,
    /// Optional explicit bitmask for `X100` compute cap (must not overlap dedicated active control cores).
    pub custom_x100_compute_cap_mask: Option<u64>,
}

impl Default for ControlPlaneCapacityConfig {
    fn default() -> Self {
        Self {
            active_control_cores: 2,
            spare_recovery_cores: 1,
            custom_active_control_mask: None,
            custom_spare_recovery_mask: None,
            custom_x100_compute_cap_mask: None,
        }
    }
}

impl ControlPlaneCapacityConfig {
    /// Returns default 8-core `X100` configuration (2 active cores, 1 spare core).
    #[must_use]
    pub const fn new() -> Self {
        Self {
            active_control_cores: 2,
            spare_recovery_cores: 1,
            custom_active_control_mask: None,
            custom_spare_recovery_mask: None,
            custom_x100_compute_cap_mask: None,
        }
    }

    /// Alias for default 8-core `X100` configuration.
    #[must_use]
    pub const fn default_8core() -> Self {
        Self::new()
    }

    /// Returns minimal configuration for constrained topologies (1 active core, 0 spare cores).
    #[must_use]
    pub const fn minimal() -> Self {
        Self {
            active_control_cores: 1,
            spare_recovery_cores: 0,
            custom_active_control_mask: None,
            custom_spare_recovery_mask: None,
            custom_x100_compute_cap_mask: None,
        }
    }

    /// Alias for [`Self::minimal`].
    #[must_use]
    pub const fn minimal_1core() -> Self {
        Self::minimal()
    }

    /// Sets requested active control core count.
    #[must_use]
    pub const fn with_active_cores(mut self, cores: usize) -> Self {
        self.active_control_cores = cores;
        self
    }

    /// Alias for [`Self::with_active_cores`].
    #[must_use]
    pub const fn with_active_control_cores(self, cores: usize) -> Self {
        self.with_active_cores(cores)
    }

    /// Sets requested spare recovery core count.
    #[must_use]
    pub const fn with_spare_cores(mut self, cores: usize) -> Self {
        self.spare_recovery_cores = cores;
        self
    }

    /// Alias for [`Self::with_spare_cores`].
    #[must_use]
    pub const fn with_spare_recovery_cores(self, cores: usize) -> Self {
        self.with_spare_cores(cores)
    }

    /// Sets custom active control core bitmask.
    #[must_use]
    pub const fn with_custom_active_mask(mut self, mask: u64) -> Self {
        self.custom_active_control_mask = Some(mask);
        self
    }

    /// Alias for [`Self::with_custom_active_mask`].
    #[must_use]
    pub const fn with_custom_active_control_mask(self, mask: u64) -> Self {
        self.with_custom_active_mask(mask)
    }

    /// Sets custom spare recovery core bitmask.
    #[must_use]
    pub const fn with_custom_spare_mask(mut self, mask: u64) -> Self {
        self.custom_spare_recovery_mask = Some(mask);
        self
    }

    /// Alias for [`Self::with_custom_spare_mask`].
    #[must_use]
    pub const fn with_custom_spare_recovery_mask(self, mask: u64) -> Self {
        self.with_custom_spare_mask(mask)
    }

    /// Sets custom `X100` compute cap bitmask.
    #[must_use]
    pub const fn with_custom_compute_cap_mask(mut self, mask: u64) -> Self {
        self.custom_x100_compute_cap_mask = Some(mask);
        self
    }

    /// Alias for [`Self::with_custom_compute_cap_mask`].
    #[must_use]
    pub const fn with_custom_x100_compute_cap_mask(self, mask: u64) -> Self {
        self.with_custom_compute_cap_mask(mask)
    }

    /// Resolves capacity against the standard 8-core `X100` topology (`0x00FF`).
    ///
    /// # Errors
    ///
    /// Returns [`K3Error::AffinityError`] if configuration invariants are violated.
    pub fn resolve_default(&self) -> Result<ResolvedControlCapacity, K3Error> {
        self.resolve_with_online_mask(X100_CLUSTER_MASK)
    }

    /// Resolves capacity against a specified online core bitmask.
    ///
    /// # Errors
    ///
    /// Returns [`K3Error::AffinityError`] if configuration invariants are violated.
    #[allow(clippy::too_many_lines)]
    pub fn resolve_with_online_mask(
        &self,
        online_mask: u64,
    ) -> Result<ResolvedControlCapacity, K3Error> {
        let valid_online = online_mask & X100_CLUSTER_MASK;
        if valid_online == 0 {
            return Err(K3Error::AffinityError {
                requested_mask: online_mask,
                requested_threads: 0,
                reason: "no online X100 general-purpose cores available for control plane"
                    .to_owned(),
            });
        }

        if self.active_control_cores == 0 && self.custom_active_control_mask.is_none() {
            return Err(K3Error::AffinityError {
                requested_mask: 0,
                requested_threads: 0,
                reason: "control plane requires at least 1 active control core".to_owned(),
            });
        }

        let (active_mask, active_cores) = if let Some(custom_mask) = self.custom_active_control_mask
        {
            if (custom_mask & !X100_CLUSTER_MASK) != 0 {
                return Err(K3Error::AffinityError {
                    requested_mask: custom_mask,
                    requested_threads: custom_mask.count_ones() as usize,
                    reason: format!(
                        "active control mask 0x{custom_mask:04X} encroaches on Cluster 1 (A100, 0x{A100_CLUSTER_MASK:04X}) or outside X100 (0x{X100_CLUSTER_MASK:04X})"
                    ),
                });
            }
            if custom_mask == 0 {
                return Err(K3Error::AffinityError {
                    requested_mask: 0,
                    requested_threads: 0,
                    reason: "active control mask cannot be zero".to_owned(),
                });
            }
            if (custom_mask & !valid_online) != 0 {
                return Err(K3Error::AffinityError {
                    requested_mask: custom_mask,
                    requested_threads: custom_mask.count_ones() as usize,
                    reason: format!(
                        "active control mask 0x{custom_mask:04X} specifies offline cores (offline: 0x{:04X})",
                        custom_mask & !valid_online
                    ),
                });
            }
            let mut cores = Vec::new();
            for bit in 0..8 {
                if (custom_mask & (1 << bit)) != 0 {
                    cores.push(bit);
                }
            }
            (custom_mask, cores)
        } else {
            (0, Vec::new())
        };

        let (spare_mask, spare_cores) = if let Some(custom_mask) = self.custom_spare_recovery_mask
        {
            if (custom_mask & !X100_CLUSTER_MASK) != 0 {
                return Err(K3Error::AffinityError {
                    requested_mask: custom_mask,
                    requested_threads: custom_mask.count_ones() as usize,
                    reason: format!(
                        "spare recovery mask 0x{custom_mask:04X} encroaches on Cluster 1 (A100, 0x{A100_CLUSTER_MASK:04X}) or outside X100 (0x{X100_CLUSTER_MASK:04X})"
                    ),
                });
            }
            if (custom_mask & !valid_online) != 0 {
                return Err(K3Error::AffinityError {
                    requested_mask: custom_mask,
                    requested_threads: custom_mask.count_ones() as usize,
                    reason: format!(
                        "spare recovery mask 0x{custom_mask:04X} specifies offline cores (offline: 0x{:04X})",
                        custom_mask & !valid_online
                    ),
                });
            }
            let mut cores = Vec::new();
            for bit in 0..8 {
                if (custom_mask & (1 << bit)) != 0 {
                    cores.push(bit);
                }
            }
            (custom_mask, cores)
        } else {
            (0, Vec::new())
        };

        let (
            resolved_active_mask,
            resolved_active_cores,
            resolved_spare_mask,
            resolved_spare_cores,
        ) = match (
            self.custom_active_control_mask.is_some(),
            self.custom_spare_recovery_mask.is_some(),
        ) {
            // Case 1: Both active and spare custom masks provided
            (true, true) => {
                if (active_mask & spare_mask) != 0 {
                    return Err(K3Error::AffinityError {
                        requested_mask: active_mask,
                        requested_threads: active_cores.len(),
                        reason: format!(
                            "active control mask 0x{active_mask:04X} overlaps spare recovery mask 0x{spare_mask:04X} (intersection: 0x{:04X})",
                            active_mask & spare_mask
                        ),
                    });
                }
                (active_mask, active_cores, spare_mask, spare_cores)
            }

            // Case 2: Custom active mask provided, spare recovery mask default (0)
            (true, false) => (active_mask, active_cores, 0, Vec::new()),

            // Case 3: Custom spare mask provided, active control cores auto-allocated from remaining online cores
            (false, true) => {
                let available_for_active = valid_online & !spare_mask;
                let available_count = available_for_active.count_ones() as usize;
                if self.active_control_cores > available_count {
                    return Err(K3Error::AffinityError {
                        requested_mask: valid_online,
                        requested_threads: self.active_control_cores,
                        reason: format!(
                            "requested {} active control cores exceeds {} available online X100 cores (excluding custom spare mask 0x{spare_mask:04X})",
                            self.active_control_cores, available_count
                        ),
                    });
                }

                let mut act_cores = Vec::new();
                let mut act_mask = 0_u64;
                for bit in (0..8).rev() {
                    if (available_for_active & (1 << bit)) != 0 {
                        act_cores.push(bit);
                        act_mask |= 1 << bit;
                        if act_cores.len() == self.active_control_cores {
                            break;
                        }
                    }
                }
                act_cores.sort_unstable();

                (act_mask, act_cores, spare_mask, spare_cores)
            }

            // Case 4: Neither custom mask provided, full auto-allocation
            (false, false) => {
                let online_count = valid_online.count_ones() as usize;
                let Some(total_needed) = self.active_control_cores.checked_add(self.spare_recovery_cores) else {
                    return Err(K3Error::AffinityError {
                        requested_mask: valid_online,
                        requested_threads: usize::MAX,
                        reason: format!(
                            "requested control core count overflow: {} active + {} spare exceeds usize::MAX",
                            self.active_control_cores, self.spare_recovery_cores
                        ),
                    });
                };
                if total_needed > online_count {
                    return Err(K3Error::AffinityError {
                        requested_mask: valid_online,
                        requested_threads: total_needed,
                        reason: format!(
                            "requested {total_needed} control cores ({} active + {} spare) exceeds {online_count} online X100 cores",
                            self.active_control_cores, self.spare_recovery_cores
                        ),
                    });
                }

                let mut available_bits = Vec::new();
                for bit in (0..8).rev() {
                    if (valid_online & (1 << bit)) != 0 {
                        available_bits.push(bit);
                    }
                }

                let mut act_cores = Vec::new();
                let mut act_mask = 0_u64;
                for &bit in available_bits.iter().take(self.active_control_cores) {
                    act_cores.push(bit);
                    act_mask |= 1 << bit;
                }
                act_cores.sort_unstable();

                let mut spr_cores = Vec::new();
                let mut spr_mask = 0_u64;
                for &bit in available_bits
                    .iter()
                    .skip(self.active_control_cores)
                    .take(self.spare_recovery_cores)
                {
                    spr_cores.push(bit);
                    spr_mask |= 1 << bit;
                }
                spr_cores.sort_unstable();

                (act_mask, act_cores, spr_mask, spr_cores)
            }
        };

        if resolved_active_mask == 0 || resolved_active_cores.is_empty() {
            return Err(K3Error::AffinityError {
                requested_mask: resolved_active_mask,
                requested_threads: resolved_active_cores.len(),
                reason: "control plane requires at least 1 active control core (resolved 0)".to_string(),
            });
        }

        let resolved_compute_cap_mask = if let Some(custom_cap) = self.custom_x100_compute_cap_mask
        {
            if (custom_cap & !X100_CLUSTER_MASK) != 0 {
                return Err(K3Error::AffinityError {
                    requested_mask: custom_cap,
                    requested_threads: custom_cap.count_ones() as usize,
                    reason: format!(
                        "compute cap mask 0x{custom_cap:04X} encroaches on Cluster 1 (A100, 0x{A100_CLUSTER_MASK:04X}) or outside X100 (0x{X100_CLUSTER_MASK:04X})"
                    ),
                });
            }
            if (custom_cap & !valid_online) != 0 {
                return Err(K3Error::AffinityError {
                    requested_mask: custom_cap,
                    requested_threads: custom_cap.count_ones() as usize,
                    reason: format!(
                        "compute cap mask 0x{custom_cap:04X} specifies offline cores (offline: 0x{:04X})",
                        custom_cap & !valid_online
                    ),
                });
            }
            if (custom_cap & resolved_active_mask) != 0 {
                return Err(K3Error::AffinityError {
                    requested_mask: custom_cap,
                    requested_threads: custom_cap.count_ones() as usize,
                    reason: format!(
                        "compute cap mask 0x{custom_cap:04X} overlaps dedicated active control mask 0x{resolved_active_mask:04X} (intersection: 0x{:04X})",
                        custom_cap & resolved_active_mask
                    ),
                });
            }
            custom_cap
        } else {
            valid_online & !resolved_active_mask
        };

        let max_x100_compute = resolved_compute_cap_mask.count_ones() as usize;

        Ok(ResolvedControlCapacity {
            active_control_mask: resolved_active_mask,
            active_control_cores: resolved_active_cores,
            spare_recovery_mask: resolved_spare_mask,
            spare_recovery_cores: resolved_spare_cores,
            x100_compute_cap_mask: resolved_compute_cap_mask,
            x100_max_compute_cores: max_x100_compute,
            a100_max_compute_cores: DEFAULT_A100_MAX_COMPUTE_CORES,
            round_robin_counter: Arc::new(AtomicUsize::new(0)),
        })
    }
}

/// Resolved capacity state representing `X100` control-plane allocations and compute bounds.
#[derive(Debug, Clone)]
pub struct ResolvedControlCapacity {
    /// Dedicated active control core bitmask.
    pub active_control_mask: u64,
    /// Ordered list of active control core IDs in ascending order.
    pub active_control_cores: Vec<usize>,
    /// Spare recovery core bitmask.
    pub spare_recovery_mask: u64,
    /// Ordered list of spare recovery core IDs in ascending order.
    pub spare_recovery_cores: Vec<usize>,
    /// Permitted `X100` compute reservation bitmask.
    pub x100_compute_cap_mask: u64,
    /// Maximum compute workers allowed on `X100`.
    pub x100_max_compute_cores: usize,
    /// Maximum compute workers allowed on `A100`.
    pub a100_max_compute_cores: usize,
    /// Thread-safe atomic round-robin counter for active control distribution.
    pub round_robin_counter: Arc<AtomicUsize>,
}

impl PartialEq for ResolvedControlCapacity {
    fn eq(&self, other: &Self) -> bool {
        self.active_control_mask == other.active_control_mask
            && self.active_control_cores == other.active_control_cores
            && self.spare_recovery_mask == other.spare_recovery_mask
            && self.spare_recovery_cores == other.spare_recovery_cores
            && self.x100_compute_cap_mask == other.x100_compute_cap_mask
            && self.x100_max_compute_cores == other.x100_max_compute_cores
            && self.a100_max_compute_cores == other.a100_max_compute_cores
    }
}

impl Eq for ResolvedControlCapacity {}

impl Serialize for ResolvedControlCapacity {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut state = serializer.serialize_struct("ResolvedControlCapacity", 7)?;
        state.serialize_field("active_control_mask", &self.active_control_mask)?;
        state.serialize_field("active_control_cores", &self.active_control_cores)?;
        state.serialize_field("spare_recovery_mask", &self.spare_recovery_mask)?;
        state.serialize_field("spare_recovery_cores", &self.spare_recovery_cores)?;
        state.serialize_field("x100_compute_cap_mask", &self.x100_compute_cap_mask)?;
        state.serialize_field("x100_max_compute_cores", &self.x100_max_compute_cores)?;
        state.serialize_field("a100_max_compute_cores", &self.a100_max_compute_cores)?;
        state.end()
    }
}

impl<'de> Deserialize<'de> for ResolvedControlCapacity {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct CapacityData {
            active_control_mask: u64,
            active_control_cores: Vec<usize>,
            spare_recovery_mask: u64,
            spare_recovery_cores: Vec<usize>,
            x100_compute_cap_mask: u64,
            x100_max_compute_cores: usize,
            a100_max_compute_cores: usize,
        }

        let data = CapacityData::deserialize(deserializer)?;
        Ok(Self {
            active_control_mask: data.active_control_mask,
            active_control_cores: data.active_control_cores,
            spare_recovery_mask: data.spare_recovery_mask,
            spare_recovery_cores: data.spare_recovery_cores,
            x100_compute_cap_mask: data.x100_compute_cap_mask,
            x100_max_compute_cores: data.x100_max_compute_cores,
            a100_max_compute_cores: data.a100_max_compute_cores,
            round_robin_counter: Arc::new(AtomicUsize::new(0)),
        })
    }
}

impl ResolvedControlCapacity {
    /// Pure, deterministic round-robin core selection given an explicit counter index.
    ///
    /// # Panics
    ///
    /// Panics if `active_control_cores` is empty (which is prevented by capacity validation).
    #[must_use]
    pub fn round_robin_active_core(&self, index: usize) -> usize {
        assert!(
            !self.active_control_cores.is_empty(),
            "active control cores cannot be empty"
        );
        self.active_control_cores[index % self.active_control_cores.len()]
    }

    /// Thread-safe round-robin core selection advancing the internal atomic cursor.
    #[must_use]
    pub fn next_active_control_core(&self) -> usize {
        let idx = self.round_robin_counter.fetch_add(1, Ordering::Relaxed);
        self.round_robin_active_core(idx)
    }

    /// Returns the primary spare recovery core for health supervisor / watchdog binding.
    #[must_use]
    pub fn spare_recovery_core(&self) -> Option<usize> {
        self.spare_recovery_cores.first().copied()
    }

    /// Returns `true` if `core` is an active control core.
    #[must_use]
    pub const fn is_active_control_core(&self, core: usize) -> bool {
        if core >= 64 {
            return false;
        }
        (self.active_control_mask & (1 << core)) != 0
    }

    /// Returns `true` if `core` is a spare recovery core.
    #[must_use]
    pub const fn is_spare_recovery_core(&self, core: usize) -> bool {
        if core >= 64 {
            return false;
        }
        (self.spare_recovery_mask & (1 << core)) != 0
    }

    /// Returns `true` if `core` is permitted for `X100` compute reservations.
    #[must_use]
    pub const fn is_x100_compute_allowed(&self, core: usize) -> bool {
        if core >= 64 {
            return false;
        }
        (self.x100_compute_cap_mask & (1 << core)) != 0
    }

    /// Returns the resolved dedicated active control core bitmask.
    #[must_use]
    pub const fn active_control_mask(&self) -> u64 {
        self.active_control_mask
    }

    /// Returns the slice of active control core IDs in ascending order.
    #[must_use]
    pub fn active_control_cores(&self) -> &[usize] {
        &self.active_control_cores
    }

    /// Returns the resolved spare recovery core bitmask.
    #[must_use]
    pub const fn spare_recovery_mask(&self) -> u64 {
        self.spare_recovery_mask
    }

    /// Returns the slice of spare recovery core IDs in ascending order.
    #[must_use]
    pub fn spare_recovery_cores(&self) -> &[usize] {
        &self.spare_recovery_cores
    }

    /// Returns the `X100` compute reservation cap bitmask.
    #[must_use]
    pub const fn x100_compute_cap_mask(&self) -> u64 {
        self.x100_compute_cap_mask
    }

    /// Returns maximum allowed compute threads on `X100`.
    #[must_use]
    pub const fn x100_max_compute_cores(&self) -> usize {
        self.x100_max_compute_cores
    }

    /// Returns maximum allowed compute threads on `A100`.
    #[must_use]
    pub const fn a100_max_compute_cores(&self) -> usize {
        self.a100_max_compute_cores
    }

    /// Alias for [`Self::x100_max_compute_cores`].
    #[must_use]
    pub const fn max_x100_compute_cores(&self) -> usize {
        self.x100_max_compute_cores
    }

    /// Alias for [`Self::a100_max_compute_cores`].
    #[must_use]
    pub const fn max_a100_compute_cores(&self) -> usize {
        self.a100_max_compute_cores
    }

    /// Returns total reserved control-plane cores (`active + spare`).
    #[must_use]
    pub fn total_reserved_cores(&self) -> usize {
        self.active_control_cores.len() + self.spare_recovery_cores.len()
    }

    /// Validates a requested compute mask against the resolved capacity bounds.
    ///
    /// # Errors
    ///
    /// Returns [`K3Error::AffinityError`] if the compute mask overlaps dedicated active control cores
    /// or exceeds the permitted `X100` compute cap mask.
    pub fn validate_compute_mask(&self, mask: u64) -> Result<(), K3Error> {
        if (mask & !SOC_CLUSTER_MASK) != 0 {
            return Err(K3Error::AffinityError {
                requested_mask: mask,
                requested_threads: mask.count_ones() as usize,
                reason: format!(
                    "compute mask 0x{mask:04X} specifies invalid cores outside valid 16-core SoC mask 0x{SOC_CLUSTER_MASK:04X}"
                ),
            });
        }
        if (mask & self.active_control_mask) != 0 {
            return Err(K3Error::AffinityError {
                requested_mask: mask,
                requested_threads: mask.count_ones() as usize,
                reason: format!(
                    "compute mask 0x{mask:04X} overlaps dedicated active control mask 0x{:04X}",
                    self.active_control_mask
                ),
            });
        }
        if (mask & !self.x100_compute_cap_mask & X100_CLUSTER_MASK) != 0 {
            return Err(K3Error::AffinityError {
                requested_mask: mask,
                requested_threads: mask.count_ones() as usize,
                reason: format!(
                    "compute mask 0x{mask:04X} exceeds permitted X100 compute cap mask 0x{:04X}",
                    self.x100_compute_cap_mask
                ),
            });
        }
        Ok(())
    }

    /// Creates an independent [`ActiveControlRotator`] bound to this capacity's active control cores.
    #[must_use]
    pub fn rotator(&self) -> ActiveControlRotator {
        ActiveControlRotator::new(self.active_control_cores.clone())
    }
}

/// Standalone thread-safe round-robin rotator across a fixed core set.
#[derive(Debug, Clone)]
pub struct ActiveControlRotator {
    cores: Vec<usize>,
    cursor: Arc<AtomicUsize>,
}

impl ActiveControlRotator {
    /// Creates a new rotator for the specified core list.
    #[must_use]
    pub fn new(cores: Vec<usize>) -> Self {
        Self {
            cores,
            cursor: Arc::new(AtomicUsize::new(0)),
        }
    }

    /// Yields the next core in round-robin sequence.
    ///
    /// # Panics
    ///
    /// Panics if `cores` is empty.
    #[must_use]
    pub fn next_core(&self) -> usize {
        assert!(!self.cores.is_empty(), "core set cannot be empty");
        let idx = self.cursor.fetch_add(1, Ordering::Relaxed);
        self.cores[idx % self.cores.len()]
    }

    /// Resets the internal counter to 0.
    pub fn reset(&self) {
        self.cursor.store(0, Ordering::Relaxed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_8core_x100_capacity_resolution() {
        let cfg = ControlPlaneCapacityConfig::default();
        let res = cfg.resolve_default().expect("default resolution");

        assert_eq!(res.active_control_mask, 0x00C0);
        assert_eq!(res.active_control_cores, vec![6, 7]);
        assert_eq!(res.spare_recovery_mask, 0x0020);
        assert_eq!(res.spare_recovery_cores, vec![5]);
        assert_eq!(res.x100_compute_cap_mask, 0x003F);
        assert_eq!(res.x100_max_compute_cores, 6);
        assert_eq!(res.a100_max_compute_cores, 8);
        assert_eq!(res.total_reserved_cores(), 3);
        assert_eq!(res.active_control_mask(), 0x00C0);
        assert_eq!(res.spare_recovery_mask(), 0x0020);
        assert_eq!(res.x100_compute_cap_mask(), 0x003F);
    }

    #[test]
    fn test_lendable_spare_recovery_core_semantics() {
        let res = ControlPlaneCapacityConfig::default()
            .resolve_default()
            .expect("default resolution");

        assert_eq!(res.spare_recovery_mask & res.x100_compute_cap_mask, 0x0020);
        assert_eq!(res.active_control_mask & res.x100_compute_cap_mask, 0x0000);
        assert_eq!(res.active_control_mask & res.spare_recovery_mask, 0x0000);
    }

    #[test]
    fn test_minimal_1core_resolution_on_8core_topology() {
        let cfg = ControlPlaneCapacityConfig::minimal();
        let res = cfg.resolve_default().expect("minimal resolution");

        assert_eq!(res.active_control_mask, 0x0080);
        assert_eq!(res.active_control_cores, vec![7]);
        assert_eq!(res.spare_recovery_mask, 0x0000);
        assert_eq!(res.spare_recovery_cores, [] as [usize; 0]);
        assert_eq!(res.x100_compute_cap_mask, 0x007F);
        assert_eq!(res.x100_max_compute_cores, 7);
        assert_eq!(res.spare_recovery_core(), None);
    }

    #[test]
    fn test_constrained_4core_topology_default_config() {
        let cfg = ControlPlaneCapacityConfig::default();
        let res = cfg
            .resolve_with_online_mask(0x000F)
            .expect("constrained 4-core default");

        assert_eq!(res.active_control_mask, 0x000C);
        assert_eq!(res.active_control_cores, vec![2, 3]);
        assert_eq!(res.spare_recovery_mask, 0x0002);
        assert_eq!(res.spare_recovery_cores, vec![1]);
        assert_eq!(res.x100_compute_cap_mask, 0x0003);
        assert_eq!(res.x100_max_compute_cores, 2);
    }

    #[test]
    fn test_constrained_4core_topology_minimal_config() {
        let cfg = ControlPlaneCapacityConfig::minimal();
        let res = cfg
            .resolve_with_online_mask(0x000F)
            .expect("constrained 4-core minimal");

        assert_eq!(res.active_control_mask, 0x0008);
        assert_eq!(res.active_control_cores, vec![3]);
        assert_eq!(res.spare_recovery_mask, 0x0000);
        assert_eq!(res.x100_compute_cap_mask, 0x0007);
        assert_eq!(res.x100_max_compute_cores, 3);
    }

    #[test]
    fn test_constrained_2core_topology_minimal_config() {
        let cfg = ControlPlaneCapacityConfig::minimal_1core();
        let res = cfg
            .resolve_with_online_mask(0x0003)
            .expect("constrained 2-core minimal");

        assert_eq!(res.active_control_mask, 0x0002);
        assert_eq!(res.active_control_cores, vec![1]);
        assert_eq!(res.spare_recovery_mask, 0x0000);
        assert_eq!(res.x100_compute_cap_mask, 0x0001);
        assert_eq!(res.x100_max_compute_cores, 1);
    }

    #[test]
    fn test_constrained_1core_topology_minimal_config() {
        let cfg = ControlPlaneCapacityConfig::minimal();
        let res = cfg
            .resolve_with_online_mask(0x0001)
            .expect("constrained 1-core minimal");

        assert_eq!(res.active_control_mask, 0x0001);
        assert_eq!(res.active_control_cores, vec![0]);
        assert_eq!(res.spare_recovery_mask, 0x0000);
        assert_eq!(res.x100_compute_cap_mask, 0x0000);
        assert_eq!(res.x100_max_compute_cores, 0);
    }

    #[test]
    fn test_non_contiguous_online_cores_resolution() {
        let cfg = ControlPlaneCapacityConfig::default();
        let res = cfg
            .resolve_with_online_mask(0x0055)
            .expect("non-contiguous resolution");

        assert_eq!(res.active_control_mask, 0x0050);
        assert_eq!(res.active_control_cores, vec![4, 6]);
        assert_eq!(res.spare_recovery_mask, 0x0004);
        assert_eq!(res.spare_recovery_cores, vec![2]);
        assert_eq!(res.x100_compute_cap_mask, 0x0005);
        assert_eq!(res.x100_max_compute_cores, 2);
    }

    #[test]
    fn test_custom_masks_valid_resolution() {
        let cfg = ControlPlaneCapacityConfig::new()
            .with_custom_active_control_mask(0x0030)
            .with_custom_spare_recovery_mask(0x0008)
            .with_custom_x100_compute_cap_mask(0x0007);
        let res = cfg.resolve_default().expect("custom valid resolution");

        assert_eq!(res.active_control_mask, 0x0030);
        assert_eq!(res.active_control_cores, vec![4, 5]);
        assert_eq!(res.spare_recovery_mask, 0x0008);
        assert_eq!(res.spare_recovery_cores, vec![3]);
        assert_eq!(res.x100_compute_cap_mask, 0x0007);
        assert_eq!(res.x100_max_compute_cores, 3);
    }

    #[test]
    fn test_reject_zero_active_control_cores_requested() {
        let cfg = ControlPlaneCapacityConfig::default().with_active_control_cores(0);
        let err = cfg.resolve_default().unwrap_err();

        assert_eq!(err.diagnostic_code(), "affinity_error");
        assert!(err.to_string().contains("at least 1 active control core"));
    }

    #[test]
    fn test_reject_custom_active_mask_zero() {
        let cfg = ControlPlaneCapacityConfig::default().with_custom_active_mask(0);
        let err = cfg.resolve_default().unwrap_err();

        assert_eq!(err.diagnostic_code(), "affinity_error");
        assert!(err.to_string().contains("active control mask cannot be zero"));
    }

    #[test]
    fn test_reject_active_mask_on_a100_cluster() {
        let cfg = ControlPlaneCapacityConfig::default().with_custom_active_mask(0x0100);
        let err = cfg.resolve_default().unwrap_err();

        assert_eq!(err.diagnostic_code(), "affinity_error");
        assert!(err.to_string().contains("Cluster 1 (A100"));
    }

    #[test]
    fn test_reject_spare_mask_on_a100_cluster() {
        let cfg = ControlPlaneCapacityConfig::default().with_custom_spare_mask(0x0800);
        let err = cfg.resolve_default().unwrap_err();

        assert_eq!(err.diagnostic_code(), "affinity_error");
        assert!(err.to_string().contains("Cluster 1 (A100"));
    }

    #[test]
    fn test_reject_active_mask_beyond_valid_topology() {
        let cfg = ControlPlaneCapacityConfig::default().with_custom_active_mask(0x0001_0000);
        let err = cfg.resolve_default().unwrap_err();

        assert_eq!(err.diagnostic_code(), "affinity_error");
        assert!(err.to_string().contains("outside X100"));
    }

    #[test]
    fn test_reject_compute_cap_mask_outside_x100() {
        let cfg = ControlPlaneCapacityConfig::default().with_custom_compute_cap_mask(0x013F);
        let err = cfg.resolve_default().unwrap_err();

        assert_eq!(err.diagnostic_code(), "affinity_error");
        assert!(err.to_string().contains("outside X100"));
    }

    #[test]
    fn test_reject_active_and_spare_mask_overlap() {
        let cfg = ControlPlaneCapacityConfig::default()
            .with_custom_active_mask(0x00C0)
            .with_custom_spare_mask(0x0040);
        let err = cfg.resolve_default().unwrap_err();

        assert_eq!(err.diagnostic_code(), "affinity_error");
        assert!(err.to_string().contains("overlaps spare recovery mask"));
    }

    #[test]
    fn test_reject_compute_cap_overlapping_active_control() {
        let cfg = ControlPlaneCapacityConfig::default()
            .with_custom_active_mask(0x00C0)
            .with_custom_compute_cap_mask(0x007F);
        let err = cfg.resolve_default().unwrap_err();

        assert_eq!(err.diagnostic_code(), "affinity_error");
        assert!(err
            .to_string()
            .contains("overlaps dedicated active control mask"));
    }

    #[test]
    fn test_reject_requested_cores_exceeding_online_x100() {
        let cfg = ControlPlaneCapacityConfig::default()
            .with_active_cores(2)
            .with_spare_cores(2);
        let err = cfg.resolve_with_online_mask(0x0007).unwrap_err();

        assert_eq!(err.diagnostic_code(), "affinity_error");
        assert!(err.to_string().contains("exceeds 3 online X100 cores"));
    }

    #[test]
    fn test_reject_custom_mask_specifies_offline_core() {
        let cfg = ControlPlaneCapacityConfig::default().with_custom_active_mask(0x00C0);
        let err = cfg.resolve_with_online_mask(0x007F).unwrap_err();

        assert_eq!(err.diagnostic_code(), "affinity_error");
        assert!(err.to_string().contains("specifies offline cores"));
    }

    #[test]
    fn test_reject_zero_online_x100_cores() {
        let cfg = ControlPlaneCapacityConfig::default();
        let err = cfg.resolve_with_online_mask(0x0000).unwrap_err();

        assert_eq!(err.diagnostic_code(), "affinity_error");
        assert!(err
            .to_string()
            .contains("no online X100 general-purpose cores available"));
    }

    #[test]
    fn test_deterministic_round_robin_active_core_assignment() {
        let res = ControlPlaneCapacityConfig::default()
            .resolve_default()
            .expect("resolve");

        let expected = [6, 7, 6, 7, 6, 7, 6, 7, 6, 7];
        for (i, &exp) in expected.iter().enumerate() {
            assert_eq!(res.round_robin_active_core(i), exp);
        }

        let mut observed = Vec::new();
        for _ in 0..10 {
            observed.push(res.next_active_control_core());
        }
        assert_eq!(observed, expected);
    }

    #[test]
    fn test_spare_recovery_core_selector_and_predicates() {
        let default_res = ControlPlaneCapacityConfig::default()
            .resolve_default()
            .expect("resolve");
        assert_eq!(default_res.spare_recovery_core(), Some(5));
        assert!(default_res.is_spare_recovery_core(5));
        assert!(!default_res.is_spare_recovery_core(6));
        assert!(default_res.is_active_control_core(6));
        assert!(default_res.is_active_control_core(7));
        assert!(!default_res.is_active_control_core(5));
        assert!(default_res.is_x100_compute_allowed(5));
        assert!(default_res.is_x100_compute_allowed(0));
        assert!(!default_res.is_x100_compute_allowed(6));
        assert!(!default_res.is_x100_compute_allowed(100));
        assert!(!default_res.is_active_control_core(100));
        assert!(!default_res.is_spare_recovery_core(100));

        let min_res = ControlPlaneCapacityConfig::minimal()
            .resolve_default()
            .expect("resolve");
        assert_eq!(min_res.spare_recovery_core(), None);
    }

    #[test]
    fn test_serde_roundtrip_resolved_capacity() {
        let res = ControlPlaneCapacityConfig::default()
            .resolve_default()
            .expect("resolve");
        let json = serde_json::to_string(&res).expect("serialize");
        let deserialized: ResolvedControlCapacity =
            serde_json::from_str(&json).expect("deserialize");

        assert_eq!(res, deserialized);
        assert_eq!(deserialized.next_active_control_core(), 6);
        assert_eq!(deserialized.next_active_control_core(), 7);
    }

    #[test]
    fn test_active_control_rotator() {
        let rotator = ActiveControlRotator::new(vec![4, 6]);
        assert_eq!(rotator.next_core(), 4);
        assert_eq!(rotator.next_core(), 6);
        assert_eq!(rotator.next_core(), 4);
        rotator.reset();
        assert_eq!(rotator.next_core(), 4);
    }

    #[test]
    fn test_custom_spare_mask_without_custom_active_allocates_active_cores() {
        let cfg = ControlPlaneCapacityConfig::default()
            .with_custom_spare_mask(0x0020);
        let resolved = cfg.resolve_default().expect("valid capacity");

        assert_eq!(resolved.active_control_mask, 0x00C0);
        assert_eq!(resolved.active_control_cores, vec![6, 7]);
        assert_eq!(resolved.spare_recovery_mask, 0x0020);
        assert_eq!(resolved.spare_recovery_core(), Some(5));
        assert_eq!(resolved.x100_compute_cap_mask, 0x003F);
        assert_eq!(resolved.next_active_control_core(), 6);
        assert_eq!(resolved.next_active_control_core(), 7);
    }

    #[test]
    fn test_custom_spare_mask_exhausting_active_cores_rejected() {
        let cfg = ControlPlaneCapacityConfig::minimal()
            .with_custom_spare_mask(0x0020);
        let err = cfg.resolve_with_online_mask(0x0020).unwrap_err();

        assert_eq!(err.diagnostic_code(), "affinity_error");
        assert!(err.to_string().contains("exceeds 0 available online X100 cores"));
    }

    #[test]
    fn test_reject_arithmetic_overflow_requested_cores() {
        let cfg = ControlPlaneCapacityConfig::new()
            .with_active_cores(usize::MAX)
            .with_spare_cores(1);
        let err = cfg.resolve_default().unwrap_err();
        assert_eq!(err.diagnostic_code(), "affinity_error");
        assert!(err.to_string().contains("overflow"));
    }

    #[test]
    fn test_reject_custom_compute_cap_specifies_offline_core() {
        let cfg = ControlPlaneCapacityConfig::minimal()
            .with_custom_compute_cap_mask(0x000F);
        let err = cfg.resolve_with_online_mask(0x00C0).unwrap_err();
        assert_eq!(err.diagnostic_code(), "affinity_error");
        assert!(err.to_string().contains("specifies offline cores"));
    }

    #[test]
    fn test_reject_compute_mask_outside_soc_range() {
        let res = ControlPlaneCapacityConfig::default().resolve_default().unwrap();
        let err = res.validate_compute_mask(0x0001_0000).unwrap_err();
        assert_eq!(err.diagnostic_code(), "affinity_error");
        assert!(err.to_string().contains("outside valid 16-core SoC mask"));
    }
}
