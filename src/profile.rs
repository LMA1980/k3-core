//! Structured cost profiling, hardware-ceiling efficiency metrics, and tuning advisories.
//!
//! Provides the canonical machine-readable [`CostProfile`] baseline schema for static scheduling
//! weights and operator tuning advisories (US-04, ADR-010, ADR-032, ADR-034).

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::str::FromStr;

use crate::error::K3Error;
use crate::gguf::TensorDescriptor;

/// Identity and provenance of an evaluated model.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelIdentity {
    /// Model identifier or name (from `general.name` or file stem).
    pub name: String,
    /// Architecture identifier (from `general.architecture`).
    pub architecture: String,
    /// Model file path.
    pub file_path: String,
    /// Model content hex digest (first 1 MB or header sample).
    pub file_digest: String,
}

/// Quantization format detection and tensor distribution summary.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct QuantizationInfo {
    /// Primary weight quantization format (highest byte footprint).
    pub primary: String,
    /// Count of tensors per quantization format.
    pub summary: BTreeMap<String, usize>,
    /// Aggregate byte footprint per quantization format.
    pub tensor_bytes: BTreeMap<String, u64>,
}

impl QuantizationInfo {
    /// Detects the primary quantization format and aggregates tensor counts and bytes.
    #[must_use]
    pub fn from_descriptors(descriptors: &[TensorDescriptor]) -> Self {
        let mut summary = BTreeMap::new();
        let mut tensor_bytes = BTreeMap::new();

        for desc in descriptors {
            let q_str = desc.quant().as_str().to_owned();
            *summary.entry(q_str.clone()).or_insert(0) += 1;
            *tensor_bytes.entry(q_str).or_insert(0) += desc.byte_len() as u64;
        }

        let primary = tensor_bytes
            .iter()
            .max_by_key(|(_, bytes)| *bytes)
            .map_or_else(|| "F32".to_owned(), |(k, _)| k.clone());

        Self {
            primary,
            summary,
            tensor_bytes,
        }
    }
}

/// Target cluster placement and runtime configuration.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ClusterPlacement {
    /// Target compute cluster: "x100", "a100", or "both".
    pub cluster: String,
    /// Effective core affinity bitmask formatted in hex (e.g., "0x00FF").
    pub effective_core_mask: String,
    /// Total number of cores allocated.
    pub allocated_cores: usize,
    /// Number of Cluster 0 (X100) cores allocated.
    pub x100_cores: usize,
    /// Number of Cluster 1 (A100) cores allocated.
    pub a100_cores: usize,
    /// Multi-core coordination profile ("spin", "futex", "rayon").
    pub coordination_profile: String,
    /// Cross-cluster workload split policy ("none", "phase", "op", "row").
    pub split_strategy: String,
    /// `SpacemiT` IME matrix acceleration toggle ("on" or "off").
    pub ime: String,
}

/// Allocated core counts and topology breakdown.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CoreAllocation {
    /// Total allocated core count.
    pub total: usize,
    /// Total allocated core count (alias for compatibility).
    pub total_cores: usize,
    /// Number of X100 cores allocated.
    pub x100_cores: usize,
    /// Number of A100 cores allocated.
    pub a100_cores: usize,
    /// Effective core bitmask string.
    pub effective_mask: String,
}

/// Detailed execution cost and hardware-ceiling attainment metrics.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExecutionCost {
    /// Prompt prefill cost per token in milliseconds (ms/token).
    pub prefill_cost_per_token_ms: f64,
    /// Prompt prefill throughput in tokens per second.
    pub prefill_tokens_per_sec: f64,
    /// Achieved percentage of theoretical compute capacity (% of `peak_macs`).
    pub prefill_hardware_ceiling_pct: f64,
    /// Autoregressive token decode cost per token in milliseconds (ms/token).
    pub decode_cost_per_token_ms: f64,
    /// Autoregressive token decode rate in tokens per second.
    pub decode_tokens_per_sec: f64,
    /// Achieved percentage of available memory bandwidth (% of `memory_bandwidth`).
    pub decode_hardware_ceiling_pct: f64,
    /// Derived static scheduling pass-cost weight for fair time-slicing (ADR-010).
    pub static_scheduling_weight: f64,
}

/// Resident memory footprint and locked capacity requirement.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MemoryFootprintProfile {
    /// Page-aligned locked weight bytes.
    pub weight_bytes: u64,
    /// Page-aligned locked KV cache arena bytes.
    pub kv_bytes: u64,
    /// Total required locked bytes (`weight_bytes + kv_bytes`).
    pub total_required_bytes: u64,
    /// Measured peak resident memory footprint in bytes.
    pub peak_resident_bytes: u64,
}

/// Evidence-based tuning advisory recommendations derived from ceiling attainment (ADR-032).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TuningAdvisory {
    /// Primary bottleneck classification ("bandwidth-bound" or "compute-bound").
    pub bottleneck_classification: String,
    /// Recommended cluster targeting ("x100", "a100", or "both").
    pub recommended_cluster: String,
    /// Recommended cross-cluster split strategy (e.g., Some("phase") or None).
    pub recommended_split: Option<String>,
    /// Recommended `SpacemiT` IME matrix acceleration posture.
    pub recommended_ime: bool,
    /// Analytical rationale for the advisory recommendation.
    pub rationale: String,
    /// Actionable operator recommendation items.
    pub recommendations: Vec<String>,
    /// Explicit notice preserving operator authority per ADR-032.
    pub operator_authority_notice: String,
}

/// Machine-readable cost baseline profile (US-04 Scenario 2, ADR-010, ADR-032, ADR-034).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CostProfile {
    /// Evaluated model identity and provenance.
    pub model_identity: ModelIdentity,
    /// Primary weight quantization level and tensor summary.
    pub quantization_level: QuantizationInfo,
    /// Processor cluster placement and runtime policy.
    pub cluster_placement: ClusterPlacement,
    /// Core allocation and topology breakdown.
    pub allocated_cores: CoreAllocation,
    /// Workload split strategy ("none", "phase", "op", "row").
    pub split_strategy: String,
    /// Structured execution cost metrics.
    pub execution_cost: ExecutionCost,
    /// Prefill latency per token in ms/token.
    pub prefill_cost_per_token_ms: f64,
    /// Prefill throughput in tokens per second.
    pub prefill_tokens_per_sec: f64,
    /// Prefill hardware ceiling percentage (% of `peak_macs`).
    pub prefill_hardware_ceiling_pct: f64,
    /// Decode throughput in tokens per second.
    pub decode_tokens_per_sec: f64,
    /// Decode latency per token in ms/token.
    pub decode_cost_per_token_ms: f64,
    /// Decode hardware ceiling percentage (% of `memory_bandwidth`).
    pub decode_hardware_ceiling_pct: f64,
    /// Static scheduling pass-cost weight for fair time-slicing (ADR-010).
    pub static_scheduling_weight: f64,
    /// Static scheduling weight alias.
    pub scheduling_weight: f64,
    /// Structured resident memory footprint.
    pub memory_footprint: MemoryFootprintProfile,
    /// Locked weight bytes.
    pub weight_bytes: u64,
    /// Locked KV cache bytes.
    pub kv_bytes: u64,
    /// Total required locked bytes (`weight_bytes + kv_bytes`).
    pub total_required_bytes: u64,
    /// Measured peak resident memory footprint (`peak_resident_bytes >= total_required_bytes`).
    pub peak_resident_bytes: u64,
    /// Evidence-based tuning advisory recommendations.
    pub tuning_advisory: TuningAdvisory,
}

impl CostProfile {
    /// Constructs a baseline mock cost profile for an unprofiled model.
    #[must_use]
    pub fn unprofiled_default(model_identity: ModelIdentity) -> Self {
        Self {
            model_identity,
            quantization_level: QuantizationInfo {
                primary: "unprofiled".to_owned(),
                summary: BTreeMap::new(),
                tensor_bytes: BTreeMap::new(),
            },
            cluster_placement: ClusterPlacement {
                cluster: "x100".to_owned(),
                effective_core_mask: format!(
                    "{:#06x}",
                    crate::affinity::DEFAULT_X100_COMPUTE_CAP_MASK
                ),
                allocated_cores: CONSERVATIVE_DEFAULT_X100_CORES,
                x100_cores: CONSERVATIVE_DEFAULT_X100_CORES,
                a100_cores: CONSERVATIVE_DEFAULT_A100_CORES,
                coordination_profile: "rayon".to_owned(),
                split_strategy: "none".to_owned(),
                ime: "off".to_owned(),
            },
            allocated_cores: CoreAllocation {
                total: CONSERVATIVE_DEFAULT_X100_CORES,
                total_cores: CONSERVATIVE_DEFAULT_X100_CORES,
                x100_cores: CONSERVATIVE_DEFAULT_X100_CORES,
                a100_cores: CONSERVATIVE_DEFAULT_A100_CORES,
                effective_mask: format!("{:#06x}", crate::affinity::DEFAULT_X100_COMPUTE_CAP_MASK),
            },
            split_strategy: "none".to_owned(),
            execution_cost: ExecutionCost {
                prefill_cost_per_token_ms: 0.0,
                prefill_tokens_per_sec: 0.0,
                prefill_hardware_ceiling_pct: 0.0,
                decode_cost_per_token_ms: 0.0,
                decode_tokens_per_sec: 0.0,
                decode_hardware_ceiling_pct: 0.0,
                static_scheduling_weight: 1.0,
            },
            prefill_cost_per_token_ms: 0.0,
            prefill_tokens_per_sec: 0.0,
            prefill_hardware_ceiling_pct: 0.0,
            decode_tokens_per_sec: 0.0,
            decode_cost_per_token_ms: 0.0,
            decode_hardware_ceiling_pct: 0.0,
            static_scheduling_weight: 1.0,
            scheduling_weight: 1.0,
            memory_footprint: MemoryFootprintProfile {
                weight_bytes: 0,
                kv_bytes: 0,
                total_required_bytes: 0,
                peak_resident_bytes: 0,
            },
            weight_bytes: 0,
            kv_bytes: 0,
            total_required_bytes: 0,
            peak_resident_bytes: 0,
            tuning_advisory: TuningAdvisory {
                bottleneck_classification: "unprofiled".to_owned(),
                recommended_cluster: "x100".to_owned(),
                recommended_split: None,
                recommended_ime: false,
                rationale: "Model is unprofiled; conservative defaults active".to_owned(),
                recommendations: vec!["Run 'k3-diag' to profile".to_owned()],
                operator_authority_notice: "Advisory only per ADR-032".to_owned(),
            },
        }
    }

    /// Serializes this cost profile to pretty-printed JSON.
    ///
    /// # Errors
    /// Returns [`serde_json::Error`] if serialization fails.
    pub fn to_json_pretty(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }

    /// Deserializes a cost profile from a JSON string.
    ///
    /// # Errors
    /// Returns [`serde_json::Error`] if deserialization fails.
    pub fn from_json_str(json: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(json)
    }
}

/// Priority class for model scheduling and forward-pass arbitration (ADR-014).
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default, Serialize, Deserialize,
)]
#[serde(rename_all = "lowercase")]
pub enum PriorityClass {
    /// Standard throughput priority class (default for unprofiled and batch models).
    #[default]
    Throughput,
    /// High-priority latency class (preempts throughput forward passes).
    Latency,
}

impl PriorityClass {
    /// Returns canonical lowercase string identifier ("throughput" or "latency").
    #[must_use]
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::Throughput => "throughput",
            Self::Latency => "latency",
        }
    }

    /// Parses a priority class string.
    ///
    /// # Errors
    /// Returns [`K3Error::CapabilityMismatch`] if string is unrecognized.
    pub fn parse(s: &str) -> Result<Self, K3Error> {
        s.parse()
    }
}

impl FromStr for PriorityClass {
    type Err = K3Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_ascii_lowercase().as_str() {
            "throughput" => Ok(Self::Throughput),
            "latency" => Ok(Self::Latency),
            _ => Err(K3Error::CapabilityMismatch {
                architecture: "priority_class".to_owned(),
                reason: format!("unknown priority class '{s}'; expected 'throughput' or 'latency'"),
            }),
        }
    }
}

impl std::fmt::Display for PriorityClass {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Target processor compute pool on the `SpacemiT` K3 `SoC` (ADR-006, ADR-028, ADR-031).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ComputePool {
    /// Cluster 0: General-purpose X100 Out-of-Order cores (0–7, 0x00FF, VLEN=256).
    #[default]
    X100,
    /// Cluster 1: Specialized A100 In-Order cores (8–15, 0xFF00, VLEN=1024 + IME).
    A100,
    /// Heterogeneous dual-cluster placement across both X100 and A100.
    Both,
}

impl ComputePool {
    /// Returns canonical lowercase string ("x100", "a100", "both").
    #[must_use]
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::X100 => "x100",
            Self::A100 => "a100",
            Self::Both => "both",
        }
    }

    /// Returns physical core cluster bitmask.
    #[must_use]
    pub const fn core_mask(&self) -> u64 {
        match self {
            Self::X100 => crate::affinity::X100_CLUSTER_MASK,
            Self::A100 => crate::affinity::A100_CLUSTER_MASK,
            Self::Both => crate::affinity::SOC_CLUSTER_MASK,
        }
    }

    /// Returns default maximum compute thread reservation.
    #[must_use]
    pub const fn default_max_compute_cores(&self) -> usize {
        match self {
            Self::X100 => crate::affinity::DEFAULT_X100_MAX_COMPUTE_CORES, // 6
            Self::A100 => crate::affinity::DEFAULT_A100_MAX_COMPUTE_CORES, // 8
            Self::Both => {
                crate::affinity::DEFAULT_X100_MAX_COMPUTE_CORES
                    + crate::affinity::DEFAULT_A100_MAX_COMPUTE_CORES
            } // 14
        }
    }

    /// Parses a compute pool string.
    ///
    /// # Errors
    /// Returns [`K3Error::CapabilityMismatch`] if string is unrecognized.
    pub fn parse(s: &str) -> Result<Self, K3Error> {
        s.parse()
    }
}

impl FromStr for ComputePool {
    type Err = K3Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_ascii_lowercase().as_str() {
            "x100" => Ok(Self::X100),
            "a100" => Ok(Self::A100),
            "both" => Ok(Self::Both),
            _ => Err(K3Error::CapabilityMismatch {
                architecture: "compute_pool".to_owned(),
                reason: format!("unknown compute pool '{s}'; expected 'x100', 'a100', or 'both'"),
            }),
        }
    }
}

impl std::fmt::Display for ComputePool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Profiling and operator tuning status for a model placement (ADR-031).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ProfileStatus {
    /// Model has no diagnostic cost profile or operator tuning; runs under conservative defaults.
    #[default]
    Unprofiled,
    /// Model has been profiled and configured with an operator tuning profile.
    Profiled,
}

impl ProfileStatus {
    /// Returns `true` if the status is [`ProfileStatus::Unprofiled`].
    #[must_use]
    pub const fn is_unprofiled(&self) -> bool {
        matches!(self, Self::Unprofiled)
    }

    /// Returns canonical lowercase string ("unprofiled" or "profiled").
    #[must_use]
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::Unprofiled => "unprofiled",
            Self::Profiled => "profiled",
        }
    }

    /// Parses a profile status string.
    ///
    /// # Errors
    /// Returns [`K3Error::CapabilityMismatch`] if string is unrecognized.
    pub fn parse(s: &str) -> Result<Self, K3Error> {
        s.parse()
    }
}

impl FromStr for ProfileStatus {
    type Err = K3Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_ascii_lowercase().as_str() {
            "unprofiled" => Ok(Self::Unprofiled),
            "profiled" => Ok(Self::Profiled),
            _ => Err(K3Error::CapabilityMismatch {
                architecture: "profile_status".to_owned(),
                reason: format!(
                    "unknown profile status '{s}'; expected 'unprofiled' or 'profiled'"
                ),
            }),
        }
    }
}

impl std::fmt::Display for ProfileStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Standard runtime conservative default context window bound in tokens (US-07 R1, ADR-027).
pub const CONSERVATIVE_DEFAULT_N_CTX: usize = 2048;

/// Conservative context window bound in tokens for constrained/unprivileged test harnesses (AGENTS.md Rule 5).
pub const CONSTRAINED_TEST_N_CTX: usize = 64;

/// Conservative default X100 compute core reservation (max 6 cores, within `x100_compute_cap_mask` 0x003F).
pub const CONSERVATIVE_DEFAULT_X100_CORES: usize = crate::affinity::DEFAULT_X100_MAX_COMPUTE_CORES; // 6

/// Conservative default A100 core reservation (0 cores; A100 strictly isolated per ADR-028).
pub const CONSERVATIVE_DEFAULT_A100_CORES: usize = 0;

/// Actionable operator advisory notice emitted when a model runs under conservative defaults (ADR-031, ADR-032).
pub const CONSERVATIVE_TUNING_ADVISORY: &str = "Model is running with safe conservative defaults (unprofiled). Run 'k3-diag' to profile hardware performance and apply an operator tuning profile.";

/// Operator-authoritative runtime tuning profile and execution configuration (ADR-006, ADR-014, ADR-027, ADR-030, ADR-031, ADR-032).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TuningProfile {
    /// Optional logical model identifier.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model_id: Option<String>,
    /// Target processor compute pool.
    pub compute_pool: ComputePool,
    /// Canonical cluster string representation ("x100", "a100", "both").
    pub cluster: String,
    /// Number of Cluster 0 (X100) compute cores allocated (max 6, within 0x003F).
    pub x100_cores: usize,
    /// Number of Cluster 1 (A100) compute cores allocated (0 for unprofiled defaults).
    pub a100_cores: usize,
    /// Scheduling priority class ("throughput" or "latency", ADR-014).
    pub priority_class: PriorityClass,
    /// Fixed context window limit in tokens (ADR-027).
    pub n_ctx: usize,
    /// Whether dynamic idle core lending is enabled (ADR-030; default false).
    pub lend_idle_cores: bool,
    /// Whether dynamic idle core borrowing is enabled (ADR-030; default false).
    pub borrow_idle_cores: bool,
    /// Current profiling status (Unprofiled vs Profiled, ADR-031).
    pub profile_status: ProfileStatus,
    /// Explicit boolean flag indicating whether the model is unprofiled (`profile_status == ProfileStatus::Unprofiled`).
    pub unprofiled: bool,
    /// Actionable advisory notice when running unprofiled (cleared once profiled per ADR-031).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub advisory_notice: Option<String>,
}

impl Default for TuningProfile {
    fn default() -> Self {
        Self::conservative_default()
    }
}

impl TuningProfile {
    /// Constructs the shipped conservative default profile for an unprofiled model (ADR-028, ADR-031):
    /// - Compute pool: X100 general-purpose pool only (`cluster = "x100"`, `a100_cores = 0`)
    /// - Priority class: standard `throughput` (ADR-014)
    /// - Context window: conservative runtime default `n_ctx = 2048` (ADR-027)
    /// - Core spillover: idle core lending and borrowing disabled (`lend_idle_cores = false`, `borrow_idle_cores = false`) (ADR-030)
    /// - Status: `unprofiled = true` / `profile_status = ProfileStatus::Unprofiled` with actionable advisory notice
    #[must_use]
    pub fn conservative_default() -> Self {
        Self::conservative_default_with_n_ctx(CONSERVATIVE_DEFAULT_N_CTX)
    }

    /// Constructs the conservative default profile with a specified context window bound `n_ctx`.
    ///
    /// Use [`CONSTRAINED_TEST_N_CTX`] (64) for constrained test harnesses and unprivileged environments per AGENTS.md Rule 5.
    #[must_use]
    pub fn conservative_default_with_n_ctx(n_ctx: usize) -> Self {
        Self {
            model_id: None,
            compute_pool: ComputePool::X100,
            cluster: "x100".to_owned(),
            x100_cores: CONSERVATIVE_DEFAULT_X100_CORES,
            a100_cores: CONSERVATIVE_DEFAULT_A100_CORES,
            priority_class: PriorityClass::Throughput,
            n_ctx,
            lend_idle_cores: false,
            borrow_idle_cores: false,
            profile_status: ProfileStatus::Unprofiled,
            unprofiled: true,
            advisory_notice: Some(CONSERVATIVE_TUNING_ADVISORY.to_owned()),
        }
    }

    /// Returns `true` if this profile is unprofiled.
    #[must_use]
    pub const fn is_unprofiled(&self) -> bool {
        self.unprofiled
    }

    /// Transitions this profile to an operator-applied tuned state, updating placement, priority,
    /// context window, and core lending parameters, while clearing the `unprofiled` flag and advisory notice (ADR-031, ADR-032).
    #[allow(clippy::too_many_arguments)]
    pub fn apply_operator_tuning(
        &mut self,
        compute_pool: ComputePool,
        x100_cores: usize,
        a100_cores: usize,
        priority_class: PriorityClass,
        n_ctx: usize,
        lend_idle_cores: bool,
        borrow_idle_cores: bool,
    ) {
        self.compute_pool = compute_pool;
        compute_pool.as_str().clone_into(&mut self.cluster);
        self.x100_cores = x100_cores;
        self.a100_cores = a100_cores;
        self.priority_class = priority_class;
        self.n_ctx = n_ctx;
        self.lend_idle_cores = lend_idle_cores;
        self.borrow_idle_cores = borrow_idle_cores;
        self.profile_status = ProfileStatus::Profiled;
        self.unprofiled = false;
        self.advisory_notice = None;
    }

    /// Serializes this profile to pretty-printed JSON.
    ///
    /// # Errors
    /// Returns [`serde_json::Error`] if serialization fails.
    pub fn to_json_pretty(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }

    /// Deserializes a tuning profile from a JSON string.
    ///
    /// # Errors
    /// Returns [`serde_json::Error`] if deserialization fails.
    pub fn from_json_str(json: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(json)
    }
}

/// Unified model registry and runtime profile combining identity, tuning profile, and empirical cost profile.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelProfile {
    /// Logical model identifier.
    pub model_id: String,
    /// Active tuning profile (conservative default or operator-applied).
    pub tuning: TuningProfile,
    /// Optional empirical diagnostic cost profile if evaluated by k3-diag.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cost_profile: Option<CostProfile>,
}

impl ModelProfile {
    /// Creates a model profile initialized with the conservative unprofiled default.
    #[must_use]
    pub fn unprofiled_default(model_id: impl Into<String>) -> Self {
        let id = model_id.into();
        let mut tuning = TuningProfile::conservative_default();
        tuning.model_id = Some(id.clone());
        Self {
            model_id: id,
            tuning,
            cost_profile: None,
        }
    }

    /// Creates a model profile initialized with the conservative unprofiled default and custom `n_ctx`.
    #[must_use]
    pub fn unprofiled_with_n_ctx(model_id: impl Into<String>, n_ctx: usize) -> Self {
        let id = model_id.into();
        let mut tuning = TuningProfile::conservative_default_with_n_ctx(n_ctx);
        tuning.model_id = Some(id.clone());
        Self {
            model_id: id,
            tuning,
            cost_profile: None,
        }
    }

    /// Returns `true` if the model is currently unprofiled.
    #[must_use]
    pub fn is_unprofiled(&self) -> bool {
        self.tuning.is_unprofiled()
    }

    /// Transitions the model to operator-tuned state by applying an operator tuning profile (ADR-031, ADR-032).
    pub fn apply_tuning(&mut self, mut tuning: TuningProfile) {
        tuning.model_id = Some(self.model_id.clone());
        tuning.profile_status = ProfileStatus::Profiled;
        tuning.unprofiled = false;
        tuning.advisory_notice = None;
        self.tuning = tuning;
    }

    /// Attaches an empirical diagnostic cost profile from `k3-diag`.
    pub fn attach_cost_profile(&mut self, cost_profile: CostProfile) {
        self.cost_profile = Some(cost_profile);
    }

    /// Converts this model profile to an `OpenAI` `/v1/models`-compatible listing entry.
    #[must_use]
    pub fn to_model_entry(&self, created_timestamp: u64) -> ModelEntry {
        ModelEntry {
            id: self.model_id.clone(),
            object: "model".to_owned(),
            created: created_timestamp,
            owned_by: "k3".to_owned(),
            unprofiled: self.tuning.unprofiled,
            profile_status: self.tuning.profile_status.as_str().to_owned(),
            priority_class: self.tuning.priority_class.as_str().to_owned(),
            cluster: self.tuning.cluster.clone(),
            a100_cores: self.tuning.a100_cores,
            n_ctx: self.tuning.n_ctx,
            lend_idle_cores: self.tuning.lend_idle_cores,
            borrow_idle_cores: self.tuning.borrow_idle_cores,
            advisory_notice: self.tuning.advisory_notice.clone(),
        }
    }
}

/// Serializable model registry listing entry compatible with `OpenAI` `/v1/models` (ADR-031).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelEntry {
    /// Unique model identifier (e.g. "qwen2.5-7b").
    pub id: String,
    /// Object type, always "model".
    pub object: String,
    /// Unix creation timestamp in seconds.
    pub created: u64,
    /// Owner identifier ("k3").
    pub owned_by: String,
    /// Explicit boolean flag tagging unprofiled models (ADR-031).
    pub unprofiled: bool,
    /// Profile status string ("unprofiled" or "profiled").
    pub profile_status: String,
    /// Active priority class ("throughput" or "latency", ADR-014).
    pub priority_class: String,
    /// Processor cluster target ("x100", "a100", "both").
    pub cluster: String,
    /// Allocated A100 cores (0 for conservative defaults).
    pub a100_cores: usize,
    /// Fixed context window limit in tokens (ADR-027).
    pub n_ctx: usize,
    /// Whether idle core lending is enabled (ADR-030).
    pub lend_idle_cores: bool,
    /// Whether idle core borrowing is enabled (ADR-030).
    pub borrow_idle_cores: bool,
    /// Actionable advisory notice when unprofiled.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub advisory_notice: Option<String>,
}

/// Serializable `/v1/models` collection response (ADR-031).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelListResponse {
    /// Object type, always "list".
    pub object: String,
    /// List of model entries.
    pub data: Vec<ModelEntry>,
}

impl ModelListResponse {
    /// Creates a new model list collection response.
    #[must_use]
    pub fn new(data: Vec<ModelEntry>) -> Self {
        Self {
            object: "list".to_owned(),
            data,
        }
    }
}

/// Structured operational log / audit event emitted upon model load or status query (US-07 R2).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelLoadAuditEvent {
    /// Model identifier.
    pub model_id: String,
    /// Explicit unprofiled boolean flag.
    pub unprofiled: bool,
    /// Profile status ("unprofiled" or "profiled").
    pub profile_status: String,
    /// Processor cluster ("x100", "a100", "both").
    pub cluster: String,
    /// Number of A100 cores allocated.
    pub a100_cores: usize,
    /// Priority class ("throughput" or "latency").
    pub priority_class: String,
    /// Context window limit in tokens.
    pub n_ctx: usize,
    /// Actionable advisory notice if unprofiled.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub advisory_notice: Option<String>,
}

impl ModelLoadAuditEvent {
    /// Constructs an audit event from a [`TuningProfile`].
    #[must_use]
    pub fn from_tuning(model_id: impl Into<String>, tuning: &TuningProfile) -> Self {
        Self {
            model_id: model_id.into(),
            unprofiled: tuning.unprofiled,
            profile_status: tuning.profile_status.as_str().to_owned(),
            cluster: tuning.cluster.clone(),
            a100_cores: tuning.a100_cores,
            priority_class: tuning.priority_class.as_str().to_owned(),
            n_ctx: tuning.n_ctx,
            advisory_notice: tuning.advisory_notice.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::QuantType;
    use crate::gguf::{GgufFile, test_helpers::create_synthetic_gguf_v3};

    #[test]
    fn test_quantization_detection_multi_quant() {
        let tensor_specs = [
            ("blk.0.attn_q.weight", QuantType::Q4_K_M, &[256, 2][..]),
            ("output_norm.weight", QuantType::F32, &[16][..]),
        ];
        let bytes = create_synthetic_gguf_v3("synthetic", 32, &tensor_specs);
        let gguf = GgufFile::parse(&bytes).expect("parse synthetic GGUF");

        let q_info = QuantizationInfo::from_descriptors(gguf.metadata().tensors());
        assert_eq!(q_info.primary, "Q4_K_M");
        assert_eq!(q_info.summary.get("Q4_K_M"), Some(&1));
        assert_eq!(q_info.summary.get("F32"), Some(&1));
        assert_eq!(q_info.tensor_bytes.get("Q4_K_M"), Some(&288));
        assert_eq!(q_info.tensor_bytes.get("F32"), Some(&64));
    }

    #[test]
    fn test_quantization_detection_uniform_f32() {
        let tensor_specs = [
            ("output_norm.weight", QuantType::F32, &[16][..]),
            ("blk.0.attn_norm.weight", QuantType::F32, &[16][..]),
        ];
        let bytes = create_synthetic_gguf_v3("synthetic", 32, &tensor_specs);
        let gguf = GgufFile::parse(&bytes).expect("parse synthetic GGUF");

        let q_info = QuantizationInfo::from_descriptors(gguf.metadata().tensors());
        assert_eq!(q_info.primary, "F32");
        assert_eq!(q_info.summary.get("F32"), Some(&2));
    }

    #[test]
    fn test_quantization_detection_q8_0() {
        let tensor_specs = [
            ("blk.0.attn_q.weight", QuantType::Q8_0, &[32, 2][..]),
            ("output_norm.weight", QuantType::F32, &[8][..]),
        ];
        let bytes = create_synthetic_gguf_v3("synthetic", 32, &tensor_specs);
        let gguf = GgufFile::parse(&bytes).expect("parse synthetic GGUF");

        let q_info = QuantizationInfo::from_descriptors(gguf.metadata().tensors());
        assert_eq!(q_info.primary, "Q8_0");
        assert_eq!(q_info.summary.get("Q8_0"), Some(&1));
        assert_eq!(q_info.summary.get("F32"), Some(&1));
        assert_eq!(q_info.tensor_bytes.get("Q8_0"), Some(&68));
        assert_eq!(q_info.tensor_bytes.get("F32"), Some(&32));
    }

    #[test]
    fn test_cost_profile_serialization_roundtrip_and_invariants() {
        let profile = CostProfile {
            model_identity: ModelIdentity {
                name: "test-model".to_owned(),
                architecture: "synthetic".to_owned(),
                file_path: "/path/to/test.gguf".to_owned(),
                file_digest: "0123456789abcdef".to_owned(),
            },
            quantization_level: QuantizationInfo {
                primary: "Q4_K_M".to_owned(),
                summary: [("Q4_K_M".to_owned(), 1), ("F32".to_owned(), 1)]
                    .into_iter()
                    .collect(),
                tensor_bytes: [("Q4_K_M".to_owned(), 288), ("F32".to_owned(), 64)]
                    .into_iter()
                    .collect(),
            },
            cluster_placement: ClusterPlacement {
                cluster: "both".to_owned(),
                effective_core_mask: "0xFFFF".to_owned(),
                allocated_cores: 16,
                x100_cores: 8,
                a100_cores: 8,
                coordination_profile: "spin".to_owned(),
                split_strategy: "phase".to_owned(),
                ime: "on".to_owned(),
            },
            allocated_cores: CoreAllocation {
                total: 16,
                total_cores: 16,
                x100_cores: 8,
                a100_cores: 8,
                effective_mask: "0xFFFF".to_owned(),
            },
            split_strategy: "phase".to_owned(),
            execution_cost: ExecutionCost {
                prefill_cost_per_token_ms: 0.15,
                prefill_tokens_per_sec: 6666.6,
                prefill_hardware_ceiling_pct: 42.5,
                decode_cost_per_token_ms: 2.5,
                decode_tokens_per_sec: 400.0,
                decode_hardware_ceiling_pct: 55.0,
                static_scheduling_weight: 2.5,
            },
            prefill_cost_per_token_ms: 0.15,
            prefill_tokens_per_sec: 6666.6,
            prefill_hardware_ceiling_pct: 42.5,
            decode_tokens_per_sec: 400.0,
            decode_cost_per_token_ms: 2.5,
            decode_hardware_ceiling_pct: 55.0,
            static_scheduling_weight: 2.5,
            scheduling_weight: 2.5,
            memory_footprint: MemoryFootprintProfile {
                weight_bytes: 4096,
                kv_bytes: 4096,
                total_required_bytes: 8192,
                peak_resident_bytes: 16384,
            },
            weight_bytes: 4096,
            kv_bytes: 4096,
            total_required_bytes: 8192,
            peak_resident_bytes: 16384,
            tuning_advisory: TuningAdvisory {
                bottleneck_classification: "bandwidth-bound".to_owned(),
                recommended_cluster: "both".to_owned(),
                recommended_split: Some("phase".to_owned()),
                recommended_ime: true,
                rationale: "Decode phase is memory bandwidth bound".to_owned(),
                recommendations: vec!["Keep phase split".to_owned()],
                operator_authority_notice: "Advisory only per ADR-032".to_owned(),
            },
        };

        // Assert memory footprint invariant: peak_resident_bytes >= total_required_bytes >= weight_bytes + kv_bytes
        assert!(profile.peak_resident_bytes >= profile.total_required_bytes);
        assert_eq!(
            profile.total_required_bytes,
            profile.weight_bytes + profile.kv_bytes
        );

        let json = profile.to_json_pretty().expect("serialize profile");
        assert!(json.contains("\"prefill_cost_per_token_ms\": 0.15"));
        assert!(json.contains("\"primary\": \"Q4_K_M\""));
        assert!(json.contains("\"split_strategy\": \"phase\""));

        let deserialized = CostProfile::from_json_str(&json).expect("deserialize profile");
        assert_eq!(profile, deserialized);
    }

    #[test]
    fn test_priority_class_parsing_and_invariants() {
        assert_eq!(PriorityClass::default(), PriorityClass::Throughput);
        assert_eq!(PriorityClass::Throughput.as_str(), "throughput");
        assert_eq!(PriorityClass::Latency.as_str(), "latency");
        assert_eq!(PriorityClass::Throughput.to_string(), "throughput");
        assert_eq!(PriorityClass::Latency.to_string(), "latency");

        assert_eq!(
            PriorityClass::parse("throughput").unwrap(),
            PriorityClass::Throughput
        );
        assert_eq!(
            PriorityClass::parse("THROUGHPUT").unwrap(),
            PriorityClass::Throughput
        );
        assert_eq!(
            PriorityClass::parse("latency").unwrap(),
            PriorityClass::Latency
        );
        assert_eq!(
            PriorityClass::parse("LATENCY").unwrap(),
            PriorityClass::Latency
        );
        assert!(PriorityClass::parse("invalid").is_err());

        // Serde roundtrip
        let json = serde_json::to_string(&PriorityClass::Latency).unwrap();
        assert_eq!(json, "\"latency\"");
        let parsed: PriorityClass = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, PriorityClass::Latency);
    }

    #[test]
    fn test_compute_pool_parsing_and_invariants() {
        assert_eq!(ComputePool::default(), ComputePool::X100);
        assert_eq!(ComputePool::X100.as_str(), "x100");
        assert_eq!(ComputePool::A100.as_str(), "a100");
        assert_eq!(ComputePool::Both.as_str(), "both");
        assert_eq!(ComputePool::X100.to_string(), "x100");

        assert_eq!(
            ComputePool::X100.core_mask(),
            crate::affinity::X100_CLUSTER_MASK
        );
        assert_eq!(
            ComputePool::A100.core_mask(),
            crate::affinity::A100_CLUSTER_MASK
        );
        assert_eq!(
            ComputePool::Both.core_mask(),
            crate::affinity::SOC_CLUSTER_MASK
        );

        assert_eq!(ComputePool::X100.default_max_compute_cores(), 6);
        assert_eq!(ComputePool::A100.default_max_compute_cores(), 8);
        assert_eq!(ComputePool::Both.default_max_compute_cores(), 14);

        assert_eq!(ComputePool::parse("x100").unwrap(), ComputePool::X100);
        assert_eq!(ComputePool::parse("A100").unwrap(), ComputePool::A100);
        assert_eq!(ComputePool::parse("both").unwrap(), ComputePool::Both);
        assert!(ComputePool::parse("unknown").is_err());

        // Serde roundtrip
        let json = serde_json::to_string(&ComputePool::Both).unwrap();
        assert_eq!(json, "\"both\"");
        let parsed: ComputePool = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, ComputePool::Both);
    }

    #[test]
    fn test_profile_status_parsing_and_invariants() {
        assert_eq!(ProfileStatus::default(), ProfileStatus::Unprofiled);
        assert!(ProfileStatus::Unprofiled.is_unprofiled());
        assert!(!ProfileStatus::Profiled.is_unprofiled());
        assert_eq!(ProfileStatus::Unprofiled.as_str(), "unprofiled");
        assert_eq!(ProfileStatus::Profiled.as_str(), "profiled");

        assert_eq!(
            ProfileStatus::parse("unprofiled").unwrap(),
            ProfileStatus::Unprofiled
        );
        assert_eq!(
            ProfileStatus::parse("profiled").unwrap(),
            ProfileStatus::Profiled
        );
        assert!(ProfileStatus::parse("invalid").is_err());

        let json = serde_json::to_string(&ProfileStatus::Unprofiled).unwrap();
        assert_eq!(json, "\"unprofiled\"");
    }

    #[test]
    fn test_conservative_default_invariants() {
        let profile = TuningProfile::conservative_default();
        assert_eq!(profile.compute_pool, ComputePool::X100);
        assert_eq!(profile.cluster, "x100");
        assert_eq!(profile.x100_cores, CONSERVATIVE_DEFAULT_X100_CORES);
        assert_eq!(profile.x100_cores, 6);
        assert_eq!(profile.a100_cores, CONSERVATIVE_DEFAULT_A100_CORES);
        assert_eq!(profile.a100_cores, 0);
        assert_eq!(profile.priority_class, PriorityClass::Throughput);
        assert_eq!(profile.n_ctx, CONSERVATIVE_DEFAULT_N_CTX);
        assert_eq!(profile.n_ctx, 2048);
        assert!(!profile.lend_idle_cores);
        assert!(!profile.borrow_idle_cores);
        assert_eq!(profile.profile_status, ProfileStatus::Unprofiled);
        assert!(profile.unprofiled);
        assert!(profile.is_unprofiled());
        assert_eq!(
            profile.advisory_notice.as_deref(),
            Some(CONSERVATIVE_TUNING_ADVISORY)
        );

        // Test Default trait
        assert_eq!(TuningProfile::default(), profile);
    }

    #[test]
    fn test_conservative_default_with_constrained_n_ctx() {
        let profile = TuningProfile::conservative_default_with_n_ctx(CONSTRAINED_TEST_N_CTX);
        assert_eq!(profile.n_ctx, 64);
        assert_eq!(profile.compute_pool, ComputePool::X100);
        assert_eq!(profile.x100_cores, 6);
        assert_eq!(profile.a100_cores, 0);
        assert_eq!(profile.priority_class, PriorityClass::Throughput);
        assert!(!profile.lend_idle_cores);
        assert!(!profile.borrow_idle_cores);
        assert!(profile.unprofiled);
        assert_eq!(
            profile.advisory_notice.as_deref(),
            Some(CONSERVATIVE_TUNING_ADVISORY)
        );
    }

    #[test]
    fn test_tuning_profile_operator_transition() {
        let mut profile = TuningProfile::conservative_default();
        assert!(profile.is_unprofiled());
        assert!(profile.advisory_notice.is_some());

        profile.apply_operator_tuning(
            ComputePool::Both,
            4,
            8,
            PriorityClass::Latency,
            4096,
            true,
            false,
        );

        assert!(!profile.is_unprofiled());
        assert!(!profile.unprofiled);
        assert_eq!(profile.profile_status, ProfileStatus::Profiled);
        assert_eq!(profile.compute_pool, ComputePool::Both);
        assert_eq!(profile.cluster, "both");
        assert_eq!(profile.x100_cores, 4);
        assert_eq!(profile.a100_cores, 8);
        assert_eq!(profile.priority_class, PriorityClass::Latency);
        assert_eq!(profile.n_ctx, 4096);
        assert!(profile.lend_idle_cores);
        assert!(!profile.borrow_idle_cores);
        assert!(profile.advisory_notice.is_none());

        // Serde roundtrip for tuned profile
        let json = profile.to_json_pretty().unwrap();
        assert!(json.contains("\"cluster\": \"both\""));
        assert!(json.contains("\"profile_status\": \"profiled\""));
        assert!(json.contains("\"unprofiled\": false"));
        assert!(!json.contains("advisory_notice"));

        let deserialized = TuningProfile::from_json_str(&json).unwrap();
        assert_eq!(profile, deserialized);
    }

    #[test]
    fn test_model_profile_and_openai_v1_models_serialization() {
        let model_prof = ModelProfile::unprofiled_default("qwen2.5-7b");
        assert_eq!(model_prof.model_id, "qwen2.5-7b");
        assert!(model_prof.is_unprofiled());
        assert_eq!(model_prof.tuning.n_ctx, CONSERVATIVE_DEFAULT_N_CTX);

        let created_ts = 1_700_000_000_u64;
        let entry = model_prof.to_model_entry(created_ts);
        assert_eq!(entry.id, "qwen2.5-7b");
        assert_eq!(entry.object, "model");
        assert_eq!(entry.created, created_ts);
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

        let entry_json = serde_json::to_string_pretty(&entry).unwrap();
        assert!(entry_json.contains("\"id\": \"qwen2.5-7b\""));
        assert!(entry_json.contains("\"unprofiled\": true"));
        assert!(entry_json.contains("\"profile_status\": \"unprofiled\""));
        assert!(entry_json.contains("\"priority_class\": \"throughput\""));
        assert!(entry_json.contains("\"a100_cores\": 0"));
        assert!(entry_json.contains("\"n_ctx\": 2048"));
        assert!(entry_json.contains(CONSERVATIVE_TUNING_ADVISORY));

        let list_resp = ModelListResponse::new(vec![entry]);
        let list_json = serde_json::to_string_pretty(&list_resp).unwrap();
        assert!(list_json.contains("\"object\": \"list\""));
        assert!(list_json.contains("\"id\": \"qwen2.5-7b\""));

        // Constrained test n_ctx model profile
        let constrained = ModelProfile::unprofiled_with_n_ctx("llama-3-8b", 64);
        assert_eq!(constrained.tuning.n_ctx, 64);
        assert!(constrained.is_unprofiled());
    }

    #[test]
    fn test_model_profile_apply_tuning_and_attach_cost() {
        let mut model_prof = ModelProfile::unprofiled_default("qwen2.5-7b");
        let mut new_tuning = TuningProfile::conservative_default();
        new_tuning.apply_operator_tuning(
            ComputePool::A100,
            0,
            8,
            PriorityClass::Latency,
            8192,
            false,
            false,
        );

        model_prof.apply_tuning(new_tuning);
        assert!(!model_prof.is_unprofiled());
        assert_eq!(model_prof.tuning.compute_pool, ComputePool::A100);
        assert_eq!(model_prof.tuning.a100_cores, 8);
        assert_eq!(model_prof.tuning.priority_class, PriorityClass::Latency);
        assert_eq!(model_prof.tuning.n_ctx, 8192);

        let entry = model_prof.to_model_entry(1_700_000_000);
        assert!(!entry.unprofiled);
        assert_eq!(entry.profile_status, "profiled");
        assert_eq!(entry.priority_class, "latency");
        assert_eq!(entry.cluster, "a100");
        assert_eq!(entry.a100_cores, 8);
        assert!(entry.advisory_notice.is_none());

        // Attach cost profile
        let identity = ModelIdentity {
            name: "qwen2.5-7b".to_owned(),
            architecture: "qwen2".to_owned(),
            file_path: "/tmp/model.gguf".to_owned(),
            file_digest: "abcd".to_owned(),
        };
        let cost = CostProfile::unprofiled_default(identity);
        model_prof.attach_cost_profile(cost);
        assert!(model_prof.cost_profile.is_some());
    }

    #[test]
    fn test_model_load_audit_event_generation() {
        let tuning = TuningProfile::conservative_default();
        let event = ModelLoadAuditEvent::from_tuning("qwen2.5-7b", &tuning);

        assert_eq!(event.model_id, "qwen2.5-7b");
        assert!(event.unprofiled);
        assert_eq!(event.profile_status, "unprofiled");
        assert_eq!(event.cluster, "x100");
        assert_eq!(event.a100_cores, 0);
        assert_eq!(event.priority_class, "throughput");
        assert_eq!(event.n_ctx, 2048);
        assert_eq!(
            event.advisory_notice.as_deref(),
            Some(CONSERVATIVE_TUNING_ADVISORY)
        );

        let event_json = serde_json::to_string(&event).unwrap();
        assert!(event_json.contains("\"unprofiled\":true"));
        assert!(event_json.contains("\"model_id\":\"qwen2.5-7b\""));
    }
}
