//! Structured cost profiling, hardware-ceiling efficiency metrics, and tuning advisories.
//!
//! Provides the canonical machine-readable [`CostProfile`] baseline schema for static scheduling
//! weights and operator tuning advisories (US-04, ADR-010, ADR-032, ADR-034).

use std::collections::BTreeMap;
use serde::{Deserialize, Serialize};

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
}
