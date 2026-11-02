//! Canonical model manifest describing model builds and capabilities.

use serde::{Deserialize, Serialize};

/// Canonical metadata manifest describing a registered model build (US-18, ADR-035).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelManifest {
    /// Manifest schema format version (default 1).
    pub schema_version: u32,
    /// Model format: "gguf", "safetensors", etc.
    pub format: String,
    /// Architecture identifier (e.g., "synthetic", "qwen3", "llama", "mistral", "mixtral").
    pub architecture_id: String,
    /// Architecture recipe version (e.g., "1.0.0").
    pub recipe_version: String,
    /// Quantization format (e.g., `"Q4_K_M"`, `"Q4_0"`, `"Q8_0"`, `"F32"`).
    pub quantization: String,
    /// Context length trained into weights.
    pub context_length: u64,
    /// Optional license string.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub license: Option<String>,
    /// Optional provenance or download URL.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub origin: Option<String>,
    /// Cryptographic SHA-256 digest of underlying model weights blob.
    pub weight_digest: String,
    /// Size of weights blob in bytes.
    pub weight_size_bytes: u64,
    /// Manifest creation timestamp in epoch seconds.
    pub created_at: u64,
}

impl ModelManifest {
    /// Creates a new model manifest with default schema version 1.
    #[must_use]
    pub fn new(
        format: impl Into<String>,
        architecture_id: impl Into<String>,
        recipe_version: impl Into<String>,
        quantization: impl Into<String>,
        context_length: u64,
        weight_digest: impl Into<String>,
        weight_size_bytes: u64,
    ) -> Self {
        Self {
            schema_version: 1,
            format: format.into(),
            architecture_id: architecture_id.into(),
            recipe_version: recipe_version.into(),
            quantization: quantization.into(),
            context_length,
            license: None,
            origin: None,
            weight_digest: weight_digest.into(),
            weight_size_bytes,
            created_at: 0,
        }
    }

    /// Serializes this manifest to a canonical JSON string.
    ///
    /// # Errors
    /// Returns [`serde_json::Error`] if serialization fails.
    pub fn to_canonical_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }
}
