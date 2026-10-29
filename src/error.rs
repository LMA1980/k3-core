//! Structured diagnostic error types for `k3-core` and `k3-engine`.

use core::fmt;

/// Structured diagnostic error covering capability mismatches, lockable memory admission,
/// context window bounds, CPU core affinity, and zero-copy tensor view bounds.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum K3Error {
    /// Model architecture or quantization format is not supported by compiled recipes/kernels (ADR-023).
    CapabilityMismatch {
        /// The model architecture identifier (e.g. from GGUF `general.architecture`).
        architecture: String,
        /// Detailed explanation of the unsupported capability or quantization format.
        reason: String,
    },
    /// Required model weight + KV cache footprint exceeds lockable physical memory (ADR-037, US-01 Scenario 3).
    InsufficientLockableMemory {
        /// Total page-aligned bytes required (`weight_bytes + kv_bytes`).
        required_bytes: u64,
        /// Page-aligned bytes required for `mlock`ed model weights.
        weight_bytes: u64,
        /// Page-aligned bytes required for the `mlock`ed single-slot KV cache arena.
        kv_bytes: u64,
        /// Currently available physical memory bytes (`MemAvailable` from `/proc/meminfo`).
        available_bytes: u64,
        /// Process lockable memory limit bytes (`RLIMIT_MEMLOCK`).
        rlimit_memlock_bytes: u64,
    },
    /// Prompt or decode sequence position exceeds the fixed `n_ctx` placement bound (ADR-027).
    ContextLengthExceeded {
        /// Total tokens requested or reached by the sequence.
        requested_tokens: usize,
        /// Fixed maximum context window (`n_ctx`) configured for the placement.
        max_context_tokens: usize,
    },
    /// Invalid compute cluster mask, thread count, or cross-cluster VLEN domain configuration (US-01 Scenario 2).
    AffinityError {
        /// Requested CPU core bitmask.
        requested_mask: u64,
        /// Requested worker thread count.
        requested_threads: usize,
        /// Diagnostic description of the affinity violation.
        reason: String,
    },
    /// Zero-copy tensor view failed shape, block alignment, or buffer bounds validation.
    TensorBoundsError {
        /// Name of the tensor being resolved.
        tensor_name: String,
        /// Starting byte offset within the backing buffer.
        offset: usize,
        /// Computed byte length of the tensor payload.
        byte_len: usize,
        /// Total byte length of the backing buffer.
        buffer_len: usize,
        /// Specific reason the bounds or layout check failed.
        reason: String,
    },
    /// Automatic request routing unavailable (US-12, ADR-013).
    RoutingUnavailable {
        /// Diagnostic explanation of the routing failure.
        reason: String,
    },
    /// Model admission queue capacity reached or estimated wait exceeded (US-13, ADR-015).
    QueueSaturated {
        /// Logical model identifier.
        model_id: String,
        /// Current depth of the admission queue (number of pending requests).
        current_depth: usize,
        /// Configured maximum queue capacity ceiling (default 64).
        max_depth: usize,
        /// Recommended client retry interval in seconds.
        retry_after_secs: u32,
    },
}

impl K3Error {
    /// Returns the machine-readable diagnostic error code for structured reporting.
    #[must_use]
    pub const fn diagnostic_code(&self) -> &'static str {
        match self {
            Self::CapabilityMismatch { .. } => "capability_mismatch",
            Self::InsufficientLockableMemory { .. } => "insufficient_lockable_memory",
            Self::ContextLengthExceeded { .. } => "context_length_exceeded",
            Self::AffinityError { .. } => "affinity_error",
            Self::TensorBoundsError { .. } => "tensor_bounds_error",
            Self::RoutingUnavailable { .. } => "routing_unavailable",
            Self::QueueSaturated { .. } => "queue_saturated",
        }
    }
}

impl fmt::Display for K3Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::CapabilityMismatch {
                architecture,
                reason,
            } => write!(
                f,
                "[capability_mismatch] unsupported architecture '{architecture}': {reason}"
            ),
            Self::InsufficientLockableMemory {
                required_bytes,
                weight_bytes,
                kv_bytes,
                available_bytes,
                rlimit_memlock_bytes,
            } => write!(
                f,
                "[insufficient_lockable_memory] required {required_bytes} bytes \
                 (weights: {weight_bytes} bytes, kv: {kv_bytes} bytes) exceeds lockable capacity \
                 (available RAM: {available_bytes} bytes, RLIMIT_MEMLOCK: {rlimit_memlock_bytes} bytes)"
            ),
            Self::ContextLengthExceeded {
                requested_tokens,
                max_context_tokens,
            } => write!(
                f,
                "[context_length_exceeded] sequence length {requested_tokens} tokens \
                 exceeds configured n_ctx limit of {max_context_tokens} tokens"
            ),
            Self::AffinityError {
                requested_mask,
                requested_threads,
                reason,
            } => write!(
                f,
                "[affinity_error] invalid affinity configuration (mask: 0x{requested_mask:04X}, \
                 threads: {requested_threads}): {reason}"
            ),
            Self::TensorBoundsError {
                tensor_name,
                offset,
                byte_len,
                buffer_len,
                reason,
            } => write!(
                f,
                "[tensor_bounds_error] tensor '{tensor_name}' (offset: {offset}, \
                 len: {byte_len}, buffer_len: {buffer_len}): {reason}"
            ),
            Self::RoutingUnavailable { reason } => {
                write!(f, "[routing_unavailable] routing unavailable: {reason}")
            }
            Self::QueueSaturated {
                model_id,
                current_depth,
                max_depth,
                retry_after_secs,
            } => write!(
                f,
                "[queue_saturated] model '{model_id}' admission queue is saturated (depth: {current_depth}/{max_depth}); retry after {retry_after_secs}s"
            ),
        }
    }
}

impl std::error::Error for K3Error {}
