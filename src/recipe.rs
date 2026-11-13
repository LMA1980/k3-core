//! Single-prompt execution context and architecture recipe traits (ADR-005, ADR-023, ADR-024).

use crate::error::K3Error;
use crate::tensor::{ModelWeights, QuantType};

/// Single-prompt execution context enforcing strict cycle exclusivity (no cross-prompt batching
/// per ADR-005) and fixed `n_ctx` context window bounds (per ADR-027).
#[derive(Debug)]
pub struct ExecutionContext<'a> {
    prompt_tokens: &'a [u32],
    decode_token: Option<u32>,
    jump_forward_tokens: Option<&'a [u32]>,
    position: usize,
    n_ctx: usize,
    logits: &'a mut [f32],
    kv_cache: &'a mut [f32],
}

impl<'a> ExecutionContext<'a> {
    /// Creates a new single-prompt [`ExecutionContext`] validated against `n_ctx`.
    ///
    /// # Errors
    /// Returns [`K3Error::ContextLengthExceeded`] if `prompt_tokens.len() > n_ctx` or `n_ctx == 0`.
    pub fn new(
        prompt_tokens: &'a [u32],
        n_ctx: usize,
        logits: &'a mut [f32],
        kv_cache: &'a mut [f32],
    ) -> Result<Self, K3Error> {
        if n_ctx == 0 || prompt_tokens.len() > n_ctx {
            return Err(K3Error::ContextLengthExceeded {
                requested_tokens: prompt_tokens.len(),
                max_context_tokens: n_ctx,
            });
        }

        Ok(Self {
            prompt_tokens,
            decode_token: None,
            jump_forward_tokens: None,
            position: 0,
            n_ctx,
            logits,
            kv_cache,
        })
    }

    /// Sets the starting KV sequence position cursor for this [`ExecutionContext`].
    ///
    /// # Errors
    /// Returns [`K3Error::ContextLengthExceeded`] if `position + self.prompt_tokens.len() > self.n_ctx`.
    pub fn with_position(mut self, position: usize) -> Result<Self, K3Error> {
        let end_pos = position.checked_add(self.prompt_tokens.len()).ok_or(
            K3Error::ContextLengthExceeded {
                requested_tokens: usize::MAX,
                max_context_tokens: self.n_ctx,
            },
        )?;
        if end_pos > self.n_ctx {
            return Err(K3Error::ContextLengthExceeded {
                requested_tokens: end_pos,
                max_context_tokens: self.n_ctx,
            });
        }
        self.position = position;
        Ok(self)
    }

    /// Returns the input prompt token sequence for the single active evaluation slot.
    #[must_use]
    pub const fn prompt_tokens(&self) -> &'a [u32] {
        self.prompt_tokens
    }

    /// Returns the current token ID scheduled for the next single-token [`ArchRecipe::decode_step`].
    #[must_use]
    pub const fn decode_token(&self) -> Option<u32> {
        self.decode_token
    }

    /// Returns the current KV sequence position cursor.
    #[must_use]
    pub const fn position(&self) -> usize {
        self.position
    }

    /// Returns the fixed maximum context length (`n_ctx`) for this execution slot.
    #[must_use]
    pub const fn n_ctx(&self) -> usize {
        self.n_ctx
    }

    /// Returns an immutable view of the output logits buffer.
    #[must_use]
    pub const fn logits(&self) -> &[f32] {
        self.logits
    }

    /// Returns a mutable view of the output logits buffer.
    pub const fn logits_mut(&mut self) -> &mut [f32] {
        self.logits
    }

    /// Returns an immutable view of the single-slot KV cache buffer.
    #[must_use]
    pub const fn kv_cache(&self) -> &[f32] {
        self.kv_cache
    }

    /// Returns a mutable view of the single-slot KV cache buffer.
    pub const fn kv_cache_mut(&mut self) -> &mut [f32] {
        self.kv_cache
    }

    /// Returns the staged forced token sequence for a jump-forward prefill burst, if any.
    #[must_use]
    pub const fn jump_forward_tokens(&self) -> Option<&'a [u32]> {
        self.jump_forward_tokens
    }

    /// Returns simultaneous mutable views of the logits and KV cache buffers.
    pub fn buffers_mut(&mut self) -> (&mut [f32], &mut [f32]) {
        (self.logits, self.kv_cache)
    }

    /// Stages the next generated token for a single-token decode pass, verifying `position < n_ctx`.
    ///
    /// # Errors
    /// Returns [`K3Error::ContextLengthExceeded`] if `self.position + 1 > self.n_ctx`.
    pub fn prepare_decode(&mut self, token_id: u32) -> Result<(), K3Error> {
        let next_pos = self.position.saturating_add(1);
        if next_pos > self.n_ctx {
            return Err(K3Error::ContextLengthExceeded {
                requested_tokens: next_pos,
                max_context_tokens: self.n_ctx,
            });
        }
        self.jump_forward_tokens = None;
        self.decode_token = Some(token_id);
        Ok(())
    }

    /// Stages a deterministic `forced_tokens` slice for an atomic jump-forward prefill burst.
    ///
    /// # Errors
    /// Returns [`K3Error::ContextLengthExceeded`] if `self.position + forced_tokens.len() > self.n_ctx`.
    pub fn prepare_jump_forward(&mut self, forced_tokens: &'a [u32]) -> Result<(), K3Error> {
        let next_pos = self.position.checked_add(forced_tokens.len()).ok_or(
            K3Error::ContextLengthExceeded {
                requested_tokens: usize::MAX,
                max_context_tokens: self.n_ctx,
            },
        )?;
        if next_pos > self.n_ctx {
            return Err(K3Error::ContextLengthExceeded {
                requested_tokens: next_pos,
                max_context_tokens: self.n_ctx,
            });
        }
        self.decode_token = None;
        self.jump_forward_tokens = Some(forced_tokens);
        Ok(())
    }

    /// Clears any staged jump-forward forced token sequence.
    pub fn clear_jump_forward(&mut self) {
        self.jump_forward_tokens = None;
    }

    /// Advances the KV sequence position cursor by `token_count` tokens after a forward step.
    ///
    /// # Errors
    /// Returns [`K3Error::ContextLengthExceeded`] if `self.position + token_count > self.n_ctx`.
    pub fn advance_position(&mut self, token_count: usize) -> Result<usize, K3Error> {
        let new_pos =
            self.position
                .checked_add(token_count)
                .ok_or(K3Error::ContextLengthExceeded {
                    requested_tokens: usize::MAX,
                    max_context_tokens: self.n_ctx,
                })?;
        if new_pos > self.n_ctx {
            return Err(K3Error::ContextLengthExceeded {
                requested_tokens: new_pos,
                max_context_tokens: self.n_ctx,
            });
        }
        self.position = new_pos;
        Ok(self.position)
    }
}

/// Architecture recipe trait compiled in-process per ADR-023 and ADR-024.
///
/// Each recipe maps a specific `general.architecture` identifier (treating minor versions
/// such as `qwen3` vs. `qwen3.8` as distinct algorithms) to single-prompt prefill and decode steps.
pub trait ArchRecipe: Send + Sync + core::fmt::Debug {
    /// Returns the exact GGUF `general.architecture` identifier handled by this recipe.
    fn architecture_id(&self) -> &'static str;

    /// Returns the exact recipe revision string (e.g., `"1.0.0"`) per ADR-023.
    fn recipe_version(&self) -> &'static str {
        "1.0.0"
    }

    /// Returns the supported target weight layouts (e.g., `&["native", "ime_tile_8x16"]`) per ADR-037.
    fn target_layouts(&self) -> &'static [&'static str] {
        &["native"]
    }

    /// Returns `true` if this recipe maintains an autoregressive KV cache across decode steps.
    ///
    /// Zero-KV encoder architectures (such as `ModernBERT`, `DeBERTa-v3`, and `bert`) override
    /// this to return `false` so admission and arena setup skip KV page reservation.
    fn requires_kv_cache(&self) -> bool {
        true
    }

    /// Returns `true` if this recipe and its linked kernel backend support `quant`.
    fn supports_quant(&self, quant: QuantType) -> bool;

    /// Executes a single-pass classification or logit-harvest pass over [`ExecutionContext::prompt_tokens`].
    ///
    /// Defaults to delegating to [`ArchRecipe::prefill_step`]. Encoder-only recipes override this
    /// to compute bidirectional embeddings, pooling, and classification logits without a KV cache.
    ///
    /// # Errors
    /// Returns [`K3Error`] if weight lookup, buffer validation, or forward execution fails.
    fn forward_classify(
        &self,
        ctx: &mut ExecutionContext<'_>,
        weights: &dyn ModelWeights,
    ) -> Result<(), K3Error> {
        self.prefill_step(ctx, weights)
    }

    /// Executes the prompt prefill forward pass over [`ExecutionContext::prompt_tokens`]
    /// for a single prompt (ADR-005), populating the KV cache and writing next-token logits.
    ///
    /// # Errors
    /// Returns [`K3Error`] if a required tensor is missing, has an unsupported quantization
    /// layout, or exceeds context window limits.
    fn prefill_step(
        &self,
        ctx: &mut ExecutionContext<'_>,
        weights: &dyn ModelWeights,
    ) -> Result<(), K3Error>;

    /// Executes a single-token autoregressive decode step using [`ExecutionContext::decode_token`],
    /// updating the KV cache at [`ExecutionContext::position`] and writing next-token logits.
    ///
    /// # Errors
    /// Returns [`K3Error`] if no decode token is staged, context bounds are exceeded, or weight
    /// lookup fails.
    fn decode_step(
        &self,
        ctx: &mut ExecutionContext<'_>,
        weights: &dyn ModelWeights,
    ) -> Result<(), K3Error>;
}
