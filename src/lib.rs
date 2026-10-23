//! `k3-core`: Foundational traits, zero-copy tensor views, single-prompt execution context,
//! and structured diagnostic errors for the `SpacemiT` K3 AI runtime.

pub mod coordination;
pub mod error;
pub mod gguf;
pub mod recipe;
pub mod tensor;

pub use coordination::CoordinationProfile;
pub use error::K3Error;
#[doc(hidden)]
pub use gguf::test_helpers;
pub use gguf::{
    DEFAULT_ALIGNMENT, GGUF_MAGIC, GGUF_VERSION_V3, GgufArray, GgufFile, GgufMetadataValueType,
    GgufValue, ModelMetadata, TensorDescriptor, align_up,
};
pub use recipe::{ArchRecipe, ExecutionContext};
pub use tensor::{ModelWeights, QuantType, TensorView};

#[cfg(test)]
mod tests {
    use super::{ArchRecipe, ExecutionContext, K3Error, ModelWeights, QuantType, TensorView};

    struct DummyWeights<'a> {
        buffer: &'a [u8],
    }

    impl ModelWeights for DummyWeights<'_> {
        fn tensor(&self, name: &str) -> Result<TensorView<'_>, K3Error> {
            match name {
                "blk.0.attn_q.weight" => TensorView::from_bytes(
                    "blk.0.attn_q.weight",
                    QuantType::Q4_K_M,
                    &[256, 2],
                    0,
                    self.buffer,
                ),
                "output_norm.weight" => TensorView::from_bytes(
                    "output_norm.weight",
                    QuantType::F32,
                    &[16],
                    288,
                    self.buffer,
                ),
                _ => Err(K3Error::TensorBoundsError {
                    tensor_name: name.to_owned(),
                    offset: 0,
                    byte_len: 0,
                    buffer_len: self.buffer.len(),
                    reason: "tensor not found".to_owned(),
                }),
            }
        }

        fn tensor_count(&self) -> usize {
            2
        }

        fn total_weight_bytes(&self) -> usize {
            self.buffer.len()
        }
    }

    #[derive(Debug)]
    struct DummyRecipe;

    impl ArchRecipe for DummyRecipe {
        fn architecture_id(&self) -> &'static str {
            "synthetic"
        }

        fn supports_quant(&self, quant: QuantType) -> bool {
            matches!(quant, QuantType::F32 | QuantType::F16 | QuantType::Q4_K_M)
        }

        fn prefill_step(
            &self,
            ctx: &mut ExecutionContext<'_>,
            weights: &dyn ModelWeights,
        ) -> Result<(), K3Error> {
            let q = weights.tensor("blk.0.attn_q.weight")?;
            let norm = weights.tensor("output_norm.weight")?;
            assert_eq!(q.byte_len(), 288);
            assert_eq!(norm.byte_len(), 64);

            let prompt_len = ctx.prompt_tokens().len();
            ctx.advance_position(prompt_len)?;
            if let Some(first_logit) = ctx.logits_mut().first_mut() {
                *first_logit = 1.0;
            }
            Ok(())
        }

        fn decode_step(
            &self,
            ctx: &mut ExecutionContext<'_>,
            _weights: &dyn ModelWeights,
        ) -> Result<(), K3Error> {
            let token = ctx
                .decode_token()
                .ok_or_else(|| K3Error::CapabilityMismatch {
                    architecture: self.architecture_id().to_owned(),
                    reason: "decode_step called without a staged decode token".to_owned(),
                })?;
            ctx.advance_position(1)?;
            if let Some(slot) = ctx.logits_mut().get_mut(token as usize) {
                *slot = 2.0;
            }
            Ok(())
        }
    }

    #[test]
    fn quant_type_storage_bytes_and_gguf_mapping() {
        assert_eq!(QuantType::from_gguf_type_id(0), Some(QuantType::F32));
        assert_eq!(QuantType::from_gguf_type_id(1), Some(QuantType::F16));
        assert_eq!(QuantType::from_gguf_type_id(2), Some(QuantType::Q4_0));
        assert_eq!(QuantType::from_gguf_type_id(8), Some(QuantType::Q8_0));
        assert_eq!(QuantType::from_gguf_type_id(12), Some(QuantType::Q4_K_M));
        assert_eq!(QuantType::from_gguf_type_id(14), Some(QuantType::Q6_K));
        assert_eq!(QuantType::from_gguf_type_id(30), Some(QuantType::BF16));
        assert_eq!(QuantType::from_gguf_type_id(999), None);

        assert_eq!(QuantType::F32.storage_bytes("w", 16).unwrap(), 64);
        assert_eq!(QuantType::F16.storage_bytes("w", 16).unwrap(), 32);
        assert_eq!(QuantType::BF16.storage_bytes("w", 16).unwrap(), 32);
        assert_eq!(QuantType::Q4_0.storage_bytes("w", 64).unwrap(), 36);
        assert_eq!(QuantType::Q8_0.storage_bytes("w", 64).unwrap(), 68);
        assert_eq!(QuantType::Q4_K_M.storage_bytes("w", 512).unwrap(), 288);
        assert_eq!(QuantType::Q6_K.storage_bytes("w", 256).unwrap(), 210);

        // Unaligned block size must fail
        let unaligned = QuantType::Q4_K_M.storage_bytes("blk.0.wq", 300);
        assert!(matches!(unaligned, Err(K3Error::TensorBoundsError { .. })));
    }

    #[test]
    fn tensor_view_bounds_calculation_verifies_valid_and_invalid_ranges() {
        let backing = vec![0xAB_u8; 352];

        // Valid Q4_K_M tensor: 256 * 2 = 512 elements -> 2 blocks * 144 = 288 bytes at offset 0
        let view_q4 = TensorView::from_bytes(
            "blk.0.attn_q.weight",
            QuantType::Q4_K_M,
            &[256, 2],
            0,
            &backing,
        )
        .expect("valid Q4_K_M bounds");
        assert_eq!(view_q4.name(), "blk.0.attn_q.weight");
        assert_eq!(view_q4.quant(), QuantType::Q4_K_M);
        assert_eq!(view_q4.shape(), &[256, 2]);
        assert_eq!(view_q4.num_elements(), 512);
        assert_eq!(view_q4.offset(), 0);
        assert_eq!(view_q4.byte_len(), 288);
        assert_eq!(view_q4.as_bytes().as_ptr(), backing.as_ptr());

        // Valid F32 tensor: 16 elements -> 64 bytes at offset 288 (288 + 64 == 352)
        let view_f32 =
            TensorView::from_bytes("output_norm.weight", QuantType::F32, &[16], 288, &backing)
                .expect("valid F32 bounds at end of buffer");
        assert_eq!(view_f32.offset(), 288);
        assert_eq!(view_f32.byte_len(), 64);

        // Out-of-bounds by 1 byte
        let oob =
            TensorView::from_bytes("output_norm.weight", QuantType::F32, &[16], 289, &backing);
        assert_eq!(
            oob,
            Err(K3Error::TensorBoundsError {
                tensor_name: "output_norm.weight".to_owned(),
                offset: 289,
                byte_len: 64,
                buffer_len: 352,
                reason: "tensor range [289..353] exceeds backing buffer length 352".to_owned(),
            })
        );

        // Empty shape rejected
        let empty_shape = TensorView::from_bytes("empty", QuantType::F32, &[], 0, &backing);
        assert!(matches!(
            empty_shape,
            Err(K3Error::TensorBoundsError { .. })
        ));

        // Zero dimension rejected
        let zero_dim = TensorView::from_bytes("zero", QuantType::F32, &[256, 0], 0, &backing);
        assert!(matches!(zero_dim, Err(K3Error::TensorBoundsError { .. })));

        // Offset + byte_len overflow rejected
        let overflow_offset =
            TensorView::from_bytes("overflow", QuantType::F32, &[16], usize::MAX - 10, &backing);
        assert!(matches!(
            overflow_offset,
            Err(K3Error::TensorBoundsError { .. })
        ));
    }

    #[test]
    fn k3_error_diagnostic_formatting_is_structured_and_actionable() {
        let cap_err = K3Error::CapabilityMismatch {
            architecture: "qwen3.8".to_owned(),
            reason: "recipe 'qwen3.8' is not compiled into this binary".to_owned(),
        };
        assert_eq!(cap_err.diagnostic_code(), "capability_mismatch");
        assert_eq!(
            cap_err.to_string(),
            "[capability_mismatch] unsupported architecture 'qwen3.8': recipe 'qwen3.8' is not compiled into this binary"
        );

        let mem_err = K3Error::InsufficientLockableMemory {
            required_bytes: 17_179_869_184,
            weight_bytes: 16_106_127_360,
            kv_bytes: 1_073_741_824,
            available_bytes: 12_884_901_888,
            rlimit_memlock_bytes: 8_589_934_592,
        };
        assert_eq!(mem_err.diagnostic_code(), "insufficient_lockable_memory");
        assert_eq!(
            mem_err.to_string(),
            "[insufficient_lockable_memory] required 17179869184 bytes (weights: 16106127360 bytes, kv: 1073741824 bytes) exceeds lockable capacity (available RAM: 12884901888 bytes, RLIMIT_MEMLOCK: 8589934592 bytes)"
        );

        let ctx_err = K3Error::ContextLengthExceeded {
            requested_tokens: 4097,
            max_context_tokens: 4096,
        };
        assert_eq!(ctx_err.diagnostic_code(), "context_length_exceeded");
        assert_eq!(
            ctx_err.to_string(),
            "[context_length_exceeded] sequence length 4097 tokens exceeds configured n_ctx limit of 4096 tokens"
        );

        let aff_err = K3Error::AffinityError {
            requested_mask: 0x00FF,
            requested_threads: 12,
            reason: "requested 12 threads exceeds 8 cores in mask 0x00FF".to_owned(),
        };
        assert_eq!(aff_err.diagnostic_code(), "affinity_error");
        assert_eq!(
            aff_err.to_string(),
            "[affinity_error] invalid affinity configuration (mask: 0x00FF, threads: 12): requested 12 threads exceeds 8 cores in mask 0x00FF"
        );

        let bounds_err = K3Error::TensorBoundsError {
            tensor_name: "blk.0.attn_k.weight".to_owned(),
            offset: 1024,
            byte_len: 4096,
            buffer_len: 2048,
            reason: "tensor range [1024..5120] exceeds backing buffer length 2048".to_owned(),
        };
        assert_eq!(bounds_err.diagnostic_code(), "tensor_bounds_error");
        assert_eq!(
            bounds_err.to_string(),
            "[tensor_bounds_error] tensor 'blk.0.attn_k.weight' (offset: 1024, len: 4096, buffer_len: 2048): tensor range [1024..5120] exceeds backing buffer length 2048"
        );
    }

    #[test]
    fn single_prompt_execution_context_and_arch_recipe_lifecycle() {
        let backing = vec![0u8; 352];
        let weights = DummyWeights { buffer: &backing };
        let recipe = DummyRecipe;

        assert_eq!(recipe.architecture_id(), "synthetic");
        assert!(recipe.supports_quant(QuantType::Q4_K_M));
        assert!(!recipe.supports_quant(QuantType::Q8_0));
        assert!(weights.contains_tensor("blk.0.attn_q.weight"));
        assert!(!weights.contains_tensor("nonexistent.weight"));
        assert_eq!(weights.tensor_count(), 2);
        assert_eq!(weights.total_weight_bytes(), 352);

        let prompt = [10_u32, 20, 30];
        let mut logits = [0.0_f32; 8];
        let mut kv_cache = [0.0_f32; 64];

        let mut ctx = ExecutionContext::new(&prompt, 4, &mut logits, &mut kv_cache)
            .expect("prompt fits in n_ctx=4");
        recipe
            .prefill_step(&mut ctx, &weights)
            .expect("prefill succeeds");
        assert_eq!(ctx.position(), 3);
        assert!((ctx.logits()[0] - 1.0).abs() < f32::EPSILON);

        // Decode 1 token (reaching n_ctx = 4)
        ctx.prepare_decode(5).expect("4th token fits in n_ctx=4");
        recipe
            .decode_step(&mut ctx, &weights)
            .expect("decode succeeds");
        assert_eq!(ctx.position(), 4);
        assert!((ctx.logits()[5] - 2.0).abs() < f32::EPSILON);

        // Attempting a 5th token must fail with ContextLengthExceeded (ADR-027)
        let overflow = ctx.prepare_decode(6);
        assert_eq!(
            overflow,
            Err(K3Error::ContextLengthExceeded {
                requested_tokens: 5,
                max_context_tokens: 4,
            })
        );
    }
}
