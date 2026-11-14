//! In-place shared memory arena layouts for the `SpacemiT` K3 runtime.
//!
//! All structures are `#[repr(C)]` with zero implicit compiler padding,
//! implementing zerocopy 0.8 traits (`FromBytes`, `IntoBytes`, `KnownLayout`, `Immutable`).
//!
//! Cacheline-aligned boundaries (64 bytes) ensure optimal memory bus transactions
//! and vector memory operations on `SpacemiT` K3 X100 (VLEN=256) and A100 (VLEN=1024) cores.

use core::mem::{align_of, size_of};

use zerocopy::{FromBytes, Immutable, IntoBytes, KnownLayout};

/// Magic identifier for K3 shared memory arena (`0x4B334B56` = b"K3KV").
pub const ARENA_MAGIC: u32 = 0x4B33_4B56;

/// Magic identifier byte array `*b"K3KV"`.
pub const ARENA_MAGIC_BYTES: [u8; 4] = *b"K3KV";

/// Version of the arena layout schema.
pub const ARENA_VERSION: u32 = 1;

/// Fixed number of tokens per KV cache block.
pub const BLOCK_SIZE_TOKENS: u32 = 16;

/// Standard hardware cacheline alignment (64 bytes).
pub const CACHELINE_ALIGNMENT: usize = 64;

// Block tag constants per ADR-018 & ADR-042
pub const TAG_FREE: u8 = 0;
pub const TAG_VOLATILE: u8 = 1;
pub const TAG_PERSIST: u8 = 2;
pub const TAG_PURGE: u8 = 3;

/// Symbolic enumeration of block retention tags.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum KvBlockTag {
    /// Available for allocation.
    Free = TAG_FREE,
    /// Ephemeral context (discarded on eviction or session completion).
    Volatile = TAG_VOLATILE,
    /// Persistent checkpoint (retained across worker restarts and prompt sharing).
    Persist = TAG_PERSIST,
    /// Explicitly marked for eviction and zeroing.
    Purge = TAG_PURGE,
}

impl KvBlockTag {
    /// Converts a raw `u8` to a [`KvBlockTag`].
    #[must_use]
    pub const fn from_u8(val: u8) -> Option<Self> {
        match val {
            TAG_FREE => Some(Self::Free),
            TAG_VOLATILE => Some(Self::Volatile),
            TAG_PERSIST => Some(Self::Persist),
            TAG_PURGE => Some(Self::Purge),
            _ => None,
        }
    }

    /// Returns the raw `u8` representation.
    #[must_use]
    pub const fn as_u8(self) -> u8 {
        self as u8
    }
}

/// Fixed 128-byte header aligned to 64-byte cacheline boundaries at offset 0 of the arena.
///
/// Contains placement geometry, sizing bounds, and byte offsets to all arena sub-regions.
#[repr(C, align(64))]
#[allow(clippy::pub_underscore_fields)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, FromBytes, IntoBytes, KnownLayout, Immutable)]
pub struct ArenaHeader {
    /// Magic identifier (`0x4B334B56` / `b"K3KV"`).
    pub magic: u32,
    /// Layout schema version (currently 1).
    pub version: u32,
    /// Total capacity of the mapped shared memory arena in bytes.
    pub total_bytes: u64,
    /// Total number of 16-token KV blocks allocated in this arena.
    pub block_count: u32,
    /// Number of tokens per block (fixed at 16).
    pub tokens_per_block: u32,
    /// Maximum configured context length (`n_ctx`).
    pub n_ctx: u32,
    /// Attention head dimension (e.g. 64, 128).
    pub head_dim: u32,
    /// Number of transformer layers in the model.
    pub num_layers: u32,
    /// Number of key-value attention heads (for GQA/MQA models).
    pub head_count_kv: u32,
    /// Model vocabulary size (dimension of output logits).
    pub vocab_size: u32,
    /// Number of active token slots (typically 1 for single-stream worker).
    pub active_slots: u32,
    /// Byte offset from arena base to the [`KvBlockMeta`] table.
    pub meta_table_offset: u64,
    /// Byte offset from arena base to the input token slot region.
    pub token_slots_offset: u64,
    /// Byte offset from arena base to the KV cache block vector data region.
    pub kv_blocks_offset: u64,
    /// Byte offset from arena base to the output logit slab region.
    pub output_slabs_offset: u64,
    /// Monotonically incrementing generation counter for state tracking.
    pub generation_counter: u64,
    /// Sliding window attention size (0 if full causal attention).
    pub swa_window_size: u32,
    /// Total number of routed experts in Sparse `MoE` models (0 for dense models).
    pub num_experts: u16,
    /// Number of active experts routed per token (e.g. 2 for Mixtral).
    pub num_experts_per_tok: u16,
    /// Byte offset from arena base to pre-split or derived weights.
    pub derived_weights_offset: u64,
    /// Explicit padding to ensure exact 128-byte size with zero compiler gaps.
    pub _reserved: [u8; 24],
}

impl Default for ArenaHeader {
    fn default() -> Self {
        Self {
            magic: ARENA_MAGIC,
            version: ARENA_VERSION,
            total_bytes: 0,
            block_count: 4,
            tokens_per_block: BLOCK_SIZE_TOKENS,
            n_ctx: 64,
            head_dim: 64,
            num_layers: 2,
            head_count_kv: 2,
            vocab_size: 32000,
            active_slots: 1,
            meta_table_offset: 128,
            token_slots_offset: 128 + 4 * 64,
            kv_blocks_offset: 1024,
            output_slabs_offset: 4096,
            generation_counter: 0,
            swa_window_size: 0,
            num_experts: 0,
            num_experts_per_tok: 0,
            derived_weights_offset: 0,
            _reserved: [0; 24],
        }
    }
}

impl ArenaHeader {
    /// Validates magic and version consistency.
    ///
    /// # Errors
    /// Returns an error message if magic, version, or token block size are invalid.
    pub fn validate(&self) -> Result<(), &'static str> {
        if self.magic != ARENA_MAGIC {
            return Err("invalid arena magic signature");
        }
        if self.version != ARENA_VERSION {
            return Err("unsupported arena layout version");
        }
        if self.tokens_per_block != BLOCK_SIZE_TOKENS {
            return Err("tokens_per_block must be 16");
        }
        Ok(())
    }

    /// Returns the magic number as 4 big-endian bytes.
    #[must_use]
    pub const fn magic_bytes(&self) -> [u8; 4] {
        self.magic.to_be_bytes()
    }
}

/// 64-byte aligned metadata record for a 16-token KV cache block.
///
/// Designed to fit exactly within a single 64-byte cacheline.
/// Zero implicit padding bytes ensure seamless zerocopy casting.
#[repr(C, align(64))]
#[allow(clippy::pub_underscore_fields)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, FromBytes, IntoBytes, KnownLayout, Immutable)]
pub struct KvBlockMeta {
    /// Logical block index within the arena (`0 .. block_count`).
    pub block_id: u32,
    /// Retention policy tag (0=free, 1=volatile, 2=persist, 3=purge).
    pub tag: u8,
    /// Explicit padding byte aligning `refcount` to 2-byte boundary.
    pub _pad0: u8,
    /// Reference count for shared prefix caching.
    pub refcount: u16,
    /// Number of tokens committed to this block (`0 ..= 16`).
    pub committed_token_count: u32,
    /// Explicit padding aligning `sampler_rng_state` to 8-byte boundary.
    pub _pad1: u32,
    /// Checkpointed 256-bit PRNG state (e.g. for resume from checkpoint without re-prefill).
    pub sampler_rng_state: [u64; 4],
    /// Reserved trailing padding completing the 64-byte cacheline.
    pub padding: [u8; 16],
}

impl Default for KvBlockMeta {
    fn default() -> Self {
        Self {
            block_id: 0,
            tag: TAG_VOLATILE,
            _pad0: 0,
            refcount: 1,
            committed_token_count: 0,
            _pad1: 0,
            sampler_rng_state: [42, 0, 0, 0],
            padding: [0; 16],
        }
    }
}

impl KvBlockMeta {
    /// Creates a new [`KvBlockMeta`] with the specified block ID and initial committed token count.
    #[must_use]
    pub const fn new(block_id: u32, committed_token_count: u32) -> Self {
        Self {
            block_id,
            tag: TAG_VOLATILE,
            _pad0: 0,
            refcount: 1,
            committed_token_count,
            _pad1: 0,
            sampler_rng_state: [42, 0, 0, 0],
            padding: [0; 16],
        }
    }

    /// Returns the parsed [`KvBlockTag`].
    #[must_use]
    pub fn tag_enum(&self) -> Option<KvBlockTag> {
        KvBlockTag::from_u8(self.tag)
    }

    /// Sets the block retention tag.
    pub fn set_tag(&mut self, tag: KvBlockTag) {
        self.tag = tag.as_u8();
    }

    /// Returns the number of tokens committed to this block.
    #[must_use]
    pub const fn committed_tokens(&self) -> u32 {
        self.committed_token_count
    }

    /// Sets the number of committed tokens.
    pub fn set_committed_tokens(&mut self, count: u32) {
        self.committed_token_count = count;
    }

    /// Returns the current reference count.
    #[must_use]
    #[inline]
    pub const fn ref_count(&self) -> u16 {
        self.refcount
    }

    /// Increments the reference count using saturating addition.
    ///
    /// Returns the updated reference count.
    #[inline]
    pub fn inc_ref(&mut self) -> u16 {
        self.refcount = self.refcount.saturating_add(1);
        self.refcount
    }

    /// Decrements the reference count using saturating subtraction.
    ///
    /// Returns the updated reference count.
    #[inline]
    pub fn dec_ref(&mut self) -> u16 {
        self.refcount = self.refcount.saturating_sub(1);
        self.refcount
    }

    /// Returns `true` if this block is shared by multiple sessions (`refcount > 1`).
    #[must_use]
    #[inline]
    pub const fn is_shared(&self) -> bool {
        self.refcount > 1
    }

    /// Returns `true` if mutating this block requires a Copy-On-Write (COW) clone (`refcount > 1`).
    #[must_use]
    #[inline]
    pub const fn needs_cow(&self) -> bool {
        self.refcount > 1
    }

    /// Returns `true` if this block is unreferenced and eligible for LRU eviction (`refcount == 0 && tag == TAG_VOLATILE`).
    #[must_use]
    #[inline]
    pub const fn is_evictable(&self) -> bool {
        self.refcount == 0 && self.tag == TAG_VOLATILE
    }
}

/// Header preceding an input token slot in shared memory.
///
/// Followed immediately by `token_len * 4` bytes of `u32` token IDs.
#[repr(C, align(8))]
#[derive(
    Default, Debug, Clone, Copy, PartialEq, Eq, FromBytes, IntoBytes, KnownLayout, Immutable,
)]
pub struct TokenInputSlotHeader {
    /// Unique sequence identifier or request ID for this input.
    pub slot_seq_id: u64,
    /// Number of active tokens present in the trailing tokens array.
    pub token_len: u32,
    /// Starting sequence position for decode.
    pub start_position: u32,
}

/// Slab status indicators for synchronization between worker and supervisor.
pub const SLAB_STATUS_IDLE: u32 = 0;
pub const SLAB_STATUS_WRITTEN: u32 = 1;
pub const SLAB_STATUS_READ: u32 = 2;

/// 64-byte aligned header preceding the output logits slab in shared memory.
///
/// Logit float values (`[f32; vocab_size]`) start immediately at offset 64,
/// ensuring full cacheline alignment for vector stores (`vse32.v`).
#[repr(C, align(64))]
#[allow(clippy::pub_underscore_fields)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, FromBytes, IntoBytes, KnownLayout, Immutable)]
pub struct OutputSlabHeader {
    /// Associated forward pass request identifier.
    pub request_id: u64,
    /// Vocabulary dimension (number of `f32` logit elements).
    pub vocab_size: u32,
    /// Sequence position corresponding to these logits.
    pub position: u32,
    /// Status flag (0=idle, 1=written, 2=read).
    pub status: u32,
    /// Reserved trailing padding completing the 64-byte header.
    pub _reserved: [u8; 44],
}

impl Default for OutputSlabHeader {
    fn default() -> Self {
        Self {
            request_id: 0,
            vocab_size: 0,
            position: 0,
            status: SLAB_STATUS_IDLE,
            _reserved: [0; 44],
        }
    }
}

/// 64-byte aligned header preceding the ping-pong recurrent state arena region.
///
/// Followed immediately at byte offset `64` by two identically sized banks
/// (`bank 0` and `bank 1`), each spanning `bank_stride_bytes` bytes for crash-safe
/// alternating step commits (`O(1)` memory footprint for SSM, RWKV, and LFM2 shortconv layers).
#[repr(C, align(64))]
#[allow(clippy::pub_underscore_fields)]
#[derive(
    Default, Debug, Clone, Copy, PartialEq, Eq, FromBytes, IntoBytes, KnownLayout, Immutable,
)]
pub struct RecurrentStateSlotHeader {
    /// Sequence step index last durably committed to `active_bank`.
    pub committed_step: u32,
    /// Active ping-pong state bank index (`0` or `1`).
    pub active_bank: u32,
    /// Number of recurrent/SSM layers stored inside each bank.
    pub num_recurrent_layers: u32,
    /// Per-layer convolution history byte stride within each bank.
    pub conv_state_bytes_per_layer: u32,
    /// Per-layer recurrent SSM/WKV state byte stride within each bank.
    pub ssm_state_bytes_per_layer: u32,
    /// Explicit alignment padding before [`Self::bank_stride_bytes`].
    pub _pad0: u32,
    /// Total byte stride of a single recurrent state bank.
    pub bank_stride_bytes: u64,
    /// Reserved trailing bytes completing the 64-byte header.
    pub _reserved: [u8; 32],
}

impl RecurrentStateSlotHeader {
    /// Constructs a new [`RecurrentStateSlotHeader`] positioned at step `0` with `active_bank = 0`.
    #[must_use]
    pub const fn new(
        num_recurrent_layers: u32,
        conv_state_bytes_per_layer: u32,
        ssm_state_bytes_per_layer: u32,
        bank_stride_bytes: u64,
    ) -> Self {
        Self {
            committed_step: 0,
            active_bank: 0,
            num_recurrent_layers,
            conv_state_bytes_per_layer,
            ssm_state_bytes_per_layer,
            _pad0: 0,
            bank_stride_bytes,
            _reserved: [0; 32],
        }
    }

    /// Returns the currently inactive staging bank index (`1` when `active_bank == 0`, else `0`).
    #[must_use]
    pub const fn inactive_bank(&self) -> u32 {
        if self.active_bank == 0 { 1 } else { 0 }
    }

    /// Returns the byte offset of `bank` relative to the start of the recurrent state region.
    #[must_use]
    pub const fn bank_offset_bytes(&self, bank: u32) -> u64 {
        (size_of::<Self>() as u64) + (bank as u64) * self.bank_stride_bytes
    }

    /// Returns the total byte span required for the header plus both ping-pong banks.
    #[must_use]
    pub const fn total_region_bytes(&self) -> u64 {
        (size_of::<Self>() as u64) + 2 * self.bank_stride_bytes
    }

    /// Validates structural invariants of this recurrent state header.
    ///
    /// # Errors
    /// Returns a static diagnostic message if `active_bank > 1` or `bank_stride_bytes`
    /// is smaller than the aggregate per-layer state requirements.
    pub fn validate(&self) -> Result<(), &'static str> {
        if self.active_bank > 1 {
            return Err("RecurrentStateSlotHeader active_bank must be 0 or 1");
        }
        let per_layer = u64::from(self.conv_state_bytes_per_layer)
            .saturating_add(u64::from(self.ssm_state_bytes_per_layer));
        let min_bank_stride = u64::from(self.num_recurrent_layers).saturating_mul(per_layer);
        if self.bank_stride_bytes < min_bank_stride {
            return Err(
                "RecurrentStateSlotHeader bank_stride_bytes smaller than layer requirements",
            );
        }
        Ok(())
    }
}

// Compile-time layout invariants verification
const _: () = {
    assert!(size_of::<ArenaHeader>() == 128);
    assert!(align_of::<ArenaHeader>() == 64);

    assert!(size_of::<KvBlockMeta>() == 64);
    assert!(align_of::<KvBlockMeta>() == 64);

    assert!(size_of::<TokenInputSlotHeader>() == 16);
    assert!(align_of::<TokenInputSlotHeader>() == 8);

    assert!(size_of::<OutputSlabHeader>() == 64);
    assert!(align_of::<OutputSlabHeader>() == 64);

    assert!(size_of::<RecurrentStateSlotHeader>() == 64);
    assert!(align_of::<RecurrentStateSlotHeader>() == 64);
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_arena_header_layout_and_validation() {
        assert_eq!(size_of::<ArenaHeader>(), 128);
        assert_eq!(align_of::<ArenaHeader>(), 64);

        let hdr = ArenaHeader::default();
        assert_eq!(hdr.magic, ARENA_MAGIC);
        assert_eq!(hdr.version, ARENA_VERSION);
        assert_eq!(hdr.magic_bytes(), ARENA_MAGIC_BYTES);
        assert!(hdr.validate().is_ok());

        let mut bad_magic = hdr;
        bad_magic.magic = 0xDEAD_BEEF;
        assert!(bad_magic.validate().is_err());

        let mut bad_version = hdr;
        bad_version.version = 99;
        assert!(bad_version.validate().is_err());

        let mut bad_block_size = hdr;
        bad_block_size.tokens_per_block = 32;
        assert!(bad_block_size.validate().is_err());
    }

    #[test]
    fn test_kv_block_meta_cacheline_alignment_and_tags() {
        assert_eq!(size_of::<KvBlockMeta>(), CACHELINE_ALIGNMENT);
        assert_eq!(align_of::<KvBlockMeta>(), CACHELINE_ALIGNMENT);

        let mut meta = KvBlockMeta::default();
        assert_eq!(meta.tag_enum(), Some(KvBlockTag::Volatile));
        assert_eq!(meta.committed_tokens(), 0);

        meta.set_tag(KvBlockTag::Persist);
        assert_eq!(meta.tag, TAG_PERSIST);
        assert_eq!(meta.tag_enum(), Some(KvBlockTag::Persist));

        meta.set_committed_tokens(16);
        assert_eq!(meta.committed_token_count, 16);
        assert_eq!(meta.committed_tokens(), 16);

        meta.sampler_rng_state[0] = 123_456_789;
        assert_eq!(meta.sampler_rng_state[0], 123_456_789);
    }

    #[test]
    fn test_token_input_and_output_slab_headers() {
        assert_eq!(size_of::<TokenInputSlotHeader>(), 16);
        assert_eq!(align_of::<TokenInputSlotHeader>(), 8);

        let token_hdr = TokenInputSlotHeader {
            slot_seq_id: 42,
            token_len: 10,
            start_position: 0,
        };
        assert_eq!(token_hdr.slot_seq_id, 42);
        assert_eq!(token_hdr.token_len, 10);

        assert_eq!(size_of::<OutputSlabHeader>(), 64);
        assert_eq!(align_of::<OutputSlabHeader>(), 64);

        let slab_hdr = OutputSlabHeader {
            request_id: 100,
            vocab_size: 32000,
            position: 5,
            status: SLAB_STATUS_WRITTEN,
            _reserved: [0; 44],
        };
        assert_eq!(slab_hdr.status, SLAB_STATUS_WRITTEN);
    }

    #[test]
    fn test_zerocopy_byte_roundtrip() {
        use zerocopy::{FromBytes, IntoBytes};

        let hdr = ArenaHeader::default();
        let bytes = hdr.as_bytes();
        assert_eq!(bytes.len(), 128);

        let recovered = ArenaHeader::read_from_bytes(bytes).expect("zerocopy read_from_bytes");
        assert_eq!(recovered, hdr);

        let meta = KvBlockMeta::default();
        let meta_bytes = meta.as_bytes();
        assert_eq!(meta_bytes.len(), 64);
        let recovered_meta = KvBlockMeta::read_from_bytes(meta_bytes).expect("zerocopy meta");
        assert_eq!(recovered_meta, meta);
    }

    #[test]
    fn test_kv_block_meta_refcounting_lifecycle() {
        let mut meta = KvBlockMeta::default();
        // default has refcount=1, tag=TAG_VOLATILE
        assert_eq!(meta.refcount, 1);
        assert!(!meta.is_shared());
        assert!(!meta.needs_cow());
        assert!(!meta.is_evictable());

        // Increment refcount
        assert_eq!(meta.inc_ref(), 2);
        assert!(meta.is_shared());
        assert!(meta.needs_cow());
        assert!(!meta.is_evictable());

        // Decrement refcount back to 1
        assert_eq!(meta.dec_ref(), 1);
        assert!(!meta.is_shared());
        assert!(!meta.needs_cow());
        assert!(!meta.is_evictable());

        // Decrement to 0
        assert_eq!(meta.dec_ref(), 0);
        assert!(!meta.is_shared());
        assert!(!meta.needs_cow());
        assert!(meta.is_evictable()); // refcount == 0 && tag == TAG_VOLATILE

        // Saturating sub underflow safety
        assert_eq!(meta.dec_ref(), 0);
        assert_eq!(meta.refcount, 0);

        // Saturating add overflow safety
        meta.refcount = u16::MAX;
        assert_eq!(meta.inc_ref(), u16::MAX);
        assert_eq!(meta.refcount, u16::MAX);

        // If tag is not TAG_VOLATILE, not evictable even with refcount 0
        meta.refcount = 0;
        meta.set_tag(KvBlockTag::Persist);
        assert!(!meta.is_evictable());

        meta.set_tag(KvBlockTag::Free);
        assert!(!meta.is_evictable());
    }

    #[test]
    #[expect(
        clippy::used_underscore_binding,
        reason = "verifying exact carved reserved byte array length"
    )]
    fn test_arena_header_evolution_and_carved_fields() {
        assert_eq!(size_of::<ArenaHeader>(), 128);
        assert_eq!(align_of::<ArenaHeader>(), 64);

        let hdr = ArenaHeader {
            swa_window_size: 4096,
            num_experts: 8,
            num_experts_per_tok: 2,
            derived_weights_offset: 65536,
            ..Default::default()
        };

        let bytes = hdr.as_bytes();
        assert_eq!(bytes.len(), 128);

        let recovered = ArenaHeader::read_from_bytes(bytes).expect("roundtrip");
        assert_eq!(recovered.swa_window_size, 4096);
        assert_eq!(recovered.num_experts, 8);
        assert_eq!(recovered.num_experts_per_tok, 2);
        assert_eq!(recovered.derived_weights_offset, 65536);
        assert_eq!(recovered.version, ARENA_VERSION);
        assert_eq!(recovered._reserved.len(), 24);
        assert_eq!(recovered, hdr);
    }

    #[test]
    fn test_recurrent_state_slot_header_zerocopy() {
        let header = RecurrentStateSlotHeader {
            committed_step: 42,
            active_bank: 1,
            num_recurrent_layers: 12,
            conv_state_bytes_per_layer: 1536,
            ssm_state_bytes_per_layer: 32768,
            _pad0: 0,
            bank_stride_bytes: 12 * (1536 + 32768),
            _reserved: [0; 32],
        };
        assert!(header.validate().is_ok());
        assert_eq!(header.inactive_bank(), 0);
        assert_eq!(header.bank_offset_bytes(0), 64);
        assert_eq!(header.bank_offset_bytes(1), 64 + 12 * (1536 + 32768));
        assert_eq!(header.total_region_bytes(), 64 + 2 * 12 * (1536 + 32768));

        let bytes = header.as_bytes();
        assert_eq!(bytes.len(), 64);
        let recovered =
            RecurrentStateSlotHeader::read_from_bytes(bytes).expect("zerocopy roundtrip");
        assert_eq!(recovered, header);

        let mut bad_bank = header;
        bad_bank.active_bank = 2;
        assert!(bad_bank.validate().is_err());

        let mut undersized_stride = header;
        undersized_stride.bank_stride_bytes -= 64;
        assert!(undersized_stride.validate().is_err());
    }

    #[test]
    fn test_recurrent_state_slot_header_layout() {
        assert_eq!(size_of::<RecurrentStateSlotHeader>(), 64);
        assert_eq!(align_of::<RecurrentStateSlotHeader>(), 64);

        let default_hdr = RecurrentStateSlotHeader::default();
        assert_eq!(default_hdr.committed_step, 0);
        assert_eq!(default_hdr.active_bank, 0);
        assert_eq!(default_hdr.inactive_bank(), 1);
        assert!(default_hdr.validate().is_ok());

        let constructed = RecurrentStateSlotHeader::new(4, 256, 4096, 4 * (256 + 4096));
        assert!(constructed.validate().is_ok());
        assert_eq!(constructed.as_bytes().len(), CACHELINE_ALIGNMENT);
    }
}
