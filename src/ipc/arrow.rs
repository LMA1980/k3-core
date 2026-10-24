//! Apache Arrow zero-copy tensor views over shared memory arenas.
//!
//! Provides zero-copy wrapping of mapped shared memory into Apache Arrow
//! buffers and primitive arrays (`UInt32Array`, `Float32Array`).
//!
//! Zero bulk payload bytes are copied across processes:
//! `Buffer::from_custom_allocation` retains shared ownership of the backing
//! arena via `Arc<dyn Allocation>`, allowing readers to inspect tokens,
//! logits, and KV cache blocks via standard Apache Arrow columnar interfaces.

use core::mem::{align_of, size_of};
use std::any::Any;
use std::fmt;
use std::panic::RefUnwindSafe;
use std::ptr::NonNull;
use std::sync::Arc;

use arrow_array::{Float32Array, UInt32Array};
use arrow_buffer::alloc::Allocation;
use arrow_buffer::{Buffer, ScalarBuffer};
use zerocopy::FromBytes;

use super::shm::SharedMemoryArena;
use crate::error::K3Error;

/// Externally managed allocation holder for Arrow buffers referencing shared memory.
///
/// Keeps the underlying arena or memory mapping alive for the lifetime of
/// the Apache Arrow `Buffer` and any dependent `Array`s.
pub struct ShmArrowAllocation {
    _owner: Arc<dyn Any + Send + Sync + RefUnwindSafe>,
}

impl fmt::Debug for ShmArrowAllocation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ShmArrowAllocation").finish_non_exhaustive()
    }
}

impl ShmArrowAllocation {
    /// Creates a new allocation holder wrapping an arbitrary thread-safe owner.
    #[allow(clippy::new_ret_no_self)]
    pub fn new<T: Send + Sync + RefUnwindSafe + 'static>(owner: Arc<T>) -> Arc<dyn Allocation> {
        Arc::new(Self { _owner: owner })
    }
}

/// Creates an Apache Arrow [`Buffer`] referencing a contiguous region in shared memory.
///
/// # Safety
/// - `ptr` must be non-null and point to a valid, initialized byte region of at least `byte_len` bytes.
/// - The memory region must remain valid and immutable (or free of concurrent data races)
///   for as long as `owner` is retained.
/// - The caller must guarantee that reading from this memory does not violate Rust aliasing invariants.
///
/// # Errors
/// Returns [`K3Error::TensorBoundsError`] if `ptr` is null.
pub unsafe fn create_arrow_buffer_from_shm<T: Send + Sync + RefUnwindSafe + 'static>(
    owner: Arc<T>,
    ptr: *const u8,
    byte_len: usize,
) -> Result<Buffer, K3Error> {
    let non_null = NonNull::new(ptr.cast_mut()).ok_or_else(|| K3Error::TensorBoundsError {
        tensor_name: "arrow_shm_buffer".to_owned(),
        offset: 0,
        byte_len,
        buffer_len: byte_len,
        reason: "provided memory pointer is null".to_owned(),
    })?;

    let allocation_owner = ShmArrowAllocation::new(owner);

    // SAFETY:
    // 1. `non_null` is validated non-null above.
    // 2. The backing allocation is kept alive by `allocation_owner` until all referring
    //    Arrow buffers and arrays are dropped.
    // 3. Caller guarantees the pointer references at least `byte_len` valid initialized bytes.
    unsafe {
        Ok(Buffer::from_custom_allocation(
            non_null,
            byte_len,
            allocation_owner,
        ))
    }
}

/// Slices an existing Apache Arrow [`Buffer`] without copying bytes.
///
/// # Errors
/// Returns [`K3Error::TensorBoundsError`] if `offset + byte_len` exceeds `buffer.len()`.
pub fn slice_arrow_buffer(
    buffer: &Buffer,
    offset: usize,
    byte_len: usize,
) -> Result<Buffer, K3Error> {
    let end_offset = offset
        .checked_add(byte_len)
        .ok_or_else(|| K3Error::TensorBoundsError {
            tensor_name: "buffer_slice".to_owned(),
            offset,
            byte_len,
            buffer_len: buffer.len(),
            reason: "offset + byte_len overflowed usize".to_owned(),
        })?;

    if end_offset > buffer.len() {
        return Err(K3Error::TensorBoundsError {
            tensor_name: "buffer_slice".to_owned(),
            offset,
            byte_len,
            buffer_len: buffer.len(),
            reason: format!(
                "slice range [{offset}..{end_offset}] exceeds buffer length {len}",
                len = buffer.len()
            ),
        });
    }

    Ok(buffer.slice_with_length(offset, byte_len))
}

/// Exposes shared memory input tokens as an Apache Arrow [`UInt32Array`] without copying.
///
/// # Safety
/// - `tokens_ptr` must be non-null and aligned to `align_of::<u32>()` (4 bytes).
/// - Points to at least `token_count` valid `u32` token values in shared memory.
/// - `owner` must keep the memory mapping alive.
///
/// # Errors
/// Returns [`K3Error::TensorBoundsError`] if pointer is null, unaligned, or size overflows `usize`.
pub unsafe fn wrap_tokens_as_uint32_array<T: Send + Sync + RefUnwindSafe + 'static>(
    owner: Arc<T>,
    tokens_ptr: *const u32,
    token_count: usize,
) -> Result<UInt32Array, K3Error> {
    let ptr_val = tokens_ptr as usize;
    if ptr_val % align_of::<u32>() != 0 {
        return Err(K3Error::TensorBoundsError {
            tensor_name: "token_input_array".to_owned(),
            offset: ptr_val,
            byte_len: token_count * 4,
            buffer_len: token_count * 4,
            reason: format!(
                "tokens pointer 0x{ptr_val:X} is not 4-byte aligned (modulo: {m})",
                m = ptr_val % 4
            ),
        });
    }

    let byte_len =
        token_count
            .checked_mul(size_of::<u32>())
            .ok_or_else(|| K3Error::TensorBoundsError {
                tensor_name: "token_input_array".to_owned(),
                offset: 0,
                byte_len: usize::MAX,
                buffer_len: 0,
                reason: "token count byte length overflowed usize".to_owned(),
            })?;

    // SAFETY: Caller guarantees pointer validity and lifetime managed by owner.
    let buffer = unsafe { create_arrow_buffer_from_shm(owner, tokens_ptr.cast::<u8>(), byte_len)? };
    let scalar_buffer = ScalarBuffer::<u32>::new(buffer, 0, token_count);
    Ok(UInt32Array::new(scalar_buffer, None))
}

/// Exposes shared memory logits or KV cache vectors as an Apache Arrow [`Float32Array`] without copying.
///
/// # Safety
/// - `data_ptr` must be non-null and aligned to `align_of::<f32>()` (4 bytes).
/// - Points to at least `element_count` valid `f32` float values in shared memory.
/// - `owner` must keep the memory mapping alive.
///
/// # Errors
/// Returns [`K3Error::TensorBoundsError`] if pointer is null, unaligned, or size overflows `usize`.
pub unsafe fn wrap_floats_as_float32_array<T: Send + Sync + RefUnwindSafe + 'static>(
    owner: Arc<T>,
    name: &str,
    data_ptr: *const f32,
    element_count: usize,
) -> Result<Float32Array, K3Error> {
    let ptr_val = data_ptr as usize;
    if ptr_val % align_of::<f32>() != 0 {
        return Err(K3Error::TensorBoundsError {
            tensor_name: name.to_owned(),
            offset: ptr_val,
            byte_len: element_count * 4,
            buffer_len: element_count * 4,
            reason: format!(
                "float pointer 0x{ptr_val:X} for tensor '{name}' is not 4-byte aligned",
            ),
        });
    }

    let byte_len =
        element_count
            .checked_mul(size_of::<f32>())
            .ok_or_else(|| K3Error::TensorBoundsError {
                tensor_name: name.to_owned(),
                offset: 0,
                byte_len: usize::MAX,
                buffer_len: 0,
                reason: format!("element count {element_count} byte length overflowed usize"),
            })?;

    // SAFETY: Caller guarantees pointer validity and lifetime managed by owner.
    let buffer = unsafe { create_arrow_buffer_from_shm(owner, data_ptr.cast::<u8>(), byte_len)? };
    let scalar_buffer = ScalarBuffer::<f32>::new(buffer, 0, element_count);
    Ok(Float32Array::new(scalar_buffer, None))
}

/// High-level zero-copy Apache Arrow tensor views over a [`SharedMemoryArena`].
#[derive(Debug, Clone)]
pub struct ShmTensorViews {
    arena: Arc<SharedMemoryArena>,
}

impl ShmTensorViews {
    /// Creates a new view provider over an arena reference.
    #[must_use]
    pub fn new(arena: Arc<SharedMemoryArena>) -> Self {
        Self { arena }
    }

    /// Returns the backing arena.
    #[must_use]
    pub fn arena(&self) -> &Arc<SharedMemoryArena> {
        &self.arena
    }

    /// Returns a zero-copy Apache Arrow [`UInt32Array`] view over input prompt tokens.
    ///
    /// # Errors
    /// Returns [`K3Error::TensorBoundsError`] if slot is out of bounds or length exceeds arena.
    #[allow(clippy::cast_ptr_alignment)]
    pub fn tokens_view(&self, slot_idx: usize) -> Result<UInt32Array, K3Error> {
        let hdr = self.arena.header()?;
        let slot_offset =
            usize::try_from(hdr.token_slots_offset).map_err(|e| K3Error::TensorBoundsError {
                tensor_name: "tokens_view".to_owned(),
                offset: 0,
                byte_len: 0,
                buffer_len: self.arena.len(),
                reason: format!("token_slots_offset exceeds usize: {e}"),
            })?;
        let slot_base = slot_offset
            .checked_add(slot_idx.checked_mul(64 * 4 + 16).ok_or_else(|| {
                K3Error::TensorBoundsError {
                    tensor_name: "tokens_view".to_owned(),
                    offset: 0,
                    byte_len: 0,
                    buffer_len: self.arena.len(),
                    reason: "slot index overflowed usize".to_owned(),
                }
            })?)
            .ok_or_else(|| K3Error::TensorBoundsError {
                tensor_name: "tokens_view".to_owned(),
                offset: 0,
                byte_len: 0,
                buffer_len: self.arena.len(),
                reason: "slot index overflowed usize".to_owned(),
            })?;

        let slot_slice = self.arena.slice(slot_base, 16)?;
        let token_hdr =
            super::layout::TokenInputSlotHeader::ref_from_bytes(slot_slice).map_err(|e| {
                K3Error::TensorBoundsError {
                    tensor_name: format!("token_slot[{slot_idx}]"),
                    offset: slot_base,
                    byte_len: 16,
                    buffer_len: self.arena.len(),
                    reason: format!("failed to decode TokenInputSlotHeader: {e}"),
                }
            })?;

        let token_count = token_hdr.token_len as usize;
        let tokens_data_offset = slot_base + 16;
        let tokens_bytes = self.arena.slice(tokens_data_offset, token_count * 4)?;

        // SAFETY: tokens_bytes was validated within bounds and 16-byte aligned.
        unsafe {
            wrap_tokens_as_uint32_array(
                Arc::clone(&self.arena),
                tokens_bytes.as_ptr().cast::<u32>(),
                token_count,
            )
        }
    }

    /// Returns a zero-copy Apache Arrow [`Float32Array`] view over output logits.
    ///
    /// # Errors
    /// Returns [`K3Error::TensorBoundsError`] if slab offset or vocab length exceeds arena bounds.
    #[allow(clippy::cast_ptr_alignment)]
    pub fn logits_view(&self) -> Result<Float32Array, K3Error> {
        let hdr = self.arena.header()?;
        let slab_offset =
            usize::try_from(hdr.output_slabs_offset).map_err(|e| K3Error::TensorBoundsError {
                tensor_name: "output_slab".to_owned(),
                offset: 0,
                byte_len: 0,
                buffer_len: self.arena.len(),
                reason: format!("output_slabs_offset exceeds usize: {e}"),
            })?;
        let slab_slice = self.arena.slice(slab_offset, 64)?;
        let slab_hdr =
            super::layout::OutputSlabHeader::ref_from_bytes(slab_slice).map_err(|e| {
                K3Error::TensorBoundsError {
                    tensor_name: "output_slab".to_owned(),
                    offset: slab_offset,
                    byte_len: 64,
                    buffer_len: self.arena.len(),
                    reason: format!("failed to decode OutputSlabHeader: {e}"),
                }
            })?;

        let vocab_size = slab_hdr.vocab_size as usize;
        let logits_offset = slab_offset + 64;
        let logits_bytes = self.arena.slice(logits_offset, vocab_size * 4)?;

        // SAFETY: logits_bytes is 64-byte aligned and checked within bounds.
        unsafe {
            wrap_floats_as_float32_array(
                Arc::clone(&self.arena),
                "output_logits",
                logits_bytes.as_ptr().cast::<f32>(),
                vocab_size,
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_arrow_zero_copy_buffer_creation_and_slicing() {
        let data = vec![10_u32, 20, 30, 40, 50];
        let owner = Arc::new(data);

        // SAFETY: pointer points to valid 20 bytes of initialized u32 data.
        let buf = unsafe {
            create_arrow_buffer_from_shm(Arc::clone(&owner), owner.as_ptr().cast::<u8>(), 20)
        }
        .expect("arrow buffer creation");

        assert_eq!(buf.len(), 20);
        assert_eq!(buf.as_ptr(), owner.as_ptr().cast::<u8>());

        let sliced = slice_arrow_buffer(&buf, 4, 12).expect("slice");
        assert_eq!(sliced.len(), 12);
        // SAFETY: 4 bytes into the buffer is &owner[1].
        assert_eq!(sliced.as_ptr(), unsafe {
            owner.as_ptr().add(1).cast::<u8>()
        });

        let oob = slice_arrow_buffer(&buf, 10, 20);
        assert!(matches!(oob, Err(K3Error::TensorBoundsError { .. })));
    }

    #[test]
    fn test_wrap_tokens_as_uint32_array_zero_copy() {
        let tokens = vec![101_u32, 102, 103, 104];
        let owner = Arc::new(tokens);

        // SAFETY: tokens array is 4-byte aligned and has 4 elements.
        let array =
            unsafe { wrap_tokens_as_uint32_array(Arc::clone(&owner), owner.as_ptr(), owner.len()) }
                .expect("wrap tokens array");

        assert_eq!(array.len(), 4);
        assert_eq!(array.value(0), 101);
        assert_eq!(array.value(3), 104);
        assert_eq!(array.values().as_ptr(), owner.as_ptr());
    }

    #[test]
    fn test_wrap_floats_as_float32_array_zero_copy() {
        let logits = vec![0.1_f32, -0.5, 3.5, 2.5];
        let owner = Arc::new(logits);

        // SAFETY: logits array is 4-byte aligned and has 4 elements.
        let array = unsafe {
            wrap_floats_as_float32_array(
                Arc::clone(&owner),
                "test_logits",
                owner.as_ptr(),
                owner.len(),
            )
        }
        .expect("wrap floats array");

        assert_eq!(array.len(), 4);
        assert!((array.value(2) - 3.5).abs() < f32::EPSILON);
        assert_eq!(array.values().as_ptr(), owner.as_ptr());
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_shm_tensor_views_end_to_end() {
        use zerocopy::IntoBytes;

        let mut arena = SharedMemoryArena::create("views_test", 65536).expect("arena");
        {
            let hdr = arena.header_mut().expect("header_mut");
            *hdr = super::super::layout::ArenaHeader::default();
            hdr.token_slots_offset = 1024;
            hdr.output_slabs_offset = 4096;
        }

        // Setup token slot
        {
            let slot_hdr = super::super::layout::TokenInputSlotHeader {
                slot_seq_id: 1,
                token_len: 3,
                start_position: 0,
            };
            arena
                .slice_mut(1024, 16)
                .expect("slot_hdr")
                .copy_from_slice(slot_hdr.as_bytes());
            let tokens_slice = arena.slice_mut(1040, 12).expect("tokens_slice");
            let tokens = [42_u32, 43, 44];
            tokens_slice.copy_from_slice(tokens.as_bytes());
        }

        // Setup output slab
        {
            let slab_hdr = super::super::layout::OutputSlabHeader {
                request_id: 1,
                vocab_size: 4,
                position: 0,
                status: super::super::layout::SLAB_STATUS_WRITTEN,
                _reserved: [0; 44],
            };
            arena
                .slice_mut(4096, 64)
                .expect("slab_hdr")
                .copy_from_slice(slab_hdr.as_bytes());
            let logits_slice = arena.slice_mut(4160, 16).expect("logits_slice");
            let logits = [1.0_f32, 2.0, 3.0, 4.0];
            logits_slice.copy_from_slice(logits.as_bytes());
        }

        let views = ShmTensorViews::new(Arc::new(arena));
        let token_array = views.tokens_view(0).expect("token_array");
        assert_eq!(token_array.len(), 3);
        assert_eq!(token_array.value(0), 42);
        assert_eq!(token_array.value(2), 44);

        let logits_array = views.logits_view().expect("logits_array");
        assert_eq!(logits_array.len(), 4);
        assert!((logits_array.value(1) - 2.0).abs() < f32::EPSILON);
    }
}
