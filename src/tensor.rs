//! Quantization descriptors, zero-copy tensor views, and model weight access traits.

use core::fmt;

use crate::error::K3Error;

/// Supported tensor quantization and floating-point storage formats (ADR-017).
#[allow(non_camel_case_types)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum QuantType {
    /// Unquantized 32-bit IEEE 754 floating-point (`ggml_type = 0`).
    F32,
    /// Unquantized 16-bit IEEE 754 floating-point (`ggml_type = 1`).
    F16,
    /// Unquantized 16-bit brain floating-point (`ggml_type = 30`).
    BF16,
    /// 4-bit round-to-nearest quantization in blocks of 32 elements (`ggml_type = 2`).
    Q4_0,
    /// 8-bit round-to-nearest quantization in blocks of 32 elements (`ggml_type = 8`).
    Q8_0,
    /// 4-bit K-quant super-blocks of 256 elements (`ggml_type = 12`, primary format for `Q4_K_M`).
    Q4_K_M,
    /// 6-bit K-quant super-blocks of 256 elements (`ggml_type = 14`, used by mixed `Q4_K_M` tensors).
    Q6_K,
}

impl QuantType {
    /// Returns the canonical string identifier for this quantization type.
    #[must_use]
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::F32 => "F32",
            Self::F16 => "F16",
            Self::BF16 => "BF16",
            Self::Q4_0 => "Q4_0",
            Self::Q8_0 => "Q8_0",
            Self::Q4_K_M => "Q4_K_M",
            Self::Q6_K => "Q6_K",
        }
    }

    /// Maps a GGUF v3 `ggml_type` integer ID to a [`QuantType`], if supported.
    #[must_use]
    pub const fn from_gguf_type_id(type_id: u32) -> Option<Self> {
        match type_id {
            0 => Some(Self::F32),
            1 => Some(Self::F16),
            2 => Some(Self::Q4_0),
            8 => Some(Self::Q8_0),
            12 => Some(Self::Q4_K_M),
            14 => Some(Self::Q6_K),
            30 => Some(Self::BF16),
            _ => None,
        }
    }

    /// Returns the number of scalar elements per quantization block.
    #[must_use]
    pub const fn block_elements(&self) -> usize {
        match self {
            Self::F32 | Self::F16 | Self::BF16 => 1,
            Self::Q4_0 | Self::Q8_0 => 32,
            Self::Q4_K_M | Self::Q6_K => 256,
        }
    }

    /// Returns the exact storage size in bytes of a single quantization block.
    #[must_use]
    pub const fn block_bytes(&self) -> usize {
        match self {
            Self::F32 => 4,
            Self::F16 | Self::BF16 => 2,
            Self::Q4_0 => 18,
            Self::Q8_0 => 34,
            Self::Q4_K_M => 144,
            Self::Q6_K => 210,
        }
    }

    /// Calculates the exact byte footprint for `num_elements` of this quantization format.
    ///
    /// # Errors
    /// Returns [`K3Error::TensorBoundsError`] if `num_elements` is zero, not an exact multiple
    /// of [`Self::block_elements`], or if the byte size calculation overflows `usize`.
    pub fn storage_bytes(&self, tensor_name: &str, num_elements: usize) -> Result<usize, K3Error> {
        if num_elements == 0 {
            return Err(K3Error::TensorBoundsError {
                tensor_name: tensor_name.to_owned(),
                offset: 0,
                byte_len: 0,
                buffer_len: 0,
                reason: "tensor element count must be greater than zero".to_owned(),
            });
        }

        let block_elems = self.block_elements();
        if num_elements % block_elems != 0 {
            return Err(K3Error::TensorBoundsError {
                tensor_name: tensor_name.to_owned(),
                offset: 0,
                byte_len: 0,
                buffer_len: 0,
                reason: format!(
                    "element count {num_elements} is not divisible by {quant} block size {block_elems}",
                    quant = self.as_str()
                ),
            });
        }

        let num_blocks = num_elements / block_elems;
        num_blocks
            .checked_mul(self.block_bytes())
            .ok_or_else(|| K3Error::TensorBoundsError {
                tensor_name: tensor_name.to_owned(),
                offset: 0,
                byte_len: usize::MAX,
                buffer_len: 0,
                reason: format!(
                    "byte size overflow for {num_elements} elements of type {quant}",
                    quant = self.as_str()
                ),
            })
    }
}

impl fmt::Display for QuantType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Zero-copy view into an `mlock`ed weight buffer with verified shape and byte bounds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TensorView<'a> {
    name: &'a str,
    quant: QuantType,
    shape: &'a [usize],
    num_elements: usize,
    offset: usize,
    data: &'a [u8],
}

impl<'a> TensorView<'a> {
    /// Computes `(num_elements, byte_len)` and verifies that `[offset .. offset + byte_len]`
    /// lies strictly within `buffer_len`.
    ///
    /// # Errors
    /// Returns [`K3Error::TensorBoundsError`] if `shape` is empty, contains a zero dimension,
    /// overflows `usize`, fails quantization block alignment, or exceeds `buffer_len`.
    pub fn calculate_bounds(
        name: &str,
        quant: QuantType,
        shape: &[usize],
        offset: usize,
        buffer_len: usize,
    ) -> Result<(usize, usize), K3Error> {
        if shape.is_empty() {
            return Err(K3Error::TensorBoundsError {
                tensor_name: name.to_owned(),
                offset,
                byte_len: 0,
                buffer_len,
                reason: "tensor shape must contain at least one dimension".to_owned(),
            });
        }

        let mut num_elements = 1usize;
        for &dim in shape {
            if dim == 0 {
                return Err(K3Error::TensorBoundsError {
                    tensor_name: name.to_owned(),
                    offset,
                    byte_len: 0,
                    buffer_len,
                    reason: "tensor shape dimensions must be non-zero".to_owned(),
                });
            }
            num_elements =
                num_elements
                    .checked_mul(dim)
                    .ok_or_else(|| K3Error::TensorBoundsError {
                        tensor_name: name.to_owned(),
                        offset,
                        byte_len: usize::MAX,
                        buffer_len,
                        reason: "tensor shape element product overflowed usize".to_owned(),
                    })?;
        }

        let byte_len = quant
            .storage_bytes(name, num_elements)
            .map_err(|err| match err {
                K3Error::TensorBoundsError {
                    tensor_name,
                    byte_len,
                    reason,
                    ..
                } => K3Error::TensorBoundsError {
                    tensor_name,
                    offset,
                    byte_len,
                    buffer_len,
                    reason,
                },
                other => other,
            })?;

        let end_offset =
            offset
                .checked_add(byte_len)
                .ok_or_else(|| K3Error::TensorBoundsError {
                    tensor_name: name.to_owned(),
                    offset,
                    byte_len,
                    buffer_len,
                    reason: "tensor offset + byte_len overflowed usize".to_owned(),
                })?;

        if end_offset > buffer_len {
            return Err(K3Error::TensorBoundsError {
                tensor_name: name.to_owned(),
                offset,
                byte_len,
                buffer_len,
                reason: format!(
                    "tensor range [{offset}..{end_offset}] exceeds backing buffer length {buffer_len}"
                ),
            });
        }

        Ok((num_elements, byte_len))
    }

    /// Constructs a zero-copy [`TensorView`] slicing `buffer[offset .. offset + byte_len]`.
    ///
    /// # Errors
    /// Returns [`K3Error::TensorBoundsError`] if [`Self::calculate_bounds`] fails.
    pub fn from_bytes(
        name: &'a str,
        quant: QuantType,
        shape: &'a [usize],
        offset: usize,
        buffer: &'a [u8],
    ) -> Result<Self, K3Error> {
        let (num_elements, byte_len) =
            Self::calculate_bounds(name, quant, shape, offset, buffer.len())?;
        let data = &buffer[offset..offset + byte_len];
        Ok(Self {
            name,
            quant,
            shape,
            num_elements,
            offset,
            data,
        })
    }

    /// Returns the tensor identifier name.
    #[must_use]
    pub const fn name(&self) -> &'a str {
        self.name
    }

    /// Returns the tensor's quantization format.
    #[must_use]
    pub const fn quant(&self) -> QuantType {
        self.quant
    }

    /// Returns the tensor's dimension slice.
    #[must_use]
    pub const fn shape(&self) -> &'a [usize] {
        self.shape
    }

    /// Returns the total number of scalar elements in the tensor.
    #[must_use]
    pub const fn num_elements(&self) -> usize {
        self.num_elements
    }

    /// Returns the starting byte offset of this tensor inside its backing buffer.
    #[must_use]
    pub const fn offset(&self) -> usize {
        self.offset
    }

    /// Returns the exact length in bytes of the tensor slice.
    #[must_use]
    pub const fn byte_len(&self) -> usize {
        self.data.len()
    }

    /// Returns the zero-copy byte slice backing this tensor.
    #[must_use]
    pub const fn as_bytes(&self) -> &'a [u8] {
        self.data
    }
}

/// Zero-copy interface for querying resident, `mlock`ed model weights (ADR-037).
pub trait ModelWeights: Sync {
    /// Looks up a zero-copy [`TensorView`] by tensor name.
    ///
    /// # Errors
    /// Returns [`K3Error::TensorBoundsError`] if the tensor is missing or out of bounds.
    fn tensor(&self, name: &str) -> Result<TensorView<'_>, K3Error>;

    /// Returns `true` if a tensor with `name` exists and is valid.
    fn contains_tensor(&self, name: &str) -> bool {
        self.tensor(name).is_ok()
    }

    /// Returns the total number of tensors in the model weight collection.
    fn tensor_count(&self) -> usize;

    /// Returns the total resident byte footprint of all model weights.
    fn total_weight_bytes(&self) -> usize;
}
