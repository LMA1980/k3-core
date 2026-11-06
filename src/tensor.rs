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
    /// 5-bit K-quant super-blocks of 256 elements (`ggml_type = 13`, used by `Q5_K_M` and `Q5_K_S`).
    Q5_K,
    /// 6-bit K-quant super-blocks of 256 elements (`ggml_type = 14`, used by mixed `Q4_K_M` tensors).
    Q6_K,
    /// 1.58-bit base-3 ternary quantization in blocks of 256 elements (`ggml_type = 34`).
    TQ1_0,
    /// 2-bit ternary quantization in blocks of 256 elements (`ggml_type = 35`).
    TQ2_0,
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
            Self::Q5_K => "Q5_K",
            Self::Q6_K => "Q6_K",
            Self::TQ1_0 => "TQ1_0",
            Self::TQ2_0 => "TQ2_0",
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
            13 => Some(Self::Q5_K),
            14 => Some(Self::Q6_K),
            30 => Some(Self::BF16),
            34 => Some(Self::TQ1_0),
            35 => Some(Self::TQ2_0),
            _ => None,
        }
    }

    /// Returns the GGUF v3 `ggml_type` integer ID for this quantization type.
    #[must_use]
    pub const fn ggml_type_id(&self) -> u32 {
        match self {
            Self::F32 => 0,
            Self::F16 => 1,
            Self::Q4_0 => 2,
            Self::Q8_0 => 8,
            Self::Q4_K_M => 12,
            Self::Q5_K => 13,
            Self::Q6_K => 14,
            Self::BF16 => 30,
            Self::TQ1_0 => 34,
            Self::TQ2_0 => 35,
        }
    }

    /// Returns the canonical type name identifier (alias for [`Self::as_str`]).
    #[must_use]
    pub const fn type_name(&self) -> &'static str {
        self.as_str()
    }

    /// Returns the GGUF v3 `ggml_type` integer ID (alias for [`Self::ggml_type_id`]).
    #[must_use]
    pub const fn ggml_type(&self) -> u32 {
        self.ggml_type_id()
    }

    /// Returns the number of scalar elements per quantization block.
    #[must_use]
    pub const fn block_elements(&self) -> usize {
        match self {
            Self::F32 | Self::F16 | Self::BF16 => 1,
            Self::Q4_0 | Self::Q8_0 => 32,
            Self::Q4_K_M | Self::Q5_K | Self::Q6_K | Self::TQ1_0 | Self::TQ2_0 => 256,
        }
    }

    /// Returns the number of scalar elements per quantization block (alias for [`Self::block_elements`]).
    #[must_use]
    pub const fn block_size(&self) -> usize {
        self.block_elements()
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
            Self::Q5_K => 176,
            Self::Q6_K => 210,
            Self::TQ1_0 => 54,
            Self::TQ2_0 => 66,
        }
    }

    /// Returns the exact storage size in bytes of a single quantization block (alias for [`Self::block_bytes`]).
    #[must_use]
    pub const fn type_size(&self) -> usize {
        self.block_bytes()
    }

    /// Returns true if this format is quantized (i.e. not F32, F16, or BF16).
    #[must_use]
    pub const fn is_quantized(&self) -> bool {
        !matches!(self, Self::F32 | Self::F16 | Self::BF16)
    }

    /// Returns the required memory alignment in bytes for this format.
    #[must_use]
    pub const fn alignment(&self) -> usize {
        match self {
            Self::F32 => 4,
            Self::F16 | Self::BF16 | Self::Q4_0 | Self::Q8_0 => 2,
            Self::Q4_K_M | Self::Q5_K | Self::Q6_K | Self::TQ1_0 | Self::TQ2_0 => 16,
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

/// Super-block layout for `Q5_K` quantization (GGML v3 / ADR-017).
///
/// Quantizes 256 scalar elements into 176 bytes with 16-byte alignment.
/// Total storage layout (2 + 2 + 12 + 32 + 128 = 176 bytes):
/// - `d`: IEEE 754 half-precision (`f16`) super-block scale factor (bytes 0..2)
/// - `dmin`: IEEE 754 half-precision (`f16`) super-block min factor (bytes 2..4)
/// - `scales`: 12 bytes packed 6-bit scales and minimums for 16 sub-blocks (bytes 4..16)
/// - `qh`: 32 bytes high 1-bit quants for 256 elements (bytes 16..48)
/// - `qs`: 128 bytes low 4-bit quants for 256 elements (bytes 48..176)
#[repr(C, align(16))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlockQ5_K {
    /// Super-block scale for quantized scales (IEEE 754 half-precision `f16` bit pattern).
    pub d: u16,
    /// Super-block scale for quantized mins (IEEE 754 half-precision `f16` bit pattern).
    pub dmin: u16,
    /// 16 sub-block scales and mins packed into 12 bytes using 6 bits each.
    pub scales: [u8; 12],
    /// High 5th bit for all 256 quantized elements (256 bits = 32 bytes).
    pub qh: [u8; 32],
    /// Low 4 bits for all 256 quantized elements (256 nibbles = 128 bytes).
    pub qs: [u8; 128],
}

/// Storage for tensor shape dimensions supporting borrowed slices and inline shapes.
#[derive(Debug, Clone, Copy)]
pub enum TensorShape<'a> {
    /// Borrowed slice from external metadata.
    Borrowed(&'a [usize]),
    /// Inline array for sliced/derived views (up to 8 dimensions).
    Inline([usize; 8], usize),
}

impl TensorShape<'_> {
    /// Returns the shape dimensions as a slice.
    #[must_use]
    pub fn as_slice(&self) -> &[usize] {
        match self {
            Self::Borrowed(s) => s,
            Self::Inline(dims, len) => &dims[..*len],
        }
    }
}

impl core::ops::Deref for TensorShape<'_> {
    type Target = [usize];

    fn deref(&self) -> &Self::Target {
        self.as_slice()
    }
}

impl PartialEq for TensorShape<'_> {
    fn eq(&self, other: &Self) -> bool {
        self.as_slice() == other.as_slice()
    }
}

impl Eq for TensorShape<'_> {}

impl PartialEq<[usize]> for TensorShape<'_> {
    fn eq(&self, other: &[usize]) -> bool {
        self.as_slice() == other
    }
}

impl PartialEq<&[usize]> for TensorShape<'_> {
    fn eq(&self, other: &&[usize]) -> bool {
        self.as_slice() == *other
    }
}

impl fmt::Display for TensorShape<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self.as_slice())
    }
}

/// Zero-copy view into an `mlock`ed weight buffer with verified shape and byte bounds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TensorView<'a> {
    name: &'a str,
    quant: QuantType,
    shape: TensorShape<'a>,
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
            shape: TensorShape::Borrowed(shape),
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
    pub fn shape(&self) -> &[usize] {
        self.shape.as_slice()
    }

    /// Slices a contiguous range of rows from a strictly 2D matrix view `[cols, rows]` without copying.
    ///
    /// Dimension 0 (`shape[0]`) represents the contiguous row elements (columns), while
    /// dimension 1 (`shape[1]`) represents the total row count.
    ///
    /// # Errors
    /// Returns [`K3Error::TensorBoundsError`] if:
    /// - `self.shape.len() != 2`,
    /// - `row_start + row_count > self.shape[1]`,
    /// - `self.shape[0]` is not divisible by the quantization format's `block_elements()`,
    /// - or if byte offset calculations overflow.
    #[expect(
        clippy::too_many_lines,
        reason = "comprehensive bounds, block-alignment, and slice calculations"
    )]
    pub fn slice_rows(
        &self,
        row_start: usize,
        row_count: usize,
    ) -> Result<TensorView<'a>, K3Error> {
        let shape = self.shape();
        if shape.len() != 2 {
            return Err(K3Error::TensorBoundsError {
                tensor_name: self.name.to_owned(),
                offset: self.offset,
                byte_len: self.data.len(),
                buffer_len: self.data.len(),
                reason: format!(
                    "slice_rows requires exactly 2 dimensions [cols, rows], found shape with {} dimension(s)",
                    shape.len()
                ),
            });
        }

        let total_rows = shape[1];
        let end_row =
            row_start
                .checked_add(row_count)
                .ok_or_else(|| K3Error::TensorBoundsError {
                    tensor_name: self.name.to_owned(),
                    offset: self.offset,
                    byte_len: self.data.len(),
                    buffer_len: self.data.len(),
                    reason: "row_start + row_count overflowed usize".to_owned(),
                })?;

        if end_row > total_rows {
            return Err(K3Error::TensorBoundsError {
                tensor_name: self.name.to_owned(),
                offset: self.offset,
                byte_len: self.data.len(),
                buffer_len: self.data.len(),
                reason: format!(
                    "sliced row range [{row_start}..{end_row}] exceeds tensor row count {total_rows}"
                ),
            });
        }

        let row_elements = shape[0];
        let block_elems = self.quant.block_elements();
        if row_elements % block_elems != 0 {
            return Err(K3Error::TensorBoundsError {
                tensor_name: self.name.to_owned(),
                offset: self.offset,
                byte_len: self.data.len(),
                buffer_len: self.data.len(),
                reason: format!(
                    "tensor row length {row_elements} is not divisible by {quant} block size {block_elems}",
                    quant = self.quant.as_str()
                ),
            });
        }

        let row_blocks = row_elements / block_elems;
        let row_bytes = row_blocks
            .checked_mul(self.quant.block_bytes())
            .ok_or_else(|| K3Error::TensorBoundsError {
                tensor_name: self.name.to_owned(),
                offset: self.offset,
                byte_len: self.data.len(),
                buffer_len: self.data.len(),
                reason: "row_bytes calculation overflowed usize".to_owned(),
            })?;

        let start_byte =
            row_start
                .checked_mul(row_bytes)
                .ok_or_else(|| K3Error::TensorBoundsError {
                    tensor_name: self.name.to_owned(),
                    offset: self.offset,
                    byte_len: self.data.len(),
                    buffer_len: self.data.len(),
                    reason: "start_byte calculation overflowed usize".to_owned(),
                })?;

        let slice_byte_len =
            row_count
                .checked_mul(row_bytes)
                .ok_or_else(|| K3Error::TensorBoundsError {
                    tensor_name: self.name.to_owned(),
                    offset: self.offset,
                    byte_len: self.data.len(),
                    buffer_len: self.data.len(),
                    reason: "slice_byte_len calculation overflowed usize".to_owned(),
                })?;

        let end_byte =
            start_byte
                .checked_add(slice_byte_len)
                .ok_or_else(|| K3Error::TensorBoundsError {
                    tensor_name: self.name.to_owned(),
                    offset: self.offset,
                    byte_len: self.data.len(),
                    buffer_len: self.data.len(),
                    reason: "start_byte + slice_byte_len overflowed usize".to_owned(),
                })?;

        if end_byte > self.data.len() {
            return Err(K3Error::TensorBoundsError {
                tensor_name: self.name.to_owned(),
                offset: self.offset,
                byte_len: self.data.len(),
                buffer_len: self.data.len(),
                reason: format!(
                    "sliced byte range [{start_byte}..{end_byte}] exceeds available data length {}",
                    self.data.len()
                ),
            });
        }

        let new_offset =
            self.offset
                .checked_add(start_byte)
                .ok_or_else(|| K3Error::TensorBoundsError {
                    tensor_name: self.name.to_owned(),
                    offset: self.offset,
                    byte_len: self.data.len(),
                    buffer_len: self.data.len(),
                    reason: "offset + start_byte overflowed usize".to_owned(),
                })?;

        let new_dims = [shape[0], row_count, 0, 0, 0, 0, 0, 0];

        let num_elements =
            row_count
                .checked_mul(row_elements)
                .ok_or_else(|| K3Error::TensorBoundsError {
                    tensor_name: self.name.to_owned(),
                    offset: self.offset,
                    byte_len: self.data.len(),
                    buffer_len: self.data.len(),
                    reason: "num_elements calculation overflowed usize".to_owned(),
                })?;

        let sub_data = &self.data[start_byte..end_byte];

        Ok(TensorView {
            name: self.name,
            quant: self.quant,
            shape: TensorShape::Inline(new_dims, 2),
            num_elements,
            offset: new_offset,
            data: sub_data,
        })
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

#[cfg(test)]
mod tests {
    use std::collections::{HashMap, HashSet};

    use super::*;
    use crate::error::K3Error;

    #[test]
    fn test_tq_quant_type_descriptors() {
        // 1. Canonical string identifiers and Display trait
        assert_eq!(QuantType::TQ1_0.as_str(), "TQ1_0");
        assert_eq!(QuantType::TQ1_0.type_name(), "TQ1_0");
        assert_eq!(QuantType::TQ2_0.as_str(), "TQ2_0");
        assert_eq!(QuantType::TQ2_0.type_name(), "TQ2_0");
        assert_eq!(format!("{}", QuantType::TQ1_0), "TQ1_0");
        assert_eq!(format!("{}", QuantType::TQ2_0), "TQ2_0");

        // 2. GGUF type ID mapping
        assert_eq!(QuantType::from_gguf_type_id(34), Some(QuantType::TQ1_0));
        assert_eq!(QuantType::from_gguf_type_id(35), Some(QuantType::TQ2_0));
        assert_eq!(QuantType::TQ1_0.ggml_type_id(), 34);
        assert_eq!(QuantType::TQ2_0.ggml_type_id(), 35);

        // 3. Block size and byte footprint
        assert_eq!(QuantType::TQ1_0.block_elements(), 256);
        assert_eq!(QuantType::TQ2_0.block_elements(), 256);
        assert_eq!(QuantType::TQ1_0.block_bytes(), 54);
        assert_eq!(QuantType::TQ2_0.block_bytes(), 66);

        // 4. Exact block multiples storage calculations
        // 1 block (256 elements)
        assert_eq!(QuantType::TQ1_0.storage_bytes("w", 256).unwrap(), 54);
        assert_eq!(QuantType::TQ2_0.storage_bytes("w", 256).unwrap(), 66);

        // 2 blocks (512 elements)
        assert_eq!(QuantType::TQ1_0.storage_bytes("w", 512).unwrap(), 108);
        assert_eq!(QuantType::TQ2_0.storage_bytes("w", 512).unwrap(), 132);

        // 4 blocks (1024 elements)
        assert_eq!(QuantType::TQ1_0.storage_bytes("w", 1024).unwrap(), 216);
        assert_eq!(QuantType::TQ2_0.storage_bytes("w", 1024).unwrap(), 264);

        // 8 blocks (2048 elements)
        assert_eq!(QuantType::TQ1_0.storage_bytes("w", 2048).unwrap(), 432);
        assert_eq!(QuantType::TQ2_0.storage_bytes("w", 2048).unwrap(), 528);

        // 5. Boundary Condition: Empty tensor (0 elements) must fail
        let err_zero_tq1 = QuantType::TQ1_0.storage_bytes("empty", 0).unwrap_err();
        assert_eq!(
            err_zero_tq1,
            K3Error::TensorBoundsError {
                tensor_name: "empty".to_owned(),
                offset: 0,
                byte_len: 0,
                buffer_len: 0,
                reason: "tensor element count must be greater than zero".to_owned(),
            }
        );
        let err_zero_tq2 = QuantType::TQ2_0.storage_bytes("empty", 0).unwrap_err();
        assert_eq!(
            err_zero_tq2,
            K3Error::TensorBoundsError {
                tensor_name: "empty".to_owned(),
                offset: 0,
                byte_len: 0,
                buffer_len: 0,
                reason: "tensor element count must be greater than zero".to_owned(),
            }
        );

        // 6. Boundary Condition: Non-divisible element counts must fail
        let err_rem_255 = QuantType::TQ1_0
            .storage_bytes("unaligned", 255)
            .unwrap_err();
        assert_eq!(
            err_rem_255,
            K3Error::TensorBoundsError {
                tensor_name: "unaligned".to_owned(),
                offset: 0,
                byte_len: 0,
                buffer_len: 0,
                reason: "element count 255 is not divisible by TQ1_0 block size 256".to_owned(),
            }
        );

        let err_rem_257 = QuantType::TQ1_0
            .storage_bytes("unaligned", 257)
            .unwrap_err();
        assert_eq!(
            err_rem_257,
            K3Error::TensorBoundsError {
                tensor_name: "unaligned".to_owned(),
                offset: 0,
                byte_len: 0,
                buffer_len: 0,
                reason: "element count 257 is not divisible by TQ1_0 block size 256".to_owned(),
            }
        );

        let err_rem_100 = QuantType::TQ2_0
            .storage_bytes("unaligned", 100)
            .unwrap_err();
        assert_eq!(
            err_rem_100,
            K3Error::TensorBoundsError {
                tensor_name: "unaligned".to_owned(),
                offset: 0,
                byte_len: 0,
                buffer_len: 0,
                reason: "element count 100 is not divisible by TQ2_0 block size 256".to_owned(),
            }
        );

        // 7. Boundary Condition: Arithmetic overflow in storage_bytes
        let overflow_elements_f32 = (usize::MAX / 4) + 1;
        let err_of_f32 = QuantType::F32
            .storage_bytes("overflow", overflow_elements_f32)
            .unwrap_err();
        assert!(matches!(err_of_f32, K3Error::TensorBoundsError { .. }));
        assert!(err_of_f32.to_string().contains("byte size overflow"));
    }

    #[test]
    #[allow(clippy::too_many_lines)]
    fn test_tq1_0_tq2_0_tensor_view_bounds() {
        // Storage bytes calculation:
        assert_eq!(QuantType::TQ1_0.storage_bytes("w1", 256).unwrap(), 54);
        assert_eq!(QuantType::TQ1_0.storage_bytes("w1", 512).unwrap(), 108);
        assert_eq!(QuantType::TQ2_0.storage_bytes("w2", 256).unwrap(), 66);
        assert_eq!(QuantType::TQ2_0.storage_bytes("w2", 512).unwrap(), 132);

        // Divisibility rejection:
        let err_tq1 = QuantType::TQ1_0.storage_bytes("w1", 255);
        assert!(matches!(err_tq1, Err(K3Error::TensorBoundsError { .. })));
        assert!(
            err_tq1
                .unwrap_err()
                .to_string()
                .contains("not divisible by TQ1_0 block size 256")
        );

        let err_tq2 = QuantType::TQ2_0.storage_bytes("w2", 300);
        assert!(matches!(err_tq2, Err(K3Error::TensorBoundsError { .. })));
        assert!(
            err_tq2
                .unwrap_err()
                .to_string()
                .contains("not divisible by TQ2_0 block size 256")
        );

        // Zero elements:
        assert!(matches!(
            QuantType::TQ1_0.storage_bytes("w", 0),
            Err(K3Error::TensorBoundsError { .. })
        ));
        assert!(matches!(
            QuantType::TQ2_0.storage_bytes("w", 0),
            Err(K3Error::TensorBoundsError { .. })
        ));

        // Backing buffer with 1 block TQ1_0 (54B) + 1 block TQ2_0 (66B) = 120B total
        let backing = vec![0x33_u8; 120];

        // 1. Valid TQ1_0 view: 256 elements -> 54 bytes at offset 0
        let view_tq1 =
            TensorView::from_bytes("blk.0.attn_q.weight", QuantType::TQ1_0, &[256], 0, &backing)
                .expect("valid TQ1_0 bounds");
        assert_eq!(view_tq1.name(), "blk.0.attn_q.weight");
        assert_eq!(view_tq1.quant(), QuantType::TQ1_0);
        assert_eq!(view_tq1.shape(), &[256]);
        assert_eq!(view_tq1.num_elements(), 256);
        assert_eq!(view_tq1.offset(), 0);
        assert_eq!(view_tq1.byte_len(), 54);
        assert_eq!(view_tq1.as_bytes().as_ptr(), backing.as_ptr());

        // 2. Valid TQ2_0 view: 256 elements -> 66 bytes at offset 54 (54 + 66 == 120)
        let view_tq2 = TensorView::from_bytes(
            "blk.0.attn_k.weight",
            QuantType::TQ2_0,
            &[256],
            54,
            &backing,
        )
        .expect("valid TQ2_0 bounds at end of buffer");
        assert_eq!(view_tq2.name(), "blk.0.attn_k.weight");
        assert_eq!(view_tq2.quant(), QuantType::TQ2_0);
        assert_eq!(view_tq2.shape(), &[256]);
        assert_eq!(view_tq2.num_elements(), 256);
        assert_eq!(view_tq2.offset(), 54);
        assert_eq!(view_tq2.byte_len(), 66);
        // SAFETY: 54 is strictly within backing buffer length 120.
        let expected_tq2_ptr = unsafe { backing.as_ptr().add(54) };
        assert_eq!(view_tq2.as_bytes().as_ptr(), expected_tq2_ptr);

        // 3. Out-of-bounds by exactly 1 byte
        let oob_tq1 = TensorView::from_bytes(
            "blk.0.attn_q.weight",
            QuantType::TQ1_0,
            &[256],
            67, // 67 + 54 = 121 > 120
            &backing,
        );
        assert_eq!(
            oob_tq1,
            Err(K3Error::TensorBoundsError {
                tensor_name: "blk.0.attn_q.weight".to_owned(),
                offset: 67,
                byte_len: 54,
                buffer_len: 120,
                reason: "tensor range [67..121] exceeds backing buffer length 120".to_owned(),
            })
        );

        let oob_tq2 = TensorView::from_bytes(
            "blk.0.attn_k.weight",
            QuantType::TQ2_0,
            &[256],
            55, // 55 + 66 = 121 > 120
            &backing,
        );
        assert_eq!(
            oob_tq2,
            Err(K3Error::TensorBoundsError {
                tensor_name: "blk.0.attn_k.weight".to_owned(),
                offset: 55,
                byte_len: 66,
                buffer_len: 120,
                reason: "tensor range [55..121] exceeds backing buffer length 120".to_owned(),
            })
        );

        // 4. Empty shape rejected
        let empty_shape = TensorView::from_bytes("empty", QuantType::TQ1_0, &[], 0, &backing);
        assert_eq!(
            empty_shape,
            Err(K3Error::TensorBoundsError {
                tensor_name: "empty".to_owned(),
                offset: 0,
                byte_len: 0,
                buffer_len: 120,
                reason: "tensor shape must contain at least one dimension".to_owned(),
            })
        );

        // 5. Zero dimension rejected
        let zero_dim = TensorView::from_bytes("zero", QuantType::TQ1_0, &[256, 0], 0, &backing);
        assert_eq!(
            zero_dim,
            Err(K3Error::TensorBoundsError {
                tensor_name: "zero".to_owned(),
                offset: 0,
                byte_len: 0,
                buffer_len: 120,
                reason: "tensor shape dimensions must be non-zero".to_owned(),
            })
        );

        // 6. Non-block-divisible shape rejected
        let unaligned_shape =
            TensorView::from_bytes("unaligned", QuantType::TQ1_0, &[250], 0, &backing);
        assert_eq!(
            unaligned_shape,
            Err(K3Error::TensorBoundsError {
                tensor_name: "unaligned".to_owned(),
                offset: 0,
                byte_len: 0,
                buffer_len: 120,
                reason: "element count 250 is not divisible by TQ1_0 block size 256".to_owned(),
            })
        );

        // 7. Offset overflow rejected
        let of_offset = TensorView::from_bytes(
            "of_offset",
            QuantType::TQ1_0,
            &[256],
            usize::MAX - 20,
            &backing,
        );
        assert_eq!(
            of_offset,
            Err(K3Error::TensorBoundsError {
                tensor_name: "of_offset".to_owned(),
                offset: usize::MAX - 20,
                byte_len: 54,
                buffer_len: 120,
                reason: "tensor offset + byte_len overflowed usize".to_owned(),
            })
        );

        // 8. Zero scale buffer (all 0x00 bytes) accepts cleanly without division by zero
        let zero_scale_backing = vec![0x00_u8; 66];
        let view_zero = TensorView::from_bytes(
            "zero_scale",
            QuantType::TQ2_0,
            &[256],
            0,
            &zero_scale_backing,
        )
        .expect("all-zero / zero-scale buffer is accepted as valid tensor bytes");
        assert_eq!(view_zero.byte_len(), 66);
        assert_eq!(view_zero.as_bytes()[0], 0);

        // 9. Zero-copy slice_rows on 2D TQ1_0 matrix: 4 rows of 256 elements = 4 * 54 = 216 bytes
        let mut matrix_backing = vec![0x11_u8; 216];
        matrix_backing[108..].fill(0x22);

        let view_matrix_tq1 = TensorView::from_bytes(
            "blk.0.attn_qkv.weight",
            QuantType::TQ1_0,
            &[256, 4],
            0,
            &matrix_backing,
        )
        .expect("valid 2D matrix");

        // Slice rows 0..2 (first partition: 512 elements, 108 bytes)
        let slice_q = view_matrix_tq1
            .slice_rows(0, 2)
            .expect("slice 0..2 succeeds");
        assert_eq!(slice_q.shape(), &[256, 2]);
        assert_eq!(slice_q.num_elements(), 512);
        assert_eq!(slice_q.offset(), 0);
        assert_eq!(slice_q.byte_len(), 108);
        assert_eq!(slice_q.as_bytes()[0], 0x11);

        // Slice rows 2..2 (second partition: 512 elements, 108 bytes at offset 108)
        let slice_kv = view_matrix_tq1
            .slice_rows(2, 2)
            .expect("slice 2..2 succeeds");
        assert_eq!(slice_kv.shape(), &[256, 2]);
        assert_eq!(slice_kv.num_elements(), 512);
        assert_eq!(slice_kv.offset(), 108);
        assert_eq!(slice_kv.byte_len(), 108);
        assert_eq!(slice_kv.as_bytes()[0], 0x22);

        // Slice 0 rows
        let slice_zero = view_matrix_tq1
            .slice_rows(1, 0)
            .expect("0-row slice succeeds");
        assert_eq!(slice_zero.shape(), &[256, 0]);
        assert_eq!(slice_zero.num_elements(), 0);
        assert_eq!(slice_zero.byte_len(), 0);
        assert_eq!(slice_zero.offset(), 54);

        // Row range overflow rejection
        let err_oob_rows = view_matrix_tq1.slice_rows(3, 2).unwrap_err();
        assert!(matches!(err_oob_rows, K3Error::TensorBoundsError { .. }));
        assert!(
            err_oob_rows
                .to_string()
                .contains("exceeds tensor row count 4")
        );

        // Column unaligned rejection in slice_rows:
        // [128, 2] has total 256 elements (so calculate_bounds succeeds),
        // but row length 128 is not divisible by block size 256!
        let unaligned_col_backing = vec![0u8; 1000];
        let view_unaligned_cols = TensorView::from_bytes(
            "unaligned_cols",
            QuantType::TQ1_0,
            &[128, 2],
            0,
            &unaligned_col_backing,
        )
        .expect("total elements 256 divides 256");
        let err_align = view_unaligned_cols.slice_rows(0, 1).unwrap_err();
        assert_eq!(
            err_align,
            K3Error::TensorBoundsError {
                tensor_name: "unaligned_cols".to_owned(),
                offset: 0,
                byte_len: 54,
                buffer_len: 54,
                reason: "tensor row length 128 is not divisible by TQ1_0 block size 256".to_owned(),
            }
        );
    }

    #[test]
    fn test_tq1_0_and_tq2_0_quant_type() {
        // 1. Trait conformance: Copy, Clone, PartialEq, Eq
        let tq1_a = QuantType::TQ1_0;
        let tq1_b = tq1_a;
        assert_eq!(tq1_a, tq1_b);
        assert_ne!(QuantType::TQ1_0, QuantType::TQ2_0);

        // 2. Hash trait: Works as key in HashSet and HashMap
        let mut set = HashSet::new();
        assert!(set.insert(QuantType::TQ1_0));
        assert!(set.insert(QuantType::TQ2_0));
        assert!(!set.insert(QuantType::TQ1_0));
        assert_eq!(set.len(), 2);

        let mut map = HashMap::new();
        map.insert(QuantType::TQ1_0, "1.58-bit ternary");
        map.insert(QuantType::TQ2_0, "2-bit ternary");
        assert_eq!(map.get(&QuantType::TQ1_0), Some(&"1.58-bit ternary"));
        assert_eq!(map.get(&QuantType::TQ2_0), Some(&"2-bit ternary"));

        // 3. Exhaustive GGUF type ID mapping verification
        let expected_mappings = [
            (0, Some(QuantType::F32)),
            (1, Some(QuantType::F16)),
            (2, Some(QuantType::Q4_0)),
            (8, Some(QuantType::Q8_0)),
            (12, Some(QuantType::Q4_K_M)),
            (13, Some(QuantType::Q5_K)),
            (14, Some(QuantType::Q6_K)),
            (30, Some(QuantType::BF16)),
            (34, Some(QuantType::TQ1_0)),
            (35, Some(QuantType::TQ2_0)),
            (3, None),
            (4, None),
            (31, None),
            (33, None),
            (36, None),
            (999, None),
        ];
        for (id, expected) in expected_mappings {
            assert_eq!(
                QuantType::from_gguf_type_id(id),
                expected,
                "ID {id} mapping mismatch"
            );
        }

        // 4. Multi-block scaling linearity loop (1..=16 blocks)
        for n_blocks in 1..=16 {
            let num_elements = n_blocks * 256;
            let tq1_bytes = QuantType::TQ1_0
                .storage_bytes("loop", num_elements)
                .unwrap();
            let tq2_bytes = QuantType::TQ2_0
                .storage_bytes("loop", num_elements)
                .unwrap();
            assert_eq!(tq1_bytes, n_blocks * 54);
            assert_eq!(tq2_bytes, n_blocks * 66);
        }

        // 5. Non-divisible remainder rejections across various non-zero deltas
        for delta in [1, 7, 15, 31, 63, 127, 255] {
            let num_elements = 256 + delta;
            assert!(matches!(
                QuantType::TQ1_0.storage_bytes("rem", num_elements),
                Err(K3Error::TensorBoundsError { .. })
            ));
            assert!(matches!(
                QuantType::TQ2_0.storage_bytes("rem", num_elements),
                Err(K3Error::TensorBoundsError { .. })
            ));
        }

        // 6. Bits per weight verification (mathematical properties)
        // TQ1_0: 54 bytes = 432 bits for 256 elements = 1.6875 bpw (1.58-bit ternary + 16-bit f16 scale)
        #[allow(clippy::cast_precision_loss)]
        let tq1_bpw =
            (QuantType::TQ1_0.block_bytes() * 8) as f64 / QuantType::TQ1_0.block_elements() as f64;
        assert!((tq1_bpw - 1.6875).abs() < 1e-6);

        // TQ2_0: 66 bytes = 528 bits for 256 elements = 2.0625 bpw (2-bit ternary + 16-bit f16 scale)
        #[allow(clippy::cast_precision_loss)]
        let tq2_bpw =
            (QuantType::TQ2_0.block_bytes() * 8) as f64 / QuantType::TQ2_0.block_elements() as f64;
        assert!((tq2_bpw - 2.0625).abs() < 1e-6);
    }

    #[test]
    fn test_q5_k_quant_descriptor() {
        let q = QuantType::Q5_K;

        // 1. Variant, Display, and String Identifiers
        assert_eq!(q.as_str(), "Q5_K");
        assert_eq!(q.type_name(), "Q5_K");
        assert_eq!(format!("{q}"), "Q5_K");

        // 2. GGUF type ID mapping
        assert_eq!(QuantType::from_gguf_type_id(13), Some(QuantType::Q5_K));
        assert_eq!(q.ggml_type_id(), 13);
        assert_eq!(q.ggml_type(), 13);

        // 3. Block elements, storage sizes, and helpers
        assert_eq!(q.block_elements(), 256);
        assert_eq!(q.block_size(), 256);
        assert_eq!(q.block_bytes(), 176);
        assert_eq!(q.type_size(), 176);
        assert!(q.is_quantized());
        assert_eq!(q.alignment(), 16);

        // 4. Exact block multiples storage calculations
        assert_eq!(q.storage_bytes("w", 256).unwrap(), 176);
        assert_eq!(q.storage_bytes("w", 512).unwrap(), 352);
        assert_eq!(q.storage_bytes("w", 1024).unwrap(), 704);
        assert_eq!(q.storage_bytes("w", 2048).unwrap(), 1408);

        // 5. Misaligned element count rejections
        let err_unaligned = q.storage_bytes("unaligned", 255).unwrap_err();
        assert_eq!(
            err_unaligned,
            K3Error::TensorBoundsError {
                tensor_name: "unaligned".to_owned(),
                offset: 0,
                byte_len: 0,
                buffer_len: 0,
                reason: "element count 255 is not divisible by Q5_K block size 256".to_owned(),
            }
        );

        // 6. Zero elements rejection
        let err_zero = q.storage_bytes("empty", 0).unwrap_err();
        assert_eq!(
            err_zero,
            K3Error::TensorBoundsError {
                tensor_name: "empty".to_owned(),
                offset: 0,
                byte_len: 0,
                buffer_len: 0,
                reason: "tensor element count must be greater than zero".to_owned(),
            }
        );

        // 7. Struct memory layout & alignment
        assert_eq!(size_of::<BlockQ5_K>(), 176);
        assert_eq!(align_of::<BlockQ5_K>(), 16);
        assert_eq!(core::mem::offset_of!(BlockQ5_K, d), 0);
        assert_eq!(core::mem::offset_of!(BlockQ5_K, dmin), 2);
        assert_eq!(core::mem::offset_of!(BlockQ5_K, scales), 4);
        assert_eq!(core::mem::offset_of!(BlockQ5_K, qh), 16);
        assert_eq!(core::mem::offset_of!(BlockQ5_K, qs), 48);

        // 8. 2D TensorView row slicing
        let mut backing = vec![0xAA_u8; 704];
        for r in 0..4_usize {
            backing[r * 176] = u8::try_from(r).expect("r in 0..4 fits u8");
        }
        let view = TensorView::from_bytes("matrix_q5k", QuantType::Q5_K, &[256, 4], 0, &backing)
            .expect("valid 2D Q5_K matrix");
        for r in 0..4_usize {
            let slice = view.slice_rows(r, 1).expect("valid row slice");
            assert_eq!(slice.shape(), &[256, 1]);
            assert_eq!(slice.num_elements(), 256);
            assert_eq!(slice.offset(), r * 176);
            assert_eq!(slice.byte_len(), 176);
            assert_eq!(
                slice.as_bytes()[0],
                u8::try_from(r).expect("r in 0..4 fits u8")
            );
        }
    }
}
