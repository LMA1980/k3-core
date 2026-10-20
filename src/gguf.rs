//! Zero-copy GGUF v3 binary parser, metadata extraction, and tensor descriptor mapping.
//!
//! Conforms to the GGUF v3 specification and enforces upfront architecture and quantization
//! capability verification (ADR-017, ADR-023, ADR-027, ADR-037).

use core::fmt;
use std::collections::BTreeMap;

use crate::error::K3Error;
use crate::tensor::{ModelWeights, QuantType, TensorView};

/// Magic bytes identifying a valid GGUF file (`"GGUF"` in ASCII, little-endian: `0x46554747`).
pub const GGUF_MAGIC: u32 = 0x4655_4747;

/// Supported GGUF binary specification version (v3).
pub const GGUF_VERSION_V3: u32 = 3;

/// Default tensor data alignment in bytes if `general.alignment` is omitted.
pub const DEFAULT_ALIGNMENT: usize = 32;

/// GGUF metadata value types per specification.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum GgufMetadataValueType {
    /// 8-bit unsigned integer.
    Uint8 = 0,
    /// 8-bit signed integer.
    Int8 = 1,
    /// 16-bit unsigned integer (little-endian).
    Uint16 = 2,
    /// 16-bit signed integer (little-endian).
    Int16 = 3,
    /// 32-bit unsigned integer (little-endian).
    Uint32 = 4,
    /// 32-bit signed integer (little-endian).
    Int32 = 5,
    /// 32-bit IEEE 754 float (little-endian).
    Float32 = 6,
    /// 1-byte boolean flag.
    Bool = 7,
    /// Length-prefixed UTF-8 string.
    String = 8,
    /// Homogeneous array of elements.
    Array = 9,
    /// 64-bit unsigned integer (little-endian).
    Uint64 = 10,
    /// 64-bit signed integer (little-endian).
    Int64 = 11,
    /// 64-bit IEEE 754 float (little-endian).
    Float64 = 12,
}

impl GgufMetadataValueType {
    /// Maps a raw u32 integer to [`GgufMetadataValueType`].
    #[must_use]
    pub const fn from_u32(val: u32) -> Option<Self> {
        match val {
            0 => Some(Self::Uint8),
            1 => Some(Self::Int8),
            2 => Some(Self::Uint16),
            3 => Some(Self::Int16),
            4 => Some(Self::Uint32),
            5 => Some(Self::Int32),
            6 => Some(Self::Float32),
            7 => Some(Self::Bool),
            8 => Some(Self::String),
            9 => Some(Self::Array),
            10 => Some(Self::Uint64),
            11 => Some(Self::Int64),
            12 => Some(Self::Float64),
            _ => None,
        }
    }
}

/// Zero-copy parsed GGUF metadata value.
#[derive(Debug, Clone, PartialEq)]
pub enum GgufValue<'a> {
    /// 8-bit unsigned integer.
    Uint8(u8),
    /// 8-bit signed integer.
    Int8(i8),
    /// 16-bit unsigned integer.
    Uint16(u16),
    /// 16-bit signed integer.
    Int16(i16),
    /// 32-bit unsigned integer.
    Uint32(u32),
    /// 32-bit signed integer.
    Int32(i32),
    /// 32-bit IEEE float.
    Float32(f32),
    /// 1-byte boolean flag.
    Bool(bool),
    /// Borrowed UTF-8 string slice.
    String(&'a str),
    /// Homogeneous array of values.
    Array(GgufArray<'a>),
    /// 64-bit unsigned integer.
    Uint64(u64),
    /// 64-bit signed integer.
    Int64(i64),
    /// 64-bit IEEE float.
    Float64(f64),
}

impl<'a> GgufValue<'a> {
    /// Returns the value as a borrowed string slice if it is [`GgufValue::String`].
    #[must_use]
    pub const fn as_str(&self) -> Option<&'a str> {
        match self {
            Self::String(s) => Some(s),
            _ => None,
        }
    }

    /// Returns the value as `u64` if it is an unsigned or non-negative signed integer.
    #[must_use]
    pub fn as_u64(&self) -> Option<u64> {
        match self {
            Self::Uint8(v) => Some(u64::from(*v)),
            Self::Uint16(v) => Some(u64::from(*v)),
            Self::Uint32(v) => Some(u64::from(*v)),
            Self::Uint64(v) => Some(*v),
            Self::Int8(v) => u64::try_from(*v).ok(),
            Self::Int16(v) => u64::try_from(*v).ok(),
            Self::Int32(v) => u64::try_from(*v).ok(),
            Self::Int64(v) => u64::try_from(*v).ok(),
            _ => None,
        }
    }

    /// Returns the value as `usize` if convertible.
    #[must_use]
    pub fn as_usize(&self) -> Option<usize> {
        self.as_u64().and_then(|v| usize::try_from(v).ok())
    }

    /// Returns the value as a boolean if it is [`GgufValue::Bool`].
    #[must_use]
    pub const fn as_bool(&self) -> Option<bool> {
        match self {
            Self::Bool(b) => Some(*b),
            _ => None,
        }
    }

    /// Returns the value as `f32` if it is a floating-point number.
    #[must_use]
    pub fn as_f32(&self) -> Option<f32> {
        match self {
            Self::Float32(f) => Some(*f),
            #[allow(clippy::cast_possible_truncation)]
            Self::Float64(f) => Some(*f as f32),
            _ => None,
        }
    }
}

/// Homogeneous array parsed from a GGUF metadata value.
#[derive(Debug, Clone, PartialEq)]
pub enum GgufArray<'a> {
    /// Array of 8-bit unsigned integers.
    Uint8(Vec<u8>),
    /// Array of 8-bit signed integers.
    Int8(Vec<i8>),
    /// Array of 16-bit unsigned integers.
    Uint16(Vec<u16>),
    /// Array of 16-bit signed integers.
    Int16(Vec<i16>),
    /// Array of 32-bit unsigned integers.
    Uint32(Vec<u32>),
    /// Array of 32-bit signed integers.
    Int32(Vec<i32>),
    /// Array of 32-bit floats.
    Float32(Vec<f32>),
    /// Array of booleans.
    Bool(Vec<bool>),
    /// Array of borrowed UTF-8 strings.
    String(Vec<&'a str>),
    /// Array of 64-bit unsigned integers.
    Uint64(Vec<u64>),
    /// Array of 64-bit signed integers.
    Int64(Vec<i64>),
    /// Array of 64-bit floats.
    Float64(Vec<f64>),
}

/// Internal helper for zero-copy reading from a byte slice.
struct BinaryCursor<'a> {
    buffer: &'a [u8],
    position: usize,
}

impl<'a> BinaryCursor<'a> {
    const fn new(buffer: &'a [u8]) -> Self {
        Self {
            buffer,
            position: 0,
        }
    }

    const fn position(&self) -> usize {
        self.position
    }

    fn remaining_bytes(&self) -> usize {
        self.buffer.len().saturating_sub(self.position)
    }

    fn read_slice(&mut self, len: usize) -> Result<&'a [u8], K3Error> {
        let end = self
            .position
            .checked_add(len)
            .ok_or_else(|| K3Error::TensorBoundsError {
                tensor_name: "header".to_owned(),
                offset: self.position,
                byte_len: len,
                buffer_len: self.buffer.len(),
                reason: "slice read offset overflow".to_owned(),
            })?;

        if end > self.buffer.len() {
            return Err(K3Error::TensorBoundsError {
                tensor_name: "header".to_owned(),
                offset: self.position,
                byte_len: len,
                buffer_len: self.buffer.len(),
                reason: format!(
                    "unexpected EOF: requested {len} bytes, remaining {}",
                    self.remaining_bytes()
                ),
            });
        }

        let slice = &self.buffer[self.position..end];
        self.position = end;
        Ok(slice)
    }

    fn read_u8(&mut self) -> Result<u8, K3Error> {
        let slice = self.read_slice(1)?;
        Ok(slice[0])
    }

    fn read_i8(&mut self) -> Result<i8, K3Error> {
        let u = self.read_u8()?;
        #[allow(clippy::cast_possible_wrap)]
        Ok(u as i8)
    }

    fn read_u16(&mut self) -> Result<u16, K3Error> {
        let slice = self.read_slice(2)?;
        let bytes: [u8; 2] = slice.try_into().map_err(|_| K3Error::TensorBoundsError {
            tensor_name: "header".to_owned(),
            offset: self.position,
            byte_len: 2,
            buffer_len: self.buffer.len(),
            reason: "slice conversion error".to_owned(),
        })?;
        Ok(u16::from_le_bytes(bytes))
    }

    fn read_i16(&mut self) -> Result<i16, K3Error> {
        let slice = self.read_slice(2)?;
        let bytes: [u8; 2] = slice.try_into().map_err(|_| K3Error::TensorBoundsError {
            tensor_name: "header".to_owned(),
            offset: self.position,
            byte_len: 2,
            buffer_len: self.buffer.len(),
            reason: "slice conversion error".to_owned(),
        })?;
        Ok(i16::from_le_bytes(bytes))
    }

    fn read_u32(&mut self) -> Result<u32, K3Error> {
        let slice = self.read_slice(4)?;
        let bytes: [u8; 4] = slice.try_into().map_err(|_| K3Error::TensorBoundsError {
            tensor_name: "header".to_owned(),
            offset: self.position,
            byte_len: 4,
            buffer_len: self.buffer.len(),
            reason: "slice conversion error".to_owned(),
        })?;
        Ok(u32::from_le_bytes(bytes))
    }

    fn read_i32(&mut self) -> Result<i32, K3Error> {
        let slice = self.read_slice(4)?;
        let bytes: [u8; 4] = slice.try_into().map_err(|_| K3Error::TensorBoundsError {
            tensor_name: "header".to_owned(),
            offset: self.position,
            byte_len: 4,
            buffer_len: self.buffer.len(),
            reason: "slice conversion error".to_owned(),
        })?;
        Ok(i32::from_le_bytes(bytes))
    }

    fn read_u64(&mut self) -> Result<u64, K3Error> {
        let slice = self.read_slice(8)?;
        let bytes: [u8; 8] = slice.try_into().map_err(|_| K3Error::TensorBoundsError {
            tensor_name: "header".to_owned(),
            offset: self.position,
            byte_len: 8,
            buffer_len: self.buffer.len(),
            reason: "slice conversion error".to_owned(),
        })?;
        Ok(u64::from_le_bytes(bytes))
    }

    fn read_i64(&mut self) -> Result<i64, K3Error> {
        let slice = self.read_slice(8)?;
        let bytes: [u8; 8] = slice.try_into().map_err(|_| K3Error::TensorBoundsError {
            tensor_name: "header".to_owned(),
            offset: self.position,
            byte_len: 8,
            buffer_len: self.buffer.len(),
            reason: "slice conversion error".to_owned(),
        })?;
        Ok(i64::from_le_bytes(bytes))
    }

    fn read_f32(&mut self) -> Result<f32, K3Error> {
        let slice = self.read_slice(4)?;
        let bytes: [u8; 4] = slice.try_into().map_err(|_| K3Error::TensorBoundsError {
            tensor_name: "header".to_owned(),
            offset: self.position,
            byte_len: 4,
            buffer_len: self.buffer.len(),
            reason: "slice conversion error".to_owned(),
        })?;
        Ok(f32::from_le_bytes(bytes))
    }

    fn read_f64(&mut self) -> Result<f64, K3Error> {
        let slice = self.read_slice(8)?;
        let bytes: [u8; 8] = slice.try_into().map_err(|_| K3Error::TensorBoundsError {
            tensor_name: "header".to_owned(),
            offset: self.position,
            byte_len: 8,
            buffer_len: self.buffer.len(),
            reason: "slice conversion error".to_owned(),
        })?;
        Ok(f64::from_le_bytes(bytes))
    }

    fn read_bool(&mut self) -> Result<bool, K3Error> {
        let b = self.read_u8()?;
        match b {
            0 => Ok(false),
            1 => Ok(true),
            other => Err(K3Error::TensorBoundsError {
                tensor_name: "header".to_owned(),
                offset: self.position.saturating_sub(1),
                byte_len: 1,
                buffer_len: self.buffer.len(),
                reason: format!("invalid boolean byte value {other}"),
            }),
        }
    }

    fn read_string(&mut self) -> Result<&'a str, K3Error> {
        let len_u64 = self.read_u64()?;
        let len = usize::try_from(len_u64).map_err(|_| K3Error::TensorBoundsError {
            tensor_name: "header".to_owned(),
            offset: self.position,
            byte_len: 8,
            buffer_len: self.buffer.len(),
            reason: format!("string length {len_u64} overflows usize"),
        })?;

        let bytes = self.read_slice(len)?;
        core::str::from_utf8(bytes).map_err(|utf_err| K3Error::TensorBoundsError {
            tensor_name: "header".to_owned(),
            offset: self.position.saturating_sub(len),
            byte_len: len,
            buffer_len: self.buffer.len(),
            reason: format!("invalid UTF-8 in GGUF string: {utf_err}"),
        })
    }

    fn read_value(&mut self) -> Result<GgufValue<'a>, K3Error> {
        let type_raw = self.read_u32()?;
        let value_type = GgufMetadataValueType::from_u32(type_raw).ok_or_else(|| {
            K3Error::TensorBoundsError {
                tensor_name: "header".to_owned(),
                offset: self.position.saturating_sub(4),
                byte_len: 4,
                buffer_len: self.buffer.len(),
                reason: format!("unsupported GGUF metadata value type {type_raw}"),
            }
        })?;

        self.read_value_of_type(value_type)
    }

    fn read_value_of_type(
        &mut self,
        val_type: GgufMetadataValueType,
    ) -> Result<GgufValue<'a>, K3Error> {
        match val_type {
            GgufMetadataValueType::Uint8 => self.read_u8().map(GgufValue::Uint8),
            GgufMetadataValueType::Int8 => self.read_i8().map(GgufValue::Int8),
            GgufMetadataValueType::Uint16 => self.read_u16().map(GgufValue::Uint16),
            GgufMetadataValueType::Int16 => self.read_i16().map(GgufValue::Int16),
            GgufMetadataValueType::Uint32 => self.read_u32().map(GgufValue::Uint32),
            GgufMetadataValueType::Int32 => self.read_i32().map(GgufValue::Int32),
            GgufMetadataValueType::Float32 => self.read_f32().map(GgufValue::Float32),
            GgufMetadataValueType::Bool => self.read_bool().map(GgufValue::Bool),
            GgufMetadataValueType::String => self.read_string().map(GgufValue::String),
            GgufMetadataValueType::Uint64 => self.read_u64().map(GgufValue::Uint64),
            GgufMetadataValueType::Int64 => self.read_i64().map(GgufValue::Int64),
            GgufMetadataValueType::Float64 => self.read_f64().map(GgufValue::Float64),
            GgufMetadataValueType::Array => {
                let elem_type_raw = self.read_u32()?;
                let elem_type =
                    GgufMetadataValueType::from_u32(elem_type_raw).ok_or_else(|| {
                        K3Error::TensorBoundsError {
                            tensor_name: "header".to_owned(),
                            offset: self.position.saturating_sub(4),
                            byte_len: 4,
                            buffer_len: self.buffer.len(),
                            reason: format!("unsupported GGUF array element type {elem_type_raw}"),
                        }
                    })?;

                let count_u64 = self.read_u64()?;
                let count = usize::try_from(count_u64).map_err(|_| K3Error::TensorBoundsError {
                    tensor_name: "header".to_owned(),
                    offset: self.position,
                    byte_len: 8,
                    buffer_len: self.buffer.len(),
                    reason: format!("array count {count_u64} overflows usize"),
                })?;

                let arr = self.read_array_elements(elem_type, count)?;
                Ok(GgufValue::Array(arr))
            }
        }
    }

    fn read_array_elements(
        &mut self,
        elem_type: GgufMetadataValueType,
        count: usize,
    ) -> Result<GgufArray<'a>, K3Error> {
        match elem_type {
            GgufMetadataValueType::Uint8 => {
                let mut v = Vec::with_capacity(count);
                for _ in 0..count {
                    v.push(self.read_u8()?);
                }
                Ok(GgufArray::Uint8(v))
            }
            GgufMetadataValueType::Int8 => {
                let mut v = Vec::with_capacity(count);
                for _ in 0..count {
                    v.push(self.read_i8()?);
                }
                Ok(GgufArray::Int8(v))
            }
            GgufMetadataValueType::Uint16 => {
                let mut v = Vec::with_capacity(count);
                for _ in 0..count {
                    v.push(self.read_u16()?);
                }
                Ok(GgufArray::Uint16(v))
            }
            GgufMetadataValueType::Int16 => {
                let mut v = Vec::with_capacity(count);
                for _ in 0..count {
                    v.push(self.read_i16()?);
                }
                Ok(GgufArray::Int16(v))
            }
            GgufMetadataValueType::Uint32 => {
                let mut v = Vec::with_capacity(count);
                for _ in 0..count {
                    v.push(self.read_u32()?);
                }
                Ok(GgufArray::Uint32(v))
            }
            GgufMetadataValueType::Int32 => {
                let mut v = Vec::with_capacity(count);
                for _ in 0..count {
                    v.push(self.read_i32()?);
                }
                Ok(GgufArray::Int32(v))
            }
            GgufMetadataValueType::Float32 => {
                let mut v = Vec::with_capacity(count);
                for _ in 0..count {
                    v.push(self.read_f32()?);
                }
                Ok(GgufArray::Float32(v))
            }
            GgufMetadataValueType::Bool => {
                let mut v = Vec::with_capacity(count);
                for _ in 0..count {
                    v.push(self.read_bool()?);
                }
                Ok(GgufArray::Bool(v))
            }
            GgufMetadataValueType::String => {
                let mut v = Vec::with_capacity(count);
                for _ in 0..count {
                    v.push(self.read_string()?);
                }
                Ok(GgufArray::String(v))
            }
            GgufMetadataValueType::Uint64 => {
                let mut v = Vec::with_capacity(count);
                for _ in 0..count {
                    v.push(self.read_u64()?);
                }
                Ok(GgufArray::Uint64(v))
            }
            GgufMetadataValueType::Int64 => {
                let mut v = Vec::with_capacity(count);
                for _ in 0..count {
                    v.push(self.read_i64()?);
                }
                Ok(GgufArray::Int64(v))
            }
            GgufMetadataValueType::Float64 => {
                let mut v = Vec::with_capacity(count);
                for _ in 0..count {
                    v.push(self.read_f64()?);
                }
                Ok(GgufArray::Float64(v))
            }
            GgufMetadataValueType::Array => Err(K3Error::TensorBoundsError {
                tensor_name: "header".to_owned(),
                offset: self.position,
                byte_len: 0,
                buffer_len: self.buffer.len(),
                reason: "nested GGUF arrays are not supported".to_owned(),
            }),
        }
    }
}

/// Aligns `offset` up to the next multiple of `alignment`.
#[must_use]
pub const fn align_up(offset: usize, alignment: usize) -> usize {
    if alignment <= 1 {
        return offset;
    }
    let rem = offset % alignment;
    if rem == 0 {
        offset
    } else {
        offset + (alignment - rem)
    }
}

/// Zero-copy tensor descriptor within a GGUF file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorDescriptor {
    name: String,
    quant: QuantType,
    shape: Vec<usize>,
    num_elements: usize,
    relative_offset: usize,
    absolute_offset: usize,
    byte_len: usize,
}

impl TensorDescriptor {
    /// Returns the tensor identifier name.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Returns the tensor's quantization type.
    #[must_use]
    pub const fn quant(&self) -> QuantType {
        self.quant
    }

    /// Returns the shape dimension slice.
    #[must_use]
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Returns the total scalar element count.
    #[must_use]
    pub const fn num_elements(&self) -> usize {
        self.num_elements
    }

    /// Returns the byte offset relative to the model tensor data section.
    #[must_use]
    pub const fn relative_offset(&self) -> usize {
        self.relative_offset
    }

    /// Returns the absolute byte offset within the backing buffer/file.
    #[must_use]
    pub const fn absolute_offset(&self) -> usize {
        self.absolute_offset
    }

    /// Returns the exact byte length of the tensor's stored data.
    #[must_use]
    pub const fn byte_len(&self) -> usize {
        self.byte_len
    }
}

/// Parsed model hyperparameters and structural metadata from a GGUF v3 file.
#[derive(Debug, Clone, PartialEq)]
pub struct ModelMetadata {
    architecture: String,
    layer_count: usize,
    embedding_dim: usize,
    head_count: usize,
    head_count_kv: usize,
    context_length: usize,
    alignment: usize,
    tensor_count: usize,
    total_weight_bytes: usize,
    tensors: Vec<TensorDescriptor>,
}

impl ModelMetadata {
    /// Returns the model architecture string (from `general.architecture`).
    #[must_use]
    pub fn architecture(&self) -> &str {
        &self.architecture
    }

    /// Returns the number of transformer blocks/layers.
    #[must_use]
    pub const fn layer_count(&self) -> usize {
        self.layer_count
    }

    /// Returns the embedding vector dimension.
    #[must_use]
    pub const fn embedding_dim(&self) -> usize {
        self.embedding_dim
    }

    /// Returns the attention query head count.
    #[must_use]
    pub const fn head_count(&self) -> usize {
        self.head_count
    }

    /// Returns the key/value attention head count (for GQA/MQA).
    #[must_use]
    pub const fn head_count_kv(&self) -> usize {
        self.head_count_kv
    }

    /// Returns the maximum context window length supported or trained for this model (ADR-027).
    #[must_use]
    pub const fn context_length(&self) -> usize {
        self.context_length
    }

    /// Returns the configured tensor data alignment in bytes.
    #[must_use]
    pub const fn alignment(&self) -> usize {
        self.alignment
    }

    /// Returns the total number of tensors in the model.
    #[must_use]
    pub const fn tensor_count(&self) -> usize {
        self.tensor_count
    }

    /// Returns the total resident weight footprint in bytes (ADR-037).
    #[must_use]
    pub const fn total_weight_bytes(&self) -> usize {
        self.total_weight_bytes
    }

    /// Returns the full list of tensor descriptors.
    #[must_use]
    pub fn tensors(&self) -> &[TensorDescriptor] {
        &self.tensors
    }

    /// Searches for a tensor descriptor by name.
    #[must_use]
    pub fn find_tensor(&self, name: &str) -> Option<&TensorDescriptor> {
        self.tensors.iter().find(|t| t.name == name)
    }
}

struct RawTensorInfo {
    name: String,
    quant: QuantType,
    shape: Vec<usize>,
    num_elements: usize,
    byte_len: usize,
    relative_offset: usize,
}

struct ParsedTensors {
    descriptors: Vec<TensorDescriptor>,
    tensor_index: BTreeMap<String, usize>,
    total_weight_bytes: usize,
}

/// Zero-copy GGUF v3 file view backed by a contiguous byte slice (e.g. from an `mmap` buffer).
#[derive(Debug, Clone)]
pub struct GgufFile<'a> {
    buffer: &'a [u8],
    metadata: ModelMetadata,
    tensor_index: BTreeMap<String, usize>,
}

impl<'a> GgufFile<'a> {
    /// Parses a GGUF v3 file from a contiguous byte slice without copying tensor data.
    ///
    /// # Errors
    /// Returns [`K3Error::TensorBoundsError`] if the buffer is truncated, has an invalid
    /// magic number, unsupported version, or malformed key-value structures.
    /// Returns [`K3Error::CapabilityMismatch`] if required architecture keys are missing
    /// or if a tensor uses an unmapped quantization format.
    pub fn parse(buffer: &'a [u8]) -> Result<Self, K3Error> {
        let mut cursor = BinaryCursor::new(buffer);

        // 1. Validate magic bytes: "GGUF"
        let magic = cursor.read_u32()?;
        if magic != GGUF_MAGIC {
            return Err(K3Error::TensorBoundsError {
                tensor_name: "header".to_owned(),
                offset: 0,
                byte_len: 4,
                buffer_len: buffer.len(),
                reason: format!("invalid GGUF magic 0x{magic:08X}, expected 0x{GGUF_MAGIC:08X}"),
            });
        }

        // 2. Validate version: must be 3
        let version = cursor.read_u32()?;
        if version != GGUF_VERSION_V3 {
            return Err(K3Error::CapabilityMismatch {
                architecture: "unknown".to_owned(),
                reason: format!(
                    "unsupported GGUF version {version}; runtime requires GGUF v{GGUF_VERSION_V3}"
                ),
            });
        }

        // 3. Read tensor count and metadata KV count
        let tensor_count_u64 = cursor.read_u64()?;
        let tensor_count =
            usize::try_from(tensor_count_u64).map_err(|_| K3Error::TensorBoundsError {
                tensor_name: "header".to_owned(),
                offset: cursor.position(),
                byte_len: 8,
                buffer_len: buffer.len(),
                reason: format!("tensor count {tensor_count_u64} overflows usize"),
            })?;

        let kv_count_u64 = cursor.read_u64()?;
        let kv_count = usize::try_from(kv_count_u64).map_err(|_| K3Error::TensorBoundsError {
            tensor_name: "header".to_owned(),
            offset: cursor.position(),
            byte_len: 8,
            buffer_len: buffer.len(),
            reason: format!("metadata KV count {kv_count_u64} overflows usize"),
        })?;

        // 4. Parse metadata key-value table
        let metadata_map = Self::parse_metadata_map(&mut cursor, kv_count)?;

        // Extract alignment (default 32 bytes)
        let alignment = metadata_map
            .get("general.alignment")
            .and_then(GgufValue::as_usize)
            .unwrap_or(DEFAULT_ALIGNMENT);

        // 5. Extract general.architecture
        let architecture = metadata_map
            .get("general.architecture")
            .and_then(GgufValue::as_str)
            .ok_or_else(|| K3Error::CapabilityMismatch {
                architecture: "unknown".to_owned(),
                reason: "missing required 'general.architecture' metadata key".to_owned(),
            })?
            .to_owned();

        // Extract architecture-prefixed hyperparameters
        let block_count_key = format!("{architecture}.block_count");
        let layer_count = metadata_map
            .get(block_count_key.as_str())
            .and_then(GgufValue::as_usize)
            .unwrap_or(0);

        let emb_len_key = format!("{architecture}.embedding_length");
        let embedding_dim = metadata_map
            .get(emb_len_key.as_str())
            .and_then(GgufValue::as_usize)
            .unwrap_or(0);

        let head_count_key = format!("{architecture}.attention.head_count");
        let head_count = metadata_map
            .get(head_count_key.as_str())
            .and_then(GgufValue::as_usize)
            .unwrap_or(0);

        let head_count_kv_key = format!("{architecture}.attention.head_count_kv");
        let head_count_kv = metadata_map
            .get(head_count_kv_key.as_str())
            .and_then(GgufValue::as_usize)
            .unwrap_or(head_count);

        let ctx_len_key = format!("{architecture}.context_length");
        let context_length = metadata_map
            .get(ctx_len_key.as_str())
            .and_then(GgufValue::as_usize)
            .unwrap_or(2048);

        // 6. Parse raw tensor info descriptors
        let raw_tensors =
            Self::parse_raw_tensors(&mut cursor, tensor_count, &architecture, buffer.len())?;

        // 7. Calculate data section starting offset aligned to `alignment`
        let tensor_data_offset = align_up(cursor.position(), alignment);

        // 8. Build verified tensor descriptors and total weight bytes
        let parsed_tensors =
            Self::build_tensor_descriptors(raw_tensors, tensor_data_offset, buffer.len())?;

        let metadata = ModelMetadata {
            architecture,
            layer_count,
            embedding_dim,
            head_count,
            head_count_kv,
            context_length,
            alignment,
            tensor_count,
            total_weight_bytes: parsed_tensors.total_weight_bytes,
            tensors: parsed_tensors.descriptors,
        };

        Ok(Self {
            buffer,
            metadata,
            tensor_index: parsed_tensors.tensor_index,
        })
    }

    fn parse_metadata_map(
        cursor: &mut BinaryCursor<'a>,
        kv_count: usize,
    ) -> Result<BTreeMap<&'a str, GgufValue<'a>>, K3Error> {
        let mut metadata_map = BTreeMap::new();
        for _ in 0..kv_count {
            let key = cursor.read_string()?;
            let val = cursor.read_value()?;
            metadata_map.insert(key, val);
        }
        Ok(metadata_map)
    }

    fn parse_raw_tensors(
        cursor: &mut BinaryCursor<'a>,
        tensor_count: usize,
        architecture: &str,
        buffer_len: usize,
    ) -> Result<Vec<RawTensorInfo>, K3Error> {
        let mut raw_tensors = Vec::with_capacity(tensor_count);
        for _ in 0..tensor_count {
            let tensor_name = cursor.read_string()?.to_owned();
            let n_dims_u32 = cursor.read_u32()?;
            let n_dims = usize::try_from(n_dims_u32).map_err(|_| K3Error::TensorBoundsError {
                tensor_name: tensor_name.clone(),
                offset: cursor.position(),
                byte_len: 4,
                buffer_len,
                reason: format!("dimensions count {n_dims_u32} overflows usize"),
            })?;

            if n_dims == 0 {
                return Err(K3Error::TensorBoundsError {
                    tensor_name: tensor_name.clone(),
                    offset: cursor.position(),
                    byte_len: 0,
                    buffer_len,
                    reason: "tensor dimension count must be non-zero".to_owned(),
                });
            }

            let mut shape = Vec::with_capacity(n_dims);
            let mut num_elements: usize = 1;
            for _ in 0..n_dims {
                let dim_u64 = cursor.read_u64()?;
                let dim = usize::try_from(dim_u64).map_err(|_| K3Error::TensorBoundsError {
                    tensor_name: tensor_name.clone(),
                    offset: cursor.position(),
                    byte_len: 8,
                    buffer_len,
                    reason: format!("dimension {dim_u64} overflows usize"),
                })?;
                if dim == 0 {
                    return Err(K3Error::TensorBoundsError {
                        tensor_name: tensor_name.clone(),
                        offset: cursor.position(),
                        byte_len: 0,
                        buffer_len,
                        reason: "tensor dimension must be non-zero".to_owned(),
                    });
                }
                num_elements =
                    num_elements
                        .checked_mul(dim)
                        .ok_or_else(|| K3Error::TensorBoundsError {
                            tensor_name: tensor_name.clone(),
                            offset: cursor.position(),
                            byte_len: 0,
                            buffer_len,
                            reason: "tensor element count overflowed usize".to_owned(),
                        })?;
                shape.push(dim);
            }

            let ggml_type_u32 = cursor.read_u32()?;
            let quant = QuantType::from_gguf_type_id(ggml_type_u32).ok_or_else(|| {
                K3Error::CapabilityMismatch {
                    architecture: architecture.to_owned(),
                    reason: format!(
                        "tensor '{tensor_name}' uses unsupported ggml_type ID {ggml_type_u32}"
                    ),
                }
            })?;

            let rel_offset_u64 = cursor.read_u64()?;
            let relative_offset =
                usize::try_from(rel_offset_u64).map_err(|_| K3Error::TensorBoundsError {
                    tensor_name: tensor_name.clone(),
                    offset: cursor.position(),
                    byte_len: 8,
                    buffer_len,
                    reason: format!("relative offset {rel_offset_u64} overflows usize"),
                })?;

            let byte_len = quant.storage_bytes(&tensor_name, num_elements)?;

            raw_tensors.push(RawTensorInfo {
                name: tensor_name,
                quant,
                shape,
                num_elements,
                byte_len,
                relative_offset,
            });
        }
        Ok(raw_tensors)
    }

    fn build_tensor_descriptors(
        raw_tensors: Vec<RawTensorInfo>,
        tensor_data_offset: usize,
        buffer_len: usize,
    ) -> Result<ParsedTensors, K3Error> {
        let mut total_weight_bytes: usize = 0;
        let mut descriptors = Vec::with_capacity(raw_tensors.len());
        let mut tensor_index = BTreeMap::new();

        for (idx, raw) in raw_tensors.into_iter().enumerate() {
            let absolute_offset = tensor_data_offset
                .checked_add(raw.relative_offset)
                .ok_or_else(|| K3Error::TensorBoundsError {
                    tensor_name: raw.name.clone(),
                    offset: tensor_data_offset,
                    byte_len: raw.byte_len,
                    buffer_len,
                    reason: "tensor absolute offset calculation overflowed usize".to_owned(),
                })?;

            let end_offset = absolute_offset.checked_add(raw.byte_len).ok_or_else(|| {
                K3Error::TensorBoundsError {
                    tensor_name: raw.name.clone(),
                    offset: absolute_offset,
                    byte_len: raw.byte_len,
                    buffer_len,
                    reason: "tensor end offset calculation overflowed usize".to_owned(),
                }
            })?;

            if end_offset > buffer_len {
                return Err(K3Error::TensorBoundsError {
                    tensor_name: raw.name.clone(),
                    offset: absolute_offset,
                    byte_len: raw.byte_len,
                    buffer_len,
                    reason: format!(
                        "tensor range [{absolute_offset}..{end_offset}] exceeds buffer length {buffer_len}"
                    ),
                });
            }

            total_weight_bytes = total_weight_bytes
                .checked_add(raw.byte_len)
                .ok_or_else(|| K3Error::TensorBoundsError {
                    tensor_name: raw.name.clone(),
                    offset: absolute_offset,
                    byte_len: raw.byte_len,
                    buffer_len,
                    reason: "total weight bytes overflowed usize".to_owned(),
                })?;

            tensor_index.insert(raw.name.clone(), idx);

            descriptors.push(TensorDescriptor {
                name: raw.name,
                quant: raw.quant,
                shape: raw.shape,
                num_elements: raw.num_elements,
                relative_offset: raw.relative_offset,
                absolute_offset,
                byte_len: raw.byte_len,
            });
        }

        Ok(ParsedTensors {
            descriptors,
            tensor_index,
            total_weight_bytes,
        })
    }

    /// Returns the parsed [`ModelMetadata`] for this model.
    #[must_use]
    pub const fn metadata(&self) -> &ModelMetadata {
        &self.metadata
    }

    /// Returns the model architecture string identifier.
    #[must_use]
    pub fn architecture(&self) -> &str {
        self.metadata.architecture()
    }

    /// Returns the raw backing byte slice.
    #[must_use]
    pub const fn as_bytes(&self) -> &'a [u8] {
        self.buffer
    }
}

impl ModelWeights for GgufFile<'_> {
    fn tensor(&self, name: &str) -> Result<TensorView<'_>, K3Error> {
        let &idx = self
            .tensor_index
            .get(name)
            .ok_or_else(|| K3Error::TensorBoundsError {
                tensor_name: name.to_owned(),
                offset: 0,
                byte_len: 0,
                buffer_len: self.buffer.len(),
                reason: format!("tensor '{name}' not found in GGUF weight table"),
            })?;

        let desc = &self.metadata.tensors[idx];
        TensorView::from_bytes(
            desc.name(),
            desc.quant(),
            desc.shape(),
            desc.absolute_offset(),
            self.buffer,
        )
    }

    fn tensor_count(&self) -> usize {
        self.metadata.tensor_count()
    }

    fn total_weight_bytes(&self) -> usize {
        self.metadata.total_weight_bytes()
    }
}

impl fmt::Display for ModelMetadata {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "ModelMetadata(arch='{}', layers={}, emb={}, heads={}/{}, ctx={}, tensors={}, bytes={})",
            self.architecture,
            self.layer_count,
            self.embedding_dim,
            self.head_count,
            self.head_count_kv,
            self.context_length,
            self.tensor_count,
            self.total_weight_bytes
        )
    }
}

/// Test helpers for synthesizing GGUF v3 binaries in unit tests.
#[doc(hidden)]
pub mod test_helpers {
    use super::{
        GGUF_MAGIC, GGUF_VERSION_V3, GgufMetadataValueType, GgufValue, QuantType, align_up,
    };

    /// Test helper that constructs a synthetic valid GGUF v3 byte buffer.
    #[must_use]
    #[allow(clippy::missing_panics_doc)]
    pub fn create_synthetic_gguf_v3(
        architecture: &str,
        alignment: usize,
        tensor_specs: &[(&str, QuantType, &[usize])],
    ) -> Vec<u8> {
        let mut buf = Vec::new();

        // 1. Magic & Version
        buf.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
        buf.extend_from_slice(&GGUF_VERSION_V3.to_le_bytes());

        // 2. Tensor count & KV count
        let tensor_count = u64::try_from(tensor_specs.len()).expect("tensor count fits u64");
        let k_block = format!("{architecture}.block_count");
        let k_emb = format!("{architecture}.embedding_length");
        let k_head = format!("{architecture}.attention.head_count");
        let k_ctx = format!("{architecture}.context_length");
        let alignment_u32 = u32::try_from(alignment).expect("alignment fits u32");

        let kv_pairs: Vec<(&str, GgufValue<'_>)> = vec![
            ("general.architecture", GgufValue::String(architecture)),
            ("general.alignment", GgufValue::Uint32(alignment_u32)),
            (&k_block, GgufValue::Uint32(4)),
            (&k_emb, GgufValue::Uint32(256)),
            (&k_head, GgufValue::Uint32(4)),
            (&k_ctx, GgufValue::Uint32(2048)),
        ];

        buf.extend_from_slice(&tensor_count.to_le_bytes());
        let kv_len = u64::try_from(kv_pairs.len()).expect("kv count fits u64");
        buf.extend_from_slice(&kv_len.to_le_bytes());

        // 3. Write KV pairs
        for (k, v) in kv_pairs {
            write_string(&mut buf, k);
            match v {
                GgufValue::String(s) => {
                    buf.extend_from_slice(&(GgufMetadataValueType::String as u32).to_le_bytes());
                    write_string(&mut buf, s);
                }
                GgufValue::Uint32(u) => {
                    buf.extend_from_slice(&(GgufMetadataValueType::Uint32 as u32).to_le_bytes());
                    buf.extend_from_slice(&u.to_le_bytes());
                }
                _ => panic!("unhandled value type in test helper"),
            }
        }

        // 4. Calculate relative tensor offsets and write tensor descriptors
        let mut rel_offset = 0usize;
        let mut tensor_data_chunks = Vec::new();

        for &(name, quant, shape) in tensor_specs {
            let num_elements: usize = shape.iter().product();
            let byte_len = quant.storage_bytes(name, num_elements).unwrap();

            write_string(&mut buf, name);
            let shape_len = u32::try_from(shape.len()).expect("shape rank fits u32");
            buf.extend_from_slice(&shape_len.to_le_bytes());
            for &dim in shape {
                let dim_u64 = u64::try_from(dim).expect("dimension fits u64");
                buf.extend_from_slice(&dim_u64.to_le_bytes());
            }

            // Map QuantType to ggml_type ID
            let ggml_type_id: u32 = match quant {
                QuantType::F32 => 0,
                QuantType::F16 => 1,
                QuantType::Q4_0 => 2,
                QuantType::Q8_0 => 8,
                QuantType::Q4_K_M => 12,
                QuantType::Q6_K => 14,
                QuantType::BF16 => 30,
            };
            buf.extend_from_slice(&ggml_type_id.to_le_bytes());
            let rel_offset_u64 = u64::try_from(rel_offset).expect("rel_offset fits u64");
            buf.extend_from_slice(&rel_offset_u64.to_le_bytes());

            tensor_data_chunks.push(vec![0xAA_u8; byte_len]);
            rel_offset += byte_len;
        }

        // 5. Pad buffer to alignment
        let current_len = buf.len();
        let aligned_len = align_up(current_len, alignment);
        buf.resize(aligned_len, 0);

        // 6. Append tensor data
        for chunk in tensor_data_chunks {
            buf.extend_from_slice(&chunk);
        }

        buf
    }

    fn write_string(buf: &mut Vec<u8>, s: &str) {
        let len_u64 = u64::try_from(s.len()).expect("string length fits u64");
        buf.extend_from_slice(&len_u64.to_le_bytes());
        buf.extend_from_slice(s.as_bytes());
    }
}

#[cfg(test)]
mod tests {
    use super::test_helpers::create_synthetic_gguf_v3;
    use super::*;

    #[test]
    fn test_valid_gguf_v3_parsing_and_hyperparameters() {
        let tensor_specs = [
            ("blk.0.attn_q.weight", QuantType::Q4_K_M, &[256, 2][..]),
            ("output_norm.weight", QuantType::F32, &[16][..]),
        ];
        let bytes = create_synthetic_gguf_v3("synthetic", 32, &tensor_specs);

        let gguf = GgufFile::parse(&bytes).expect("GGUF v3 parsing succeeds");
        let meta = gguf.metadata();

        assert_eq!(meta.architecture(), "synthetic");
        assert_eq!(meta.layer_count(), 4);
        assert_eq!(meta.embedding_dim(), 256);
        assert_eq!(meta.head_count(), 4);
        assert_eq!(meta.head_count_kv(), 4);
        assert_eq!(meta.context_length(), 2048);
        assert_eq!(meta.alignment(), 32);
        assert_eq!(meta.tensor_count(), 2);
        assert_eq!(meta.total_weight_bytes(), 288 + 64);

        // Verify ModelWeights trait
        let q_view = gguf
            .tensor("blk.0.attn_q.weight")
            .expect("attn_q tensor found");
        assert_eq!(q_view.name(), "blk.0.attn_q.weight");
        assert_eq!(q_view.quant(), QuantType::Q4_K_M);
        assert_eq!(q_view.shape(), &[256, 2]);
        assert_eq!(q_view.byte_len(), 288);

        let norm_view = gguf
            .tensor("output_norm.weight")
            .expect("output_norm tensor found");
        assert_eq!(norm_view.name(), "output_norm.weight");
        assert_eq!(norm_view.quant(), QuantType::F32);
        assert_eq!(norm_view.shape(), &[16]);
        assert_eq!(norm_view.byte_len(), 64);

        // Query nonexistent tensor
        assert!(gguf.tensor("nonexistent").is_err());
    }

    #[test]
    fn test_invalid_magic_rejected() {
        let mut bytes = create_synthetic_gguf_v3("synthetic", 32, &[]);
        bytes[0] = 0x00; // corrupt magic
        let err = GgufFile::parse(&bytes).expect_err("must reject invalid magic");
        assert!(matches!(err, K3Error::TensorBoundsError { .. }));
        assert!(err.to_string().contains("invalid GGUF magic"));
    }

    #[test]
    fn test_unsupported_version_rejected() {
        let mut bytes = create_synthetic_gguf_v3("synthetic", 32, &[]);
        bytes[4] = 2; // version 2 instead of 3
        let err = GgufFile::parse(&bytes).expect_err("must reject version 2");
        assert!(matches!(err, K3Error::CapabilityMismatch { .. }));
        assert!(err.to_string().contains("unsupported GGUF version"));
    }

    #[test]
    fn test_truncated_buffer_rejected() {
        let bytes = create_synthetic_gguf_v3("synthetic", 32, &[]);
        let truncated = &bytes[..12]; // truncate before reading KV count
        let err = GgufFile::parse(truncated).expect_err("must reject truncated buffer");
        assert!(matches!(err, K3Error::TensorBoundsError { .. }));
    }

    #[test]
    fn test_missing_general_architecture_rejected() {
        let mut buf = Vec::new();
        buf.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
        buf.extend_from_slice(&GGUF_VERSION_V3.to_le_bytes());
        buf.extend_from_slice(&0_u64.to_le_bytes()); // 0 tensors
        buf.extend_from_slice(&0_u64.to_le_bytes()); // 0 KV pairs -> missing general.architecture

        let err = GgufFile::parse(&buf).expect_err("missing general.architecture must fail");
        assert!(matches!(err, K3Error::CapabilityMismatch { .. }));
        assert!(
            err.to_string()
                .contains("missing required 'general.architecture'")
        );
    }

    #[test]
    fn test_tensor_oob_rejected() {
        let tensor_specs = [("blk.0.attn_q.weight", QuantType::Q4_K_M, &[256, 2][..])];
        let mut bytes = create_synthetic_gguf_v3("synthetic", 32, &tensor_specs);
        // Truncate the tensor payload data
        bytes.truncate(bytes.len() - 50);

        let err = GgufFile::parse(&bytes).expect_err("must reject payload exceeding buffer");
        assert!(matches!(err, K3Error::TensorBoundsError { .. }));
        assert!(err.to_string().contains("exceeds buffer length"));
    }
}
