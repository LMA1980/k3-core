//! Zero-copy GGUF v3 binary parser, metadata extraction, and tensor descriptor mapping.
//!
//! Conforms to the GGUF v3 specification and enforces upfront architecture and quantization
//! capability verification (ADR-017, ADR-023, ADR-027, ADR-037).

use core::fmt;
use std::collections::{BTreeMap, BTreeSet};

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

    /// Returns the value as a vector of `usize` if it is an integer array with non-negative values.
    #[must_use]
    pub fn as_usize_array(&self) -> Option<Vec<usize>> {
        let Self::Array(arr) = self else {
            return None;
        };
        match arr {
            GgufArray::Uint8(items) => Some(items.iter().map(|&v| usize::from(v)).collect()),
            GgufArray::Uint16(items) => Some(items.iter().map(|&v| usize::from(v)).collect()),
            GgufArray::Uint32(items) => items.iter().map(|&v| usize::try_from(v).ok()).collect(),
            GgufArray::Uint64(items) => items.iter().map(|&v| usize::try_from(v).ok()).collect(),
            GgufArray::Int8(items) => items.iter().map(|&v| usize::try_from(v).ok()).collect(),
            GgufArray::Int16(items) => items.iter().map(|&v| usize::try_from(v).ok()).collect(),
            GgufArray::Int32(items) => items.iter().map(|&v| usize::try_from(v).ok()).collect(),
            GgufArray::Int64(items) => items.iter().map(|&v| usize::try_from(v).ok()).collect(),
            _ => None,
        }
    }

    /// Returns the value as `u32` if convertible.
    #[must_use]
    pub fn as_u32(&self) -> Option<u32> {
        self.as_u64().and_then(|v| u32::try_from(v).ok())
    }

    /// Returns the inner [`GgufArray`] if this value is [`GgufValue::Array`].
    #[must_use]
    pub const fn as_array(&self) -> Option<&GgufArray<'a>> {
        match self {
            Self::Array(arr) => Some(arr),
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

/// Recurrent KV cache allocation strategy for looped latent-recurrent architectures.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DoryKvCacheMode {
    /// Allocate independent physical KV cache slots for each recurrent loop iteration.
    #[default]
    PerLoop,
    /// Collapse recurrent loop iterations onto shared physical KV cache slots.
    LastLoop,
}

/// Type alias for [`DoryKvCacheMode`] representing recurrent KV cache allocation strategy.
pub type RecurrentKvCacheMode = DoryKvCacheMode;

impl DoryKvCacheMode {
    /// Returns the canonical GGUF metadata string identifier for this cache mode.
    #[must_use]
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::PerLoop => "per_loop",
            Self::LastLoop => "last_loop",
        }
    }

    /// Parses a string slice into [`DoryKvCacheMode`], returning `None` if unrecognized.
    #[must_use]
    pub fn parse(s: &str) -> Option<Self> {
        let trimmed = s.trim();
        if trimmed.eq_ignore_ascii_case("per_loop") || trimmed.eq_ignore_ascii_case("per-loop") {
            Some(Self::PerLoop)
        } else if trimmed.eq_ignore_ascii_case("last_loop")
            || trimmed.eq_ignore_ascii_case("last-loop")
        {
            Some(Self::LastLoop)
        } else {
            None
        }
    }

    /// Parses a string representation into [`DoryKvCacheMode`], returning [`K3Error::CapabilityMismatch`] on error.
    ///
    /// # Errors
    /// Returns [`K3Error::CapabilityMismatch`] if `s` is not a valid KV cache mode.
    pub fn parse_mode(s: &str) -> Result<Self, K3Error> {
        Self::parse(s).ok_or_else(|| K3Error::CapabilityMismatch {
            architecture: "dory".to_owned(),
            reason: format!(
                "unknown or invalid recurrent_kv_cache_mode: '{s}', expected 'per_loop' or 'last_loop'"
            ),
        })
    }
}

impl fmt::Display for DoryKvCacheMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Classification of physical sub-layers in Dory's disaggregated layer pattern.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DoryLayerType {
    /// Sliding-window causal attention sub-layer (`'S'`).
    SlidingWindowAttention,
    /// Full / global causal attention sub-layer (`'*'`).
    FullAttention,
    /// `SwiGLU` / feed-forward MLP sub-layer (`'-'`).
    Mlp,
    /// Unrecognized layer pattern character.
    Unknown(char),
}

/// Architecture-specific metadata for NVIDIA Nemotron-Labs Dory looped nGPT models.
#[derive(Debug, Clone, PartialEq)]
pub struct DoryMetadata {
    input_layer_count: u32,
    recurrent_layer_count: u32,
    output_layer_count: u32,
    num_loops: u32,
    recurrent_kv_cache_mode: DoryKvCacheMode,
    layer_pattern: String,
    rope_profile: String,
    rope_freq_base_2: f32,
    init_std: f32,
    alpha_init: f32,
    gate_init: f32,
}

impl DoryMetadata {
    /// Returns the number of initial feed-forward / projection input layers ($N_{\text{input}}$).
    #[must_use]
    pub const fn input_layer_count(&self) -> u32 {
        self.input_layer_count
    }

    /// Returns the number of physical recurrent sub-layers ($N_{\text{recurrent}}$).
    #[must_use]
    pub const fn recurrent_layer_count(&self) -> u32 {
        self.recurrent_layer_count
    }

    /// Returns the number of final output sub-layers ($N_{\text{output}}$).
    #[must_use]
    pub const fn output_layer_count(&self) -> u32 {
        self.output_layer_count
    }

    /// Returns the active number of recurrent loop cycles ($K \ge 1$).
    #[must_use]
    pub const fn num_loops(&self) -> u32 {
        self.num_loops
    }

    /// Returns the recurrent KV cache mode ([`DoryKvCacheMode::PerLoop`] vs [`DoryKvCacheMode::LastLoop`]).
    #[must_use]
    pub const fn recurrent_kv_cache_mode(&self) -> DoryKvCacheMode {
        self.recurrent_kv_cache_mode
    }

    /// Returns the recurrent KV cache mode as a static string slice (`"per_loop"` or `"last_loop"`).
    #[must_use]
    pub const fn recurrent_kv_cache_mode_str(&self) -> &'static str {
        self.recurrent_kv_cache_mode.as_str()
    }

    /// Returns the layer topology pattern string (e.g., `"S-S-S-*-..."`).
    #[must_use]
    pub fn layer_pattern(&self) -> &str {
        &self.layer_pattern
    }

    /// Returns the `RoPE` profile identifier (e.g., `"dual_theta"` or `"neox"`).
    #[must_use]
    pub fn rope_profile(&self) -> &str {
        &self.rope_profile
    }

    /// Returns the secondary `RoPE` base frequency constant (theta 2, typically 10,000.0).
    #[must_use]
    pub const fn rope_freq_base_2(&self) -> f32 {
        self.rope_freq_base_2
    }

    /// Returns the weights initialization standard deviation.
    #[must_use]
    pub const fn init_std(&self) -> f32 {
        self.init_std
    }

    /// Returns the initial eigen-rate residual interpolation constant ($\alpha_{\text{init}}$).
    #[must_use]
    pub const fn alpha_init(&self) -> f32 {
        self.alpha_init
    }

    /// Returns the initial attention gate bias / scale constant ($g_{\text{init}}$).
    #[must_use]
    pub const fn gate_init(&self) -> f32 {
        self.gate_init
    }

    /// Returns the total physical layer count stored in the weight table ($N_{\text{input}} + N_{\text{recurrent}} + N_{\text{output}}$).
    #[must_use]
    pub const fn physical_layer_count(&self) -> u32 {
        self.input_layer_count + self.recurrent_layer_count + self.output_layer_count
    }

    /// Returns the total physical layer count stored in the weight table.
    #[must_use]
    pub const fn total_layer_count(&self) -> u32 {
        self.physical_layer_count()
    }

    /// Returns the total physical layer count stored in the weight table.
    #[must_use]
    pub const fn total_physical_layers(&self) -> u32 {
        self.physical_layer_count()
    }

    /// Returns the total virtual layers executed across all recurrent loops ($N_{\text{input}} + N_{\text{recurrent}} \times K + N_{\text{output}}$).
    #[must_use]
    pub const fn effective_layer_count(&self) -> u32 {
        self.input_layer_count
            + (self.recurrent_layer_count * self.num_loops)
            + self.output_layer_count
    }

    /// Returns the total virtual layers executed across all recurrent loops.
    #[must_use]
    pub const fn total_virtual_layers(&self) -> u32 {
        self.effective_layer_count()
    }

    /// Returns the virtual layer count for an arbitrary loop count $K$.
    #[must_use]
    pub const fn virtual_layer_count(&self, loops: u32) -> u32 {
        self.input_layer_count + (self.recurrent_layer_count * loops) + self.output_layer_count
    }

    /// Returns `true` if `physical_layer_idx` belongs to the input segment (`0..input_layer_count`).
    #[must_use]
    pub const fn is_input_layer(&self, physical_layer_idx: u32) -> bool {
        physical_layer_idx < self.input_layer_count
    }

    /// Returns `true` if `physical_layer_idx` belongs to the recurrent segment.
    #[must_use]
    pub const fn is_recurrent_layer(&self, physical_layer_idx: u32) -> bool {
        physical_layer_idx >= self.input_layer_count
            && physical_layer_idx < self.input_layer_count + self.recurrent_layer_count
    }

    /// Returns `true` if `physical_layer_idx` belongs to the output segment.
    #[must_use]
    pub const fn is_output_layer(&self, physical_layer_idx: u32) -> bool {
        physical_layer_idx >= self.input_layer_count + self.recurrent_layer_count
            && physical_layer_idx < self.physical_layer_count()
    }

    /// Maps a virtual execution step (`0..effective_layer_count()`) to its physical weight layer index (`0..physical_layer_count()`).
    #[must_use]
    pub const fn physical_layer_for_virtual_step(&self, virtual_step: u32) -> Option<u32> {
        if virtual_step < self.input_layer_count {
            Some(virtual_step)
        } else {
            let recurrent_total = self.recurrent_layer_count * self.num_loops;
            let recurrent_end = self.input_layer_count + recurrent_total;
            if virtual_step < recurrent_end {
                let rec_step = (virtual_step - self.input_layer_count) % self.recurrent_layer_count;
                Some(self.input_layer_count + rec_step)
            } else if virtual_step < self.effective_layer_count() {
                let out_step = virtual_step - recurrent_end;
                Some(self.input_layer_count + self.recurrent_layer_count + out_step)
            } else {
                None
            }
        }
    }

    /// Returns the 0-indexed loop iteration ($0..K$) for a virtual step within the recurrent segment, or `None` if input/output.
    #[must_use]
    pub const fn loop_index_for_virtual_step(&self, virtual_step: u32) -> Option<u32> {
        let recurrent_total = self.recurrent_layer_count * self.num_loops;
        let recurrent_end = self.input_layer_count + recurrent_total;
        if virtual_step >= self.input_layer_count && virtual_step < recurrent_end {
            Some((virtual_step - self.input_layer_count) / self.recurrent_layer_count)
        } else {
            None
        }
    }

    /// Returns the parsed [`DoryLayerType`] for a physical layer index by consulting `layer_pattern`.
    #[must_use]
    pub fn layer_type(&self, physical_layer_idx: u32) -> Option<DoryLayerType> {
        let idx = usize::try_from(physical_layer_idx).ok()?;
        self.layer_pattern.chars().nth(idx).map(|c| match c {
            'S' => DoryLayerType::SlidingWindowAttention,
            '*' => DoryLayerType::FullAttention,
            '-' => DoryLayerType::Mlp,
            other => DoryLayerType::Unknown(other),
        })
    }

    /// Returns `true` if `physical_layer_idx` is an attention sub-layer ('S' or '*').
    #[must_use]
    pub fn is_attention_layer(&self, physical_layer_idx: u32) -> bool {
        matches!(
            self.layer_type(physical_layer_idx),
            Some(DoryLayerType::SlidingWindowAttention | DoryLayerType::FullAttention)
        )
    }

    /// Returns `true` if `physical_layer_idx` is an MLP sub-layer ('-').
    #[must_use]
    pub fn is_mlp_layer(&self, physical_layer_idx: u32) -> bool {
        matches!(
            self.layer_type(physical_layer_idx),
            Some(DoryLayerType::Mlp)
        )
    }
}

impl fmt::Display for DoryMetadata {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "DoryMetadata(input={}, recurrent={}, output={}, loops={}, kv_mode='{}', rope_base_2={})",
            self.input_layer_count,
            self.recurrent_layer_count,
            self.output_layer_count,
            self.num_loops,
            self.recurrent_kv_cache_mode.as_str(),
            self.rope_freq_base_2
        )
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
    dory: Option<DoryMetadata>,
    kv_lora_rank: Option<usize>,
    qk_rope_head_dim: Option<usize>,
    q_lora_rank: Option<usize>,
    qk_nope_head_dim: Option<usize>,
    v_head_dim: Option<usize>,
    expert_count: Option<usize>,
    expert_used_count: Option<usize>,
    expert_shared_count: Option<usize>,
    expert_feed_forward_length: Option<usize>,
    head_count_kv_per_layer: Vec<usize>,
    ssm_conv_kernel: Option<usize>,
    ssm_inner_size: Option<usize>,
    ssm_state_size: Option<usize>,
    ssm_time_step_rank: Option<usize>,
    ssm_group_count: Option<usize>,
    shortconv_l_cache: Option<usize>,
    wkv_head_size: Option<usize>,
    embedding_scale: Option<f32>,
    residual_scale: Option<f32>,
    attention_scale: Option<f32>,
    logit_scale: Option<f32>,
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

    /// Returns the parsed Dory-specific metadata if this model uses the `"dory"` architecture.
    #[must_use]
    pub fn dory(&self) -> Option<&DoryMetadata> {
        self.dory.as_ref()
    }

    /// Returns the `MLA` KV latent `LoRA` rank (`{arch}.attention.kv_lora_rank`), if present.
    #[must_use]
    pub const fn kv_lora_rank(&self) -> Option<usize> {
        self.kv_lora_rank
    }

    /// Returns the decoupled `MLA` `RoPE` key head dimension, if present.
    #[must_use]
    pub const fn qk_rope_head_dim(&self) -> Option<usize> {
        self.qk_rope_head_dim
    }

    /// Returns the `MLA` query `LoRA` rank (`{arch}.attention.q_lora_rank`), if present.
    #[must_use]
    pub const fn q_lora_rank(&self) -> Option<usize> {
        self.q_lora_rank
    }

    /// Returns the non-rotary per-head key dimension (`{arch}.attention.key_length`), if present.
    #[must_use]
    pub const fn qk_nope_head_dim(&self) -> Option<usize> {
        self.qk_nope_head_dim
    }

    /// Returns the per-head value dimension (`{arch}.attention.value_length`), if present.
    #[must_use]
    pub const fn v_head_dim(&self) -> Option<usize> {
        self.v_head_dim
    }

    /// Returns the routed `MoE` expert count (`{arch}.expert_count`), if present.
    #[must_use]
    pub const fn expert_count(&self) -> Option<usize> {
        self.expert_count
    }

    /// Returns the active routed `MoE` experts per token (`{arch}.expert_used_count`), if present.
    #[must_use]
    pub const fn expert_used_count(&self) -> Option<usize> {
        self.expert_used_count
    }

    /// Returns the shared `MoE` expert count (`{arch}.expert_shared_count`), if present.
    #[must_use]
    pub const fn expert_shared_count(&self) -> Option<usize> {
        self.expert_shared_count
    }

    /// Returns the routed `MoE` expert feed-forward intermediate dimension (`{arch}.expert_feed_forward_length`), if present.
    #[must_use]
    pub const fn expert_feed_forward_length(&self) -> Option<usize> {
        self.expert_feed_forward_length
    }

    /// Returns per-layer key/value head counts (`{arch}.attention.head_count_kv` when encoded as an array).
    #[must_use]
    pub fn head_count_kv_per_layer(&self) -> &[usize] {
        &self.head_count_kv_per_layer
    }

    /// Returns the `SSM` depthwise convolution kernel width (`{arch}.ssm.conv_kernel`), if present.
    #[must_use]
    pub const fn ssm_conv_kernel(&self) -> Option<usize> {
        self.ssm_conv_kernel
    }

    /// Returns the `SSM` inner projection width (`{arch}.ssm.inner_size`), if present.
    #[must_use]
    pub const fn ssm_inner_size(&self) -> Option<usize> {
        self.ssm_inner_size
    }

    /// Returns the `SSM` state dimension (`{arch}.ssm.state_size`), if present.
    #[must_use]
    pub const fn ssm_state_size(&self) -> Option<usize> {
        self.ssm_state_size
    }

    /// Returns the `SSM` time-step rank or head count (`{arch}.ssm.time_step_rank`), if present.
    #[must_use]
    pub const fn ssm_time_step_rank(&self) -> Option<usize> {
        self.ssm_time_step_rank
    }

    /// Returns the `SSM` group count (`{arch}.ssm.group_count`), if present.
    #[must_use]
    pub const fn ssm_group_count(&self) -> Option<usize> {
        self.ssm_group_count
    }

    /// Returns the `LFM2` short-convolution cache length (`{arch}.shortconv.l_cache`), if present.
    #[must_use]
    pub const fn shortconv_l_cache(&self) -> Option<usize> {
        self.shortconv_l_cache
    }

    /// Returns the `RWKV` WKV head dimension (`{arch}.wkv.head_size`), if present.
    #[must_use]
    pub const fn wkv_head_size(&self) -> Option<usize> {
        self.wkv_head_size
    }

    /// Returns the Granite-style embedding multiplier (`{arch}.embedding_scale`), if present.
    #[must_use]
    pub const fn embedding_scale(&self) -> Option<f32> {
        self.embedding_scale
    }

    /// Returns the Granite-style residual branch multiplier (`{arch}.residual_scale`), if present.
    #[must_use]
    pub const fn residual_scale(&self) -> Option<f32> {
        self.residual_scale
    }

    /// Returns the Granite-style attention scaling multiplier (`{arch}.attention.scale`), if present.
    #[must_use]
    pub const fn attention_scale(&self) -> Option<f32> {
        self.attention_scale
    }

    /// Returns the Granite-style final logit multiplier (`{arch}.logit_scale`), if present.
    #[must_use]
    pub const fn logit_scale(&self) -> Option<f32> {
        self.logit_scale
    }

    /// Overrides the `MLA` compressed KV latent dimensions on this [`ModelMetadata`].
    #[must_use]
    pub const fn with_mla_dims(mut self, kv_lora_rank: usize, qk_rope_head_dim: usize) -> Self {
        self.kv_lora_rank = Some(kv_lora_rank);
        self.qk_rope_head_dim = Some(qk_rope_head_dim);
        self
    }

    /// Overrides `SSM` hyperparameters on this [`ModelMetadata`].
    #[must_use]
    pub const fn with_ssm_dims(
        mut self,
        conv_kernel: usize,
        inner_size: usize,
        state_size: usize,
        time_step_rank: usize,
        group_count: usize,
    ) -> Self {
        self.ssm_conv_kernel = Some(conv_kernel);
        self.ssm_inner_size = Some(inner_size);
        self.ssm_state_size = Some(state_size);
        self.ssm_time_step_rank = Some(time_step_rank);
        self.ssm_group_count = Some(group_count);
        self
    }

    /// Overrides the `LFM2` short-convolution cache length on this [`ModelMetadata`].
    #[must_use]
    pub const fn with_shortconv_l_cache(mut self, l_cache: usize) -> Self {
        self.shortconv_l_cache = Some(l_cache);
        self
    }

    /// Overrides the `RWKV` WKV head dimension on this [`ModelMetadata`].
    #[must_use]
    pub const fn with_wkv_head_size(mut self, head_size: usize) -> Self {
        self.wkv_head_size = Some(head_size);
        self
    }

    /// Overrides Granite scaling multipliers on this [`ModelMetadata`].
    #[must_use]
    pub const fn with_granite_scales(
        mut self,
        embedding_scale: Option<f32>,
        residual_scale: Option<f32>,
        attention_scale: Option<f32>,
        logit_scale: Option<f32>,
    ) -> Self {
        self.embedding_scale = embedding_scale;
        self.residual_scale = residual_scale;
        self.attention_scale = attention_scale;
        self.logit_scale = logit_scale;
        self
    }

    /// Overrides per-layer KV head counts on this [`ModelMetadata`].
    #[must_use]
    pub fn with_head_count_kv_per_layer(mut self, per_layer: Vec<usize>) -> Self {
        self.head_count_kv_per_layer = per_layer;
        self
    }

    /// Returns `true` when `output.weight` is omitted or shares the exact offset of `token_embd.weight`.
    #[must_use]
    pub fn has_tied_embeddings(&self) -> bool {
        let Some(embd) = self.find_tensor("token_embd.weight") else {
            return false;
        };
        match self.find_tensor("output.weight") {
            None => true,
            Some(out) => {
                out.absolute_offset() == embd.absolute_offset() && out.byte_len() == embd.byte_len()
            }
        }
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
    metadata_map: BTreeMap<&'a str, GgufValue<'a>>,
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
    #[allow(clippy::too_many_lines)]
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
        let head_count_kv_val = metadata_map.get(head_count_kv_key.as_str());
        let head_count_kv_per_layer = head_count_kv_val
            .and_then(GgufValue::as_usize_array)
            .unwrap_or_default();
        let head_count_kv = head_count_kv_val
            .and_then(GgufValue::as_usize)
            .or_else(|| head_count_kv_per_layer.iter().copied().max())
            .unwrap_or(head_count);

        let ctx_len_key = format!("{architecture}.context_length");
        let context_length = metadata_map
            .get(ctx_len_key.as_str())
            .and_then(GgufValue::as_usize)
            .unwrap_or(2048);

        let dory = Self::parse_dory_metadata(&architecture, &metadata_map)?;

        let layer_count = if layer_count == 0 {
            dory.as_ref()
                .map_or(0, |d| d.physical_layer_count() as usize)
        } else {
            layer_count
        };

        // 6. Parse raw tensor info descriptors
        let raw_tensors =
            Self::parse_raw_tensors(&mut cursor, tensor_count, &architecture, buffer.len())?;

        // 7. Calculate data section starting offset aligned to `alignment`
        let tensor_data_offset = align_up(cursor.position(), alignment);

        // 8. Build verified tensor descriptors and total weight bytes
        let parsed_tensors =
            Self::build_tensor_descriptors(raw_tensors, tensor_data_offset, buffer.len())?;

        let lookup_usize = |suffix: &str| -> Option<usize> {
            let key = format!("{architecture}.{suffix}");
            metadata_map
                .get(key.as_str())
                .and_then(GgufValue::as_usize)
                .filter(|&v| v > 0)
        };

        let mut kv_lora_rank = lookup_usize("attention.kv_lora_rank");
        let mut qk_rope_head_dim = lookup_usize("attention.key_length_mla").or_else(|| {
            if architecture == "deepseek2" {
                lookup_usize("rope.dimension_count")
            } else {
                None
            }
        });
        if kv_lora_rank.is_none()
            && let Some(norm_desc) = parsed_tensors
                .descriptors
                .iter()
                .find(|t| t.name == "blk.0.attn_kv_a_norm.weight")
            && norm_desc.shape.len() == 1
            && norm_desc.shape[0] > 0
        {
            kv_lora_rank = Some(norm_desc.shape[0]);
        }
        if qk_rope_head_dim.is_none()
            && let Some(rank) = kv_lora_rank
            && let Some(mqa_desc) = parsed_tensors
                .descriptors
                .iter()
                .find(|t| t.name == "blk.0.attn_kv_a_mqa.weight")
            && mqa_desc.shape.len() == 2
            && mqa_desc.shape[1] > rank
        {
            qk_rope_head_dim = Some(mqa_desc.shape[1] - rank);
        }

        let q_lora_rank = lookup_usize("attention.q_lora_rank");
        let key_head_dim = lookup_usize("attention.key_length");
        let v_head_dim = lookup_usize("attention.value_length");
        let expert_count = lookup_usize("expert_count");
        let expert_used_count = lookup_usize("expert_used_count");
        let expert_shared_count = lookup_usize("expert_shared_count");
        let expert_feed_forward_length = lookup_usize("expert_feed_forward_length");

        let lookup_f32 = |suffix: &str| -> Option<f32> {
            let key = format!("{architecture}.{suffix}");
            metadata_map
                .get(key.as_str())
                .and_then(GgufValue::as_f32)
                .filter(|v| v.is_finite() && *v > 0.0)
        };

        let ssm_conv_kernel = lookup_usize("ssm.conv_kernel");
        let ssm_inner_size = lookup_usize("ssm.inner_size");
        let ssm_state_size = lookup_usize("ssm.state_size");
        let ssm_time_step_rank = lookup_usize("ssm.time_step_rank");
        let ssm_group_count = lookup_usize("ssm.group_count");
        let shortconv_l_cache = lookup_usize("shortconv.l_cache");
        let wkv_head_size = lookup_usize("wkv.head_size");
        let embedding_scale = lookup_f32("embedding_scale");
        let residual_scale = lookup_f32("residual_scale");
        let attention_scale = lookup_f32("attention.scale");
        let logit_scale = lookup_f32("logit_scale");

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
            dory,
            kv_lora_rank,
            qk_rope_head_dim,
            q_lora_rank,
            qk_nope_head_dim: key_head_dim,
            v_head_dim,
            expert_count,
            expert_used_count,
            expert_shared_count,
            expert_feed_forward_length,
            head_count_kv_per_layer,
            ssm_conv_kernel,
            ssm_inner_size,
            ssm_state_size,
            ssm_time_step_rank,
            ssm_group_count,
            shortconv_l_cache,
            wkv_head_size,
            embedding_scale,
            residual_scale,
            attention_scale,
            logit_scale,
        };

        Ok(Self {
            buffer,
            metadata,
            metadata_map,
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

    #[allow(clippy::too_many_lines)]
    fn parse_dory_metadata(
        architecture: &str,
        metadata_map: &BTreeMap<&str, GgufValue<'_>>,
    ) -> Result<Option<DoryMetadata>, K3Error> {
        if architecture != "dory" {
            return Ok(None);
        }

        let input_layer_count = match metadata_map.get("dory.input_layer_count") {
            Some(val) => val.as_u32().ok_or_else(|| K3Error::CapabilityMismatch {
                architecture: "dory".to_owned(),
                reason: "dory.input_layer_count must be a non-negative unsigned integer".to_owned(),
            })?,
            None => {
                return Err(K3Error::CapabilityMismatch {
                    architecture: "dory".to_owned(),
                    reason: "missing required metadata key: dory.input_layer_count".to_owned(),
                });
            }
        };

        let recurrent_layer_count = match metadata_map.get("dory.recurrent_layer_count") {
            Some(val) => val.as_u32().ok_or_else(|| K3Error::CapabilityMismatch {
                architecture: "dory".to_owned(),
                reason: "dory.recurrent_layer_count must be a non-negative unsigned integer"
                    .to_owned(),
            })?,
            None => {
                return Err(K3Error::CapabilityMismatch {
                    architecture: "dory".to_owned(),
                    reason: "missing required metadata key: dory.recurrent_layer_count".to_owned(),
                });
            }
        };

        let output_layer_count = match metadata_map.get("dory.output_layer_count") {
            Some(val) => val.as_u32().ok_or_else(|| K3Error::CapabilityMismatch {
                architecture: "dory".to_owned(),
                reason: "dory.output_layer_count must be a non-negative unsigned integer"
                    .to_owned(),
            })?,
            None => {
                return Err(K3Error::CapabilityMismatch {
                    architecture: "dory".to_owned(),
                    reason: "missing required metadata key: dory.output_layer_count".to_owned(),
                });
            }
        };

        let layer_pattern = match metadata_map.get("dory.layer_pattern") {
            Some(val) => {
                let s = val.as_str().ok_or_else(|| K3Error::CapabilityMismatch {
                    architecture: "dory".to_owned(),
                    reason: "dory.layer_pattern must be a string".to_owned(),
                })?;
                if s.is_empty() {
                    return Err(K3Error::CapabilityMismatch {
                        architecture: "dory".to_owned(),
                        reason: "dory.layer_pattern must not be empty".to_owned(),
                    });
                }
                s.to_owned()
            }
            None => {
                return Err(K3Error::CapabilityMismatch {
                    architecture: "dory".to_owned(),
                    reason: "missing required metadata key: dory.layer_pattern".to_owned(),
                });
            }
        };

        let num_loops = match metadata_map.get("dory.num_loops") {
            Some(val) => {
                let n = val.as_u32().ok_or_else(|| K3Error::CapabilityMismatch {
                    architecture: "dory".to_owned(),
                    reason: "dory.num_loops must be a positive integer".to_owned(),
                })?;
                if n == 0 {
                    return Err(K3Error::CapabilityMismatch {
                        architecture: "dory".to_owned(),
                        reason: "dory.num_loops must be greater than zero".to_owned(),
                    });
                }
                n
            }
            None => 1,
        };

        let recurrent_kv_cache_mode = match metadata_map.get("dory.recurrent_kv_cache_mode") {
            Some(val) => {
                let s = val.as_str().ok_or_else(|| K3Error::CapabilityMismatch {
                    architecture: "dory".to_owned(),
                    reason: "dory.recurrent_kv_cache_mode must be a string".to_owned(),
                })?;
                DoryKvCacheMode::parse_mode(s)?
            }
            None => DoryKvCacheMode::PerLoop,
        };

        let rope_profile = match metadata_map.get("dory.rope.profile") {
            Some(val) => {
                if let Some(s) = val.as_str() {
                    s.to_owned()
                } else if let Some(u) = val.as_u32() {
                    format!("profile_{u}")
                } else {
                    return Err(K3Error::CapabilityMismatch {
                        architecture: "dory".to_owned(),
                        reason: "dory.rope.profile must be a string or integer".to_owned(),
                    });
                }
            }
            None => "dual_theta".to_owned(),
        };

        let rope_freq_base_2 = match metadata_map.get("dory.rope.freq_base_2") {
            Some(val) => val.as_f32().ok_or_else(|| K3Error::CapabilityMismatch {
                architecture: "dory".to_owned(),
                reason: "dory.rope.freq_base_2 must be a float".to_owned(),
            })?,
            None => 10000.0,
        };

        let init_std = match metadata_map.get("dory.init_std") {
            Some(val) => val.as_f32().ok_or_else(|| K3Error::CapabilityMismatch {
                architecture: "dory".to_owned(),
                reason: "dory.init_std must be a float".to_owned(),
            })?,
            None => 0.02,
        };

        let alpha_init = match metadata_map.get("dory.alpha_init") {
            Some(val) => val.as_f32().ok_or_else(|| K3Error::CapabilityMismatch {
                architecture: "dory".to_owned(),
                reason: "dory.alpha_init must be a float".to_owned(),
            })?,
            None => 1.0,
        };

        let gate_init = match metadata_map.get("dory.gate_init") {
            Some(val) => val.as_f32().ok_or_else(|| K3Error::CapabilityMismatch {
                architecture: "dory".to_owned(),
                reason: "dory.gate_init must be a float".to_owned(),
            })?,
            None => 0.0,
        };

        Ok(Some(DoryMetadata {
            input_layer_count,
            recurrent_layer_count,
            output_layer_count,
            num_loops,
            recurrent_kv_cache_mode,
            layer_pattern,
            rope_profile,
            rope_freq_base_2,
            init_std,
            alpha_init,
            gate_init,
        }))
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
        let mut seen_spans = BTreeSet::new();
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

            if seen_spans.insert((absolute_offset, raw.byte_len)) {
                total_weight_bytes =
                    total_weight_bytes
                        .checked_add(raw.byte_len)
                        .ok_or_else(|| K3Error::TensorBoundsError {
                            tensor_name: raw.name.clone(),
                            offset: absolute_offset,
                            byte_len: raw.byte_len,
                            buffer_len,
                            reason: "total weight bytes overflowed usize".to_owned(),
                        })?;
            }

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

    /// Returns a parsed GGUF metadata value by key, if present.
    #[must_use]
    pub fn metadata_value(&self, key: &str) -> Option<&GgufValue<'a>> {
        self.metadata_map.get(key)
    }

    /// Returns the full map of parsed GGUF metadata key-value entries.
    #[must_use]
    pub const fn metadata_entries(&self) -> &BTreeMap<&'a str, GgufValue<'a>> {
        &self.metadata_map
    }

    /// Returns the model architecture string identifier.
    #[must_use]
    pub fn architecture(&self) -> &str {
        self.metadata.architecture()
    }

    /// Returns the parsed Dory-specific metadata if this model uses the `"dory"` architecture.
    #[must_use]
    pub fn dory(&self) -> Option<&DoryMetadata> {
        self.metadata.dory()
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
        GGUF_MAGIC, GGUF_VERSION_V3, GgufArray, GgufMetadataValueType, GgufValue, QuantType,
        align_up,
    };

    /// Test helper that constructs a synthetic valid GGUF v3 byte buffer with default
    /// hyperparameters and embedded vocabulary/chat-template metadata.
    #[must_use]
    #[allow(clippy::missing_panics_doc)]
    pub fn create_synthetic_gguf_v3(
        architecture: &str,
        alignment: usize,
        tensor_specs: &[(&str, QuantType, &[usize])],
    ) -> Vec<u8> {
        create_synthetic_gguf_v3_with_kvs(architecture, alignment, &[], tensor_specs)
    }

    /// Test helper that constructs a synthetic valid GGUF v3 byte buffer with custom
    /// additional or overriding metadata key-value pairs.
    #[must_use]
    #[allow(clippy::missing_panics_doc, clippy::too_many_lines)]
    pub fn create_synthetic_gguf_v3_with_kvs(
        architecture: &str,
        alignment: usize,
        extra_kvs: &[(&str, GgufValue<'_>)],
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

        let default_tokens = vec![
            "<pad>",
            "<bos>",
            "<eos>",
            "<|im_start|>",
            "<|im_end|>",
            "user",
            "assistant",
            "\n",
            " ",
            "hello",
            "world",
            "Explain",
            "RVV",
            "1.0",
            "SpacemiT",
            "K3",
        ];
        let default_scores = vec![0.0_f32; default_tokens.len()];
        let default_token_types = vec![3_i32, 3, 3, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1];
        let default_merges = vec!["h ello", "w orld"];

        let mut kv_pairs: Vec<(&str, GgufValue<'_>)> = vec![
            ("general.architecture", GgufValue::String(architecture)),
            ("general.alignment", GgufValue::Uint32(alignment_u32)),
            (&k_block, GgufValue::Uint32(4)),
            (&k_emb, GgufValue::Uint32(256)),
            (&k_head, GgufValue::Uint32(4)),
            (&k_ctx, GgufValue::Uint32(2048)),
            ("tokenizer.ggml.model", GgufValue::String("gpt2")),
            (
                "tokenizer.ggml.tokens",
                GgufValue::Array(GgufArray::String(default_tokens)),
            ),
            (
                "tokenizer.ggml.scores",
                GgufValue::Array(GgufArray::Float32(default_scores)),
            ),
            (
                "tokenizer.ggml.token_type",
                GgufValue::Array(GgufArray::Int32(default_token_types)),
            ),
            (
                "tokenizer.ggml.merges",
                GgufValue::Array(GgufArray::String(default_merges)),
            ),
            ("tokenizer.ggml.bos_token_id", GgufValue::Uint32(1)),
            ("tokenizer.ggml.eos_token_id", GgufValue::Uint32(2)),
            ("tokenizer.ggml.padding_token_id", GgufValue::Uint32(0)),
            ("tokenizer.ggml.add_bos_token", GgufValue::Bool(true)),
            (
                "tokenizer.chat_template",
                GgufValue::String(
                    "<|im_start|>user\n{{prompt}}<|im_end|>\n<|im_start|>assistant\n",
                ),
            ),
        ];

        for &(extra_k, ref extra_v) in extra_kvs {
            if let Some(pos) = kv_pairs.iter().position(|(k, _)| *k == extra_k) {
                kv_pairs[pos] = (extra_k, extra_v.clone());
            } else {
                kv_pairs.push((extra_k, extra_v.clone()));
            }
        }

        buf.extend_from_slice(&tensor_count.to_le_bytes());
        let kv_len = u64::try_from(kv_pairs.len()).expect("kv count fits u64");
        buf.extend_from_slice(&kv_len.to_le_bytes());

        // 3. Write KV pairs
        for (k, v) in kv_pairs {
            write_string(&mut buf, k);
            write_gguf_value(&mut buf, &v);
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
                QuantType::Q5_K => 13,
                QuantType::Q6_K => 14,
                QuantType::BF16 => 30,
                QuantType::TQ1_0 => 34,
                QuantType::TQ2_0 => 35,
            };
            buf.extend_from_slice(&ggml_type_id.to_le_bytes());
            let rel_offset_u64 = u64::try_from(rel_offset).expect("rel_offset fits u64");
            buf.extend_from_slice(&rel_offset_u64.to_le_bytes());

            // Populate deterministic non-zero bytes so dequantization / dot-products are non-trivial
            let mut chunk = vec![0u8; byte_len];
            for (idx, b) in chunk.iter_mut().enumerate() {
                #[allow(clippy::cast_possible_truncation)]
                {
                    *b = ((idx * 31 + 17) & 0x7F) as u8;
                }
            }
            tensor_data_chunks.push(chunk);
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

    /// Serializes a [`GgufValue`] into `buf` in GGUF v3 binary format.
    #[allow(clippy::missing_panics_doc)]
    pub fn write_gguf_value(buf: &mut Vec<u8>, val: &GgufValue<'_>) {
        match val {
            GgufValue::Uint8(v) => {
                buf.extend_from_slice(&(GgufMetadataValueType::Uint8 as u32).to_le_bytes());
                buf.push(*v);
            }
            GgufValue::Int8(v) => {
                buf.extend_from_slice(&(GgufMetadataValueType::Int8 as u32).to_le_bytes());
                buf.extend_from_slice(&v.to_le_bytes());
            }
            GgufValue::Uint16(v) => {
                buf.extend_from_slice(&(GgufMetadataValueType::Uint16 as u32).to_le_bytes());
                buf.extend_from_slice(&v.to_le_bytes());
            }
            GgufValue::Int16(v) => {
                buf.extend_from_slice(&(GgufMetadataValueType::Int16 as u32).to_le_bytes());
                buf.extend_from_slice(&v.to_le_bytes());
            }
            GgufValue::Uint32(u) => {
                buf.extend_from_slice(&(GgufMetadataValueType::Uint32 as u32).to_le_bytes());
                buf.extend_from_slice(&u.to_le_bytes());
            }
            GgufValue::Int32(i) => {
                buf.extend_from_slice(&(GgufMetadataValueType::Int32 as u32).to_le_bytes());
                buf.extend_from_slice(&i.to_le_bytes());
            }
            GgufValue::Float32(f) => {
                buf.extend_from_slice(&(GgufMetadataValueType::Float32 as u32).to_le_bytes());
                buf.extend_from_slice(&f.to_le_bytes());
            }
            GgufValue::Bool(b) => {
                buf.extend_from_slice(&(GgufMetadataValueType::Bool as u32).to_le_bytes());
                buf.push(u8::from(*b));
            }
            GgufValue::String(s) => {
                buf.extend_from_slice(&(GgufMetadataValueType::String as u32).to_le_bytes());
                write_string(buf, s);
            }
            GgufValue::Uint64(u) => {
                buf.extend_from_slice(&(GgufMetadataValueType::Uint64 as u32).to_le_bytes());
                buf.extend_from_slice(&u.to_le_bytes());
            }
            GgufValue::Int64(i) => {
                buf.extend_from_slice(&(GgufMetadataValueType::Int64 as u32).to_le_bytes());
                buf.extend_from_slice(&i.to_le_bytes());
            }
            GgufValue::Float64(f) => {
                buf.extend_from_slice(&(GgufMetadataValueType::Float64 as u32).to_le_bytes());
                buf.extend_from_slice(&f.to_le_bytes());
            }
            GgufValue::Array(arr) => {
                buf.extend_from_slice(&(GgufMetadataValueType::Array as u32).to_le_bytes());
                write_gguf_array(buf, arr);
            }
        }
    }

    fn write_gguf_array(buf: &mut Vec<u8>, arr: &GgufArray<'_>) {
        match arr {
            GgufArray::String(items) => {
                buf.extend_from_slice(&(GgufMetadataValueType::String as u32).to_le_bytes());
                let len = u64::try_from(items.len()).expect("len fits u64");
                buf.extend_from_slice(&len.to_le_bytes());
                for s in items {
                    write_string(buf, s);
                }
            }
            GgufArray::Float32(items) => {
                buf.extend_from_slice(&(GgufMetadataValueType::Float32 as u32).to_le_bytes());
                let len = u64::try_from(items.len()).expect("len fits u64");
                buf.extend_from_slice(&len.to_le_bytes());
                for f in items {
                    buf.extend_from_slice(&f.to_le_bytes());
                }
            }
            GgufArray::Int32(items) => {
                buf.extend_from_slice(&(GgufMetadataValueType::Int32 as u32).to_le_bytes());
                let len = u64::try_from(items.len()).expect("len fits u64");
                buf.extend_from_slice(&len.to_le_bytes());
                for i in items {
                    buf.extend_from_slice(&i.to_le_bytes());
                }
            }
            GgufArray::Uint32(items) => {
                buf.extend_from_slice(&(GgufMetadataValueType::Uint32 as u32).to_le_bytes());
                let len = u64::try_from(items.len()).expect("len fits u64");
                buf.extend_from_slice(&len.to_le_bytes());
                for u in items {
                    buf.extend_from_slice(&u.to_le_bytes());
                }
            }
            GgufArray::Uint8(items) => {
                buf.extend_from_slice(&(GgufMetadataValueType::Uint8 as u32).to_le_bytes());
                let len = u64::try_from(items.len()).expect("len fits u64");
                buf.extend_from_slice(&len.to_le_bytes());
                buf.extend_from_slice(items);
            }
            GgufArray::Int8(items) => {
                buf.extend_from_slice(&(GgufMetadataValueType::Int8 as u32).to_le_bytes());
                let len = u64::try_from(items.len()).expect("len fits u64");
                buf.extend_from_slice(&len.to_le_bytes());
                for i in items {
                    buf.extend_from_slice(&i.to_le_bytes());
                }
            }
            GgufArray::Uint16(items) => {
                buf.extend_from_slice(&(GgufMetadataValueType::Uint16 as u32).to_le_bytes());
                let len = u64::try_from(items.len()).expect("len fits u64");
                buf.extend_from_slice(&len.to_le_bytes());
                for u in items {
                    buf.extend_from_slice(&u.to_le_bytes());
                }
            }
            GgufArray::Int16(items) => {
                buf.extend_from_slice(&(GgufMetadataValueType::Int16 as u32).to_le_bytes());
                let len = u64::try_from(items.len()).expect("len fits u64");
                buf.extend_from_slice(&len.to_le_bytes());
                for i in items {
                    buf.extend_from_slice(&i.to_le_bytes());
                }
            }
            GgufArray::Bool(items) => {
                buf.extend_from_slice(&(GgufMetadataValueType::Bool as u32).to_le_bytes());
                let len = u64::try_from(items.len()).expect("len fits u64");
                buf.extend_from_slice(&len.to_le_bytes());
                for b in items {
                    buf.push(u8::from(*b));
                }
            }
            GgufArray::Uint64(items) => {
                buf.extend_from_slice(&(GgufMetadataValueType::Uint64 as u32).to_le_bytes());
                let len = u64::try_from(items.len()).expect("len fits u64");
                buf.extend_from_slice(&len.to_le_bytes());
                for u in items {
                    buf.extend_from_slice(&u.to_le_bytes());
                }
            }
            GgufArray::Int64(items) => {
                buf.extend_from_slice(&(GgufMetadataValueType::Int64 as u32).to_le_bytes());
                let len = u64::try_from(items.len()).expect("len fits u64");
                buf.extend_from_slice(&len.to_le_bytes());
                for i in items {
                    buf.extend_from_slice(&i.to_le_bytes());
                }
            }
            GgufArray::Float64(items) => {
                buf.extend_from_slice(&(GgufMetadataValueType::Float64 as u32).to_le_bytes());
                let len = u64::try_from(items.len()).expect("len fits u64");
                buf.extend_from_slice(&len.to_le_bytes());
                for f in items {
                    buf.extend_from_slice(&f.to_le_bytes());
                }
            }
        }
    }

    fn write_string(buf: &mut Vec<u8>, s: &str) {
        let len_u64 = u64::try_from(s.len()).expect("string length fits u64");
        buf.extend_from_slice(&len_u64.to_le_bytes());
        buf.extend_from_slice(s.as_bytes());
    }
}

#[cfg(test)]
mod tests {
    use super::test_helpers::{create_synthetic_gguf_v3, create_synthetic_gguf_v3_with_kvs};
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

    #[test]
    fn test_tq1_0_tq2_0_bounds() {
        use crate::profile::QuantizationInfo;

        // 1. Synthesize a valid GGUF v3 file containing TQ1_0, TQ2_0, and F32 tensors
        // - "blk.0.attn_q.weight": TQ1_0, shape [256, 2] -> 512 elements = 2 blocks = 108 bytes
        // - "blk.0.attn_k.weight": TQ2_0, shape [256, 4] -> 1024 elements = 4 blocks = 264 bytes
        // - "output_norm.weight":  F32,   shape [16]     -> 16 elements = 64 bytes
        let tensor_specs = [
            ("blk.0.attn_q.weight", QuantType::TQ1_0, &[256, 2][..]),
            ("blk.0.attn_k.weight", QuantType::TQ2_0, &[256, 4][..]),
            ("output_norm.weight", QuantType::F32, &[16][..]),
        ];
        let bytes = create_synthetic_gguf_v3("bitnet", 32, &tensor_specs);

        // 2. Parse file
        let gguf = GgufFile::parse(&bytes).expect("GGUF v3 parsing succeeds for TQ1_0 & TQ2_0");
        let meta = gguf.metadata();

        assert_eq!(meta.architecture(), "bitnet");
        assert_eq!(meta.alignment(), 32);
        assert_eq!(meta.tensor_count(), 3);
        assert_eq!(meta.total_weight_bytes(), 108 + 264 + 64);

        // 3. Verify ModelWeights trait retrieval
        let q_view = gguf
            .tensor("blk.0.attn_q.weight")
            .expect("attn_q tensor found");
        assert_eq!(q_view.name(), "blk.0.attn_q.weight");
        assert_eq!(q_view.quant(), QuantType::TQ1_0);
        assert_eq!(q_view.shape(), &[256, 2]);
        assert_eq!(q_view.num_elements(), 512);
        assert_eq!(q_view.byte_len(), 108);

        let k_view = gguf
            .tensor("blk.0.attn_k.weight")
            .expect("attn_k tensor found");
        assert_eq!(k_view.name(), "blk.0.attn_k.weight");
        assert_eq!(k_view.quant(), QuantType::TQ2_0);
        assert_eq!(k_view.shape(), &[256, 4]);
        assert_eq!(k_view.num_elements(), 1024);
        assert_eq!(k_view.byte_len(), 264);

        let norm_view = gguf
            .tensor("output_norm.weight")
            .expect("output_norm tensor found");
        assert_eq!(norm_view.quant(), QuantType::F32);
        assert_eq!(norm_view.byte_len(), 64);

        // 4. Verify QuantizationInfo integration
        let q_info = QuantizationInfo::from_descriptors(meta.tensors());
        // Primary format has highest aggregate byte footprint: TQ2_0 (264) > TQ1_0 (108) > F32 (64)
        assert_eq!(q_info.primary, "TQ2_0");
        assert_eq!(q_info.summary.get("TQ1_0"), Some(&1));
        assert_eq!(q_info.summary.get("TQ2_0"), Some(&1));
        assert_eq!(q_info.summary.get("F32"), Some(&1));
        assert_eq!(q_info.tensor_bytes.get("TQ1_0"), Some(&108));
        assert_eq!(q_info.tensor_bytes.get("TQ2_0"), Some(&264));
        assert_eq!(q_info.tensor_bytes.get("F32"), Some(&64));

        // 5. Boundary Condition: Truncated GGUF buffer (missing 1 byte of payload)
        let mut truncated_bytes = bytes.clone();
        truncated_bytes.truncate(bytes.len() - 1);
        let err_trunc = GgufFile::parse(&truncated_bytes).unwrap_err();
        assert!(matches!(err_trunc, K3Error::TensorBoundsError { .. }));
        assert!(err_trunc.to_string().contains("exceeds buffer length"));

        // 6. Nonexistent tensor lookup
        assert!(gguf.tensor("nonexistent").is_err());
    }

    #[test]
    fn test_parse_dory_metadata_unit() {
        let layer_pattern =
            "S-S-S-*-S-S-S-*-S-S-S-*-S-S-S-*-S-S-S-*-S-S-S-*-S-S-S-*-S-S-S-*-S-S-S-*-";
        let extra_kvs = [
            ("dory.block_count", GgufValue::Uint32(0)),
            ("dory.input_layer_count", GgufValue::Uint32(16)),
            ("dory.recurrent_layer_count", GgufValue::Uint32(40)),
            ("dory.output_layer_count", GgufValue::Uint32(16)),
            ("dory.num_loops", GgufValue::Uint32(2)),
            (
                "dory.recurrent_kv_cache_mode",
                GgufValue::String("per_loop"),
            ),
            ("dory.layer_pattern", GgufValue::String(layer_pattern)),
            ("dory.rope.profile", GgufValue::String("neox")),
            ("dory.rope.freq_base_2", GgufValue::Float32(10000.0)),
            ("dory.init_std", GgufValue::Float32(0.02)),
            ("dory.alpha_init", GgufValue::Float32(0.05)),
            ("dory.gate_init", GgufValue::Float32(0.0)),
        ];

        let bytes = create_synthetic_gguf_v3_with_kvs("dory", 32, &extra_kvs, &[]);
        let gguf = GgufFile::parse(&bytes).expect("parse dory gguf v3");
        let meta = gguf.metadata();

        assert_eq!(meta.architecture(), "dory");
        assert_eq!(meta.layer_count(), 72);

        let dory = meta.dory().expect("dory metadata must be present");
        assert_eq!(dory.input_layer_count(), 16);
        assert_eq!(dory.recurrent_layer_count(), 40);
        assert_eq!(dory.output_layer_count(), 16);
        assert_eq!(dory.num_loops(), 2);
        assert_eq!(dory.recurrent_kv_cache_mode(), DoryKvCacheMode::PerLoop);
        assert_eq!(dory.recurrent_kv_cache_mode_str(), "per_loop");
        assert_eq!(dory.rope_profile(), "neox");
        assert!((dory.rope_freq_base_2() - 10000.0).abs() < 1e-4);
        assert!((dory.init_std() - 0.02).abs() < 1e-4);
        assert!((dory.alpha_init() - 0.05).abs() < 1e-4);
        assert!((dory.gate_init() - 0.0).abs() < 1e-4);

        // Structural layer count invariants
        assert_eq!(dory.physical_layer_count(), 72);
        assert_eq!(dory.total_physical_layers(), 72);
        assert_eq!(dory.total_layer_count(), 72);
        assert_eq!(dory.effective_layer_count(), 112); // 16 + (40 * 2) + 16
        assert_eq!(dory.total_virtual_layers(), 112);
        assert_eq!(dory.virtual_layer_count(4), 192); // 16 + (40 * 4) + 16

        // Segment classification
        assert!(dory.is_input_layer(0));
        assert!(dory.is_input_layer(15));
        assert!(!dory.is_input_layer(16));

        assert!(dory.is_recurrent_layer(16));
        assert!(dory.is_recurrent_layer(55));
        assert!(!dory.is_recurrent_layer(56));

        assert!(dory.is_output_layer(56));
        assert!(dory.is_output_layer(71));
        assert!(!dory.is_output_layer(72));

        // Virtual-to-physical loop mapping
        assert_eq!(dory.physical_layer_for_virtual_step(0), Some(0));
        assert_eq!(dory.physical_layer_for_virtual_step(15), Some(15));
        assert_eq!(dory.physical_layer_for_virtual_step(16), Some(16)); // loop 0
        assert_eq!(dory.physical_layer_for_virtual_step(55), Some(55)); // loop 0
        assert_eq!(dory.physical_layer_for_virtual_step(56), Some(16)); // loop 1
        assert_eq!(dory.physical_layer_for_virtual_step(95), Some(55)); // loop 1
        assert_eq!(dory.physical_layer_for_virtual_step(96), Some(56)); // output
        assert_eq!(dory.physical_layer_for_virtual_step(111), Some(71)); // output
        assert_eq!(dory.physical_layer_for_virtual_step(112), None);

        // Recurrent loop indexing
        assert_eq!(dory.loop_index_for_virtual_step(5), None);
        assert_eq!(dory.loop_index_for_virtual_step(16), Some(0));
        assert_eq!(dory.loop_index_for_virtual_step(56), Some(1));
        assert_eq!(dory.loop_index_for_virtual_step(96), None);

        // Layer type classification
        assert_eq!(
            dory.layer_type(0),
            Some(DoryLayerType::SlidingWindowAttention)
        );
        assert_eq!(dory.layer_type(1), Some(DoryLayerType::Mlp));
        assert_eq!(dory.layer_type(6), Some(DoryLayerType::FullAttention));
        assert!(dory.is_attention_layer(0));
        assert!(dory.is_mlp_layer(1));

        // Display check
        let display_str = format!("{dory}");
        assert!(display_str.contains("DoryMetadata"));
        assert!(display_str.contains("kv_mode='per_loop'"));
    }

    #[test]
    fn test_non_dory_metadata_is_none() {
        let bytes = create_synthetic_gguf_v3("synthetic", 32, &[]);
        let gguf = GgufFile::parse(&bytes).expect("parse synthetic gguf");
        assert!(gguf.metadata().dory().is_none());
        assert!(gguf.dory().is_none());
    }

    #[test]
    fn test_model_metadata_mla_moe_and_tied_embeddings() {
        // 1. Explicit DeepSeek2 MLA + MoE metadata extraction
        let extra_kvs = [
            ("deepseek2.block_count", GgufValue::Uint32(27)),
            ("deepseek2.attention.kv_lora_rank", GgufValue::Uint32(512)),
            ("deepseek2.attention.q_lora_rank", GgufValue::Uint32(1536)),
            ("deepseek2.attention.key_length", GgufValue::Uint32(128)),
            ("deepseek2.attention.value_length", GgufValue::Uint32(128)),
            ("deepseek2.rope.dimension_count", GgufValue::Uint32(64)),
            ("deepseek2.expert_count", GgufValue::Uint32(64)),
            ("deepseek2.expert_used_count", GgufValue::Uint32(6)),
            ("deepseek2.expert_shared_count", GgufValue::Uint32(2)),
        ];
        let tensor_specs = [("token_embd.weight", QuantType::Q8_0, &[32, 16][..])];
        let bytes = create_synthetic_gguf_v3_with_kvs("deepseek2", 32, &extra_kvs, &tensor_specs);
        let gguf = GgufFile::parse(&bytes).expect("parse deepseek2 gguf");
        let meta = gguf.metadata();
        assert_eq!(meta.kv_lora_rank(), Some(512));
        assert_eq!(meta.qk_rope_head_dim(), Some(64));
        assert_eq!(meta.q_lora_rank(), Some(1536));
        assert_eq!(meta.qk_nope_head_dim(), Some(128));
        assert_eq!(meta.v_head_dim(), Some(128));
        assert_eq!(meta.expert_count(), Some(64));
        assert_eq!(meta.expert_used_count(), Some(6));
        assert_eq!(meta.expert_shared_count(), Some(2));
        assert!(meta.has_tied_embeddings());

        // 2. Fallback inference of kv_lora_rank and qk_rope_head_dim from tensor shapes
        let fallback_specs = [
            ("blk.0.attn_kv_a_norm.weight", QuantType::F32, &[16][..]),
            ("blk.0.attn_kv_a_mqa.weight", QuantType::F32, &[32, 24][..]),
        ];
        let fb_bytes = create_synthetic_gguf_v3("deepseek2", 32, &fallback_specs);
        let fb_gguf = GgufFile::parse(&fb_bytes).expect("parse fallback deepseek2");
        assert_eq!(fb_gguf.metadata().kv_lora_rank(), Some(16));
        assert_eq!(fb_gguf.metadata().qk_rope_head_dim(), Some(8));
        assert!(!fb_gguf.metadata().has_tied_embeddings());
    }

    #[test]
    fn test_gguf_ssm_rwkv_lfm2_granite_metadata_parsing() {
        // 1. Mamba2 pure SSM metadata (zero attention heads)
        let mamba2_kvs = [
            ("mamba2.block_count", GgufValue::Uint32(48)),
            ("mamba2.embedding_length", GgufValue::Uint32(2560)),
            ("mamba2.attention.head_count", GgufValue::Uint32(0)),
            ("mamba2.attention.head_count_kv", GgufValue::Uint32(0)),
            ("mamba2.ssm.conv_kernel", GgufValue::Uint32(4)),
            ("mamba2.ssm.inner_size", GgufValue::Uint32(5120)),
            ("mamba2.ssm.state_size", GgufValue::Uint32(128)),
            ("mamba2.ssm.time_step_rank", GgufValue::Uint32(80)),
            ("mamba2.ssm.group_count", GgufValue::Uint32(1)),
        ];
        let mamba2_bytes = create_synthetic_gguf_v3_with_kvs("mamba2", 16, &mamba2_kvs, &[]);
        let mamba2_gguf = GgufFile::parse(&mamba2_bytes).expect("parse mamba2 gguf");
        let m2 = mamba2_gguf.metadata();
        assert_eq!(m2.head_count_kv(), 0);
        assert_eq!(m2.ssm_conv_kernel(), Some(4));
        assert_eq!(m2.ssm_inner_size(), Some(5120));
        assert_eq!(m2.ssm_state_size(), Some(128));
        assert_eq!(m2.ssm_time_step_rank(), Some(80));
        assert_eq!(m2.ssm_group_count(), Some(1));

        // 2. RWKV7 WKV head size
        let rwkv7_kvs = [
            ("rwkv7.block_count", GgufValue::Uint32(24)),
            ("rwkv7.embedding_length", GgufValue::Uint32(2048)),
            ("rwkv7.wkv.head_size", GgufValue::Uint32(64)),
        ];
        let rwkv7_bytes = create_synthetic_gguf_v3_with_kvs("rwkv7", 16, &rwkv7_kvs, &[]);
        let rwkv7_gguf = GgufFile::parse(&rwkv7_bytes).expect("parse rwkv7 gguf");
        assert_eq!(rwkv7_gguf.metadata().wkv_head_size(), Some(64));

        // 3. LFM2 MoE with per-layer head_count_kv array and shortconv l_cache
        let lfm2_kvs = [
            ("lfm2moe.block_count", GgufValue::Uint32(6)),
            ("lfm2moe.embedding_length", GgufValue::Uint32(2048)),
            ("lfm2moe.attention.head_count", GgufValue::Uint32(32)),
            (
                "lfm2moe.attention.head_count_kv",
                GgufValue::Array(GgufArray::Uint32(vec![0, 0, 8, 0, 0, 8])),
            ),
            ("lfm2moe.shortconv.l_cache", GgufValue::Uint32(3)),
            ("lfm2moe.expert_count", GgufValue::Uint32(32)),
            ("lfm2moe.expert_used_count", GgufValue::Uint32(4)),
            (
                "lfm2moe.expert_feed_forward_length",
                GgufValue::Uint32(1792),
            ),
        ];
        let lfm2_bytes = create_synthetic_gguf_v3_with_kvs("lfm2moe", 16, &lfm2_kvs, &[]);
        let lfm2_gguf = GgufFile::parse(&lfm2_bytes).expect("parse lfm2moe gguf");
        let l2 = lfm2_gguf.metadata();
        assert_eq!(l2.head_count_kv(), 8);
        assert_eq!(l2.head_count_kv_per_layer(), &[0, 0, 8, 0, 0, 8]);
        assert_eq!(l2.shortconv_l_cache(), Some(3));
        assert_eq!(l2.expert_count(), Some(32));
        assert_eq!(l2.expert_used_count(), Some(4));
        assert_eq!(l2.expert_feed_forward_length(), Some(1792));

        // 4. Granite multipliers + Granite Hybrid SSM fields
        let granite_kvs = [
            ("granitehybrid.block_count", GgufValue::Uint32(40)),
            ("granitehybrid.embedding_length", GgufValue::Uint32(3072)),
            ("granitehybrid.attention.head_count", GgufValue::Uint32(24)),
            (
                "granitehybrid.attention.head_count_kv",
                GgufValue::Uint32(8),
            ),
            ("granitehybrid.ssm.conv_kernel", GgufValue::Uint32(4)),
            ("granitehybrid.ssm.inner_size", GgufValue::Uint32(6144)),
            ("granitehybrid.ssm.state_size", GgufValue::Uint32(128)),
            ("granitehybrid.ssm.time_step_rank", GgufValue::Uint32(48)),
            ("granitehybrid.ssm.group_count", GgufValue::Uint32(1)),
            ("granitehybrid.embedding_scale", GgufValue::Float32(12.0)),
            ("granitehybrid.residual_scale", GgufValue::Float32(0.22)),
            (
                "granitehybrid.attention.scale",
                GgufValue::Float32(0.007_812_5),
            ),
            ("granitehybrid.logit_scale", GgufValue::Float32(16.0)),
        ];
        let granite_bytes =
            create_synthetic_gguf_v3_with_kvs("granitehybrid", 16, &granite_kvs, &[]);
        let granite_gguf = GgufFile::parse(&granite_bytes).expect("parse granitehybrid gguf");
        let g = granite_gguf.metadata();
        assert_eq!(g.embedding_scale(), Some(12.0));
        assert_eq!(g.residual_scale(), Some(0.22));
        assert_eq!(g.attention_scale(), Some(0.007_812_5));
        assert_eq!(g.logit_scale(), Some(16.0));
        assert_eq!(g.ssm_conv_kernel(), Some(4));
        assert_eq!(g.ssm_inner_size(), Some(6144));
        assert_eq!(g.ssm_state_size(), Some(128));
        assert_eq!(g.ssm_time_step_rank(), Some(48));
        assert_eq!(g.ssm_group_count(), Some(1));
    }
}
