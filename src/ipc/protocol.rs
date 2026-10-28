//! IPC message schemas, Apache Fory serialization codec, and 12-byte UDS framing.
//!
//! Provides binary control plane message serialization over Unix Domain Sockets (UDS)
//! for memory management (`k3-kv` <-> `k3-manager`) and forward pass execution (`k3-manager` <-> `k3-worker`).

#![allow(clippy::used_underscore_binding)]

use core::mem::size_of;
use std::io;
use std::os::fd::{AsRawFd, FromRawFd, OwnedFd, RawFd};
use std::os::unix::net::UnixStream;

use fory::{Fory, ForyEnum, ForyStruct};

use super::shm::IpcError;
use crate::error::K3Error;

/// Standardized IPC return status codes matching K3 process exit codes.
#[derive(ForyEnum, Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[repr(u32)]
pub enum StatusCode {
    /// Command completed successfully.
    #[default]
    Success = 0,
    /// Model architecture or quantization format mismatch (ADR-023).
    CapabilityMismatch = 10,
    /// Required memory footprint exceeds operator `LimitMEMLOCK` bounds (ADR-037).
    InsufficientLockableMemory = 11,
    /// Context length bound exceeded (ADR-027).
    ContextLengthExceeded = 12,
    /// Core affinity mask violation.
    AffinityError = 13,
    /// Zero-copy tensor bounds check failed.
    TensorBoundsError = 14,
    /// Automatic request routing unavailable (US-12).
    RoutingUnavailable = 15,
    /// Requested arena ID was not found.
    ArenaNotFound = 20,
    /// Internal daemon execution failure.
    InternalError = 21,
    /// Framing or protocol violation.
    ProtocolError = 22,
    /// Admission queue saturated or estimated wait exceeded (US-13).
    QueueSaturated = 23,
}

impl StatusCode {
    /// Returns `true` if this status indicates success.
    #[must_use]
    pub const fn is_success(&self) -> bool {
        matches!(self, Self::Success)
    }

    /// Converts status code to corresponding process exit code.
    #[must_use]
    pub const fn as_exit_code(&self) -> u8 {
        *self as u8
    }

    /// Derives [`StatusCode`] from a [`K3Error`].
    #[must_use]
    pub const fn from_k3_error(err: &K3Error) -> Self {
        match err {
            K3Error::CapabilityMismatch { .. } => Self::CapabilityMismatch,
            K3Error::InsufficientLockableMemory { .. } => Self::InsufficientLockableMemory,
            K3Error::ContextLengthExceeded { .. } => Self::ContextLengthExceeded,
            K3Error::AffinityError { .. } => Self::AffinityError,
            K3Error::TensorBoundsError { .. } => Self::TensorBoundsError,
            K3Error::RoutingUnavailable { .. } => Self::RoutingUnavailable,
            K3Error::QueueSaturated { .. } => Self::QueueSaturated,
        }
    }

    /// Derives [`StatusCode`] from an [`IpcError`].
    #[must_use]
    pub fn from_ipc_error(err: &IpcError) -> Self {
        match err {
            IpcError::InsufficientLockableMemory(e) | IpcError::TensorBoundsError(e) => {
                Self::from_k3_error(e)
            }
            IpcError::ProtocolError(_) => Self::ProtocolError,
            IpcError::IoError { .. } => Self::InternalError,
        }
    }

    /// Parses a raw `u32` value into [`StatusCode`].
    #[must_use]
    pub const fn from_u32(val: u32) -> Self {
        match val {
            0 => Self::Success,
            10 => Self::CapabilityMismatch,
            11 => Self::InsufficientLockableMemory,
            12 => Self::ContextLengthExceeded,
            13 => Self::AffinityError,
            14 => Self::TensorBoundsError,
            15 => Self::RoutingUnavailable,
            20 => Self::ArenaNotFound,
            22 => Self::ProtocolError,
            23 => Self::QueueSaturated,
            _ => Self::InternalError,
        }
    }
}

// ============================================================================
// Memory Management Commands (k3-kv <-> k3-manager)
// ============================================================================

/// Type ID: 101 — Request to allocate a shared memory arena.
#[derive(ForyStruct, Debug, Clone, PartialEq, Eq)]
pub struct AllocateArenaRequest {
    /// Logical model identifier.
    pub model_id: String,
    /// Requested arena capacity in bytes.
    pub arena_bytes: u64,
    /// Maximum context tokens (`n_ctx`).
    pub n_ctx: u32,
    /// Maximum number of 16-token KV blocks.
    pub max_blocks: u32,
}

/// Type ID: 102 — Response to arena allocation request.
#[derive(ForyStruct, Debug, Clone, PartialEq, Eq)]
pub struct AllocateArenaResponse {
    /// Assigned unique arena identifier.
    pub arena_id: u64,
    /// Actual allocated arena capacity in bytes (page aligned).
    pub arena_bytes: u64,
    /// Status code.
    pub status: StatusCode,
    /// Diagnostic error message if failed.
    pub error_message: Option<String>,
}

/// Type ID: 103 — Request to obtain arena file descriptor via `SCM_RIGHTS`.
#[derive(ForyStruct, Debug, Clone, PartialEq, Eq)]
pub struct GetArenaFdRequest {
    /// Identifier of the arena whose FD is requested.
    pub arena_id: u64,
}

/// Type ID: 104 — Response to arena file descriptor request.
#[derive(ForyStruct, Debug, Clone, PartialEq, Eq)]
pub struct GetArenaFdResponse {
    /// Identifier of the arena.
    pub arena_id: u64,
    /// Status code.
    pub status: StatusCode,
    /// Diagnostic error message if failed.
    pub error_message: Option<String>,
}

/// Type ID: 105 — Request to deallocate an arena.
#[derive(ForyStruct, Debug, Clone, PartialEq, Eq)]
pub struct DeallocateArenaRequest {
    /// Identifier of the arena to deallocate.
    pub arena_id: u64,
}

/// Type ID: 106 — Response to arena deallocation request.
#[derive(ForyStruct, Debug, Clone, PartialEq, Eq)]
pub struct DeallocateArenaResponse {
    /// Identifier of the deallocated arena.
    pub arena_id: u64,
    /// Status code.
    pub status: StatusCode,
    /// Diagnostic error message if failed.
    pub error_message: Option<String>,
}

/// Type ID: 107 — Liveness ping request.
#[derive(ForyStruct, Debug, Clone, PartialEq, Eq)]
pub struct PingRequest {
    /// Nanosecond timestamp from sender.
    pub timestamp_ns: u64,
}

/// Type ID: 108 — Liveness ping response.
#[derive(ForyStruct, Debug, Clone, PartialEq, Eq)]
pub struct PingResponse {
    /// Echoed nanosecond timestamp.
    pub timestamp_ns: u64,
}

// ============================================================================
// Forward Execution Commands (k3-manager <-> k3-worker)
// ============================================================================

/// Type ID: 201 — Command informing worker to attach an arena passed via `SCM_RIGHTS`.
#[derive(ForyStruct, Debug, Clone, PartialEq, Eq)]
pub struct AttachArenaCommand {
    /// Target arena identifier.
    pub arena_id: u64,
    /// Arena length in bytes.
    pub arena_bytes: u64,
    /// Maximum context tokens (`n_ctx`).
    pub n_ctx: u32,
    /// Maximum number of 16-token KV blocks.
    pub max_blocks: u32,
}

/// Type ID: 202 — Dispatch a forward prefill or decode step to worker.
#[derive(ForyStruct, Debug, Clone, PartialEq, Eq)]
pub struct ForwardPassCommand {
    /// Unique request identifier.
    pub request_id: u64,
    /// Sequence identifier.
    pub sequence_id: u64,
    /// Starting token offset within the input token slot.
    pub token_offset: u32,
    /// Number of tokens to evaluate.
    pub token_count: u32,
    /// List of 16-token block indices allocated for this sequence.
    pub block_indices: Vec<u32>,
    /// `true` for prompt prefill, `false` for single-token autoregressive decode.
    pub is_prefill: bool,
}

/// Type ID: 203 — Notification of completed forward pass execution.
#[derive(ForyStruct, Debug, Clone, PartialEq)]
pub struct ForwardPassCompleted {
    /// Request identifier matching [`ForwardPassCommand`].
    pub request_id: u64,
    /// Sequence identifier.
    pub sequence_id: u64,
    /// Total committed tokens checkpointed in KV cache blocks.
    pub committed_tokens: u32,
    /// Measured prefill latency in milliseconds (0.0 for decode steps).
    pub prefill_latency_ms: f64,
    /// Measured decode latency in milliseconds.
    pub decode_latency_ms: f64,
    /// Sampled next token ID (or top logit index).
    pub next_token: u32,
    /// Execution status code.
    pub status: StatusCode,
    /// Diagnostic error message if forward pass failed.
    pub error_message: Option<String>,
}

/// Type ID: 204 — Worker heartbeat request.
#[derive(ForyStruct, Debug, Clone, PartialEq, Eq)]
pub struct WorkerHeartbeatRequest {
    /// Nanosecond timestamp from manager.
    pub timestamp_ns: u64,
}

/// Type ID: 205 — Worker heartbeat response.
#[derive(ForyStruct, Debug, Clone, PartialEq, Eq)]
pub struct WorkerHeartbeatResponse {
    /// Echoed nanosecond timestamp.
    pub timestamp_ns: u64,
    /// Whether worker is currently computing a forward pass.
    pub is_busy: bool,
    /// Total forward passes completed since worker spawn.
    pub completed_passes: u64,
}

/// Type ID: 206 — Command instructing worker to terminate.
#[derive(ForyStruct, Debug, Clone, PartialEq, Eq)]
pub struct ShutdownCommand {
    /// Whether to wait for active forward pass completion before exit.
    pub graceful: bool,
}

// ============================================================================
// Top-Level Unified Message Enum
// ============================================================================

/// Unified enumeration of all IPC message types across K3 domains.
#[derive(Debug, Clone, PartialEq)]
pub enum IpcMessage {
    AllocateArenaRequest(AllocateArenaRequest),
    AllocateArenaResponse(AllocateArenaResponse),
    GetArenaFdRequest(GetArenaFdRequest),
    GetArenaFdResponse(GetArenaFdResponse),
    DeallocateArenaRequest(DeallocateArenaRequest),
    DeallocateArenaResponse(DeallocateArenaResponse),
    PingRequest(PingRequest),
    PingResponse(PingResponse),
    AttachArenaCommand(AttachArenaCommand),
    ForwardPassCommand(ForwardPassCommand),
    ForwardPassCompleted(ForwardPassCompleted),
    WorkerHeartbeatRequest(WorkerHeartbeatRequest),
    WorkerHeartbeatResponse(WorkerHeartbeatResponse),
    ShutdownCommand(ShutdownCommand),
}

impl IpcMessage {
    /// Returns the discrete numeric type ID associated with this message variant.
    #[must_use]
    pub const fn type_id(&self) -> u16 {
        match self {
            Self::AllocateArenaRequest(_) => 101,
            Self::AllocateArenaResponse(_) => 102,
            Self::GetArenaFdRequest(_) => 103,
            Self::GetArenaFdResponse(_) => 104,
            Self::DeallocateArenaRequest(_) => 105,
            Self::DeallocateArenaResponse(_) => 106,
            Self::PingRequest(_) => 107,
            Self::PingResponse(_) => 108,
            Self::AttachArenaCommand(_) => 201,
            Self::ForwardPassCommand(_) => 202,
            Self::ForwardPassCompleted(_) => 203,
            Self::WorkerHeartbeatRequest(_) => 204,
            Self::WorkerHeartbeatResponse(_) => 205,
            Self::ShutdownCommand(_) => 206,
        }
    }

    /// Serializes the message into binary bytes using Apache Fory.
    ///
    /// # Errors
    /// Returns [`fory::Error`] if serialization fails.
    pub fn serialize(&self, fory: &Fory) -> Result<(u16, Vec<u8>), fory::Error> {
        let type_id = self.type_id();
        let bytes = match self {
            Self::AllocateArenaRequest(v) => fory.serialize(v)?,
            Self::AllocateArenaResponse(v) => fory.serialize(v)?,
            Self::GetArenaFdRequest(v) => fory.serialize(v)?,
            Self::GetArenaFdResponse(v) => fory.serialize(v)?,
            Self::DeallocateArenaRequest(v) => fory.serialize(v)?,
            Self::DeallocateArenaResponse(v) => fory.serialize(v)?,
            Self::PingRequest(v) => fory.serialize(v)?,
            Self::PingResponse(v) => fory.serialize(v)?,
            Self::AttachArenaCommand(v) => fory.serialize(v)?,
            Self::ForwardPassCommand(v) => fory.serialize(v)?,
            Self::ForwardPassCompleted(v) => fory.serialize(v)?,
            Self::WorkerHeartbeatRequest(v) => fory.serialize(v)?,
            Self::WorkerHeartbeatResponse(v) => fory.serialize(v)?,
            Self::ShutdownCommand(v) => fory.serialize(v)?,
        };
        Ok((type_id, bytes))
    }

    /// Deserializes an [`IpcMessage`] from payload bytes given its numeric type ID.
    ///
    /// # Errors
    /// Returns [`fory::Error`] if type ID is unknown or decoding fails.
    pub fn deserialize(type_id: u16, payload: &[u8], fory: &Fory) -> Result<Self, fory::Error> {
        match type_id {
            101 => Ok(Self::AllocateArenaRequest(fory.deserialize(payload)?)),
            102 => Ok(Self::AllocateArenaResponse(fory.deserialize(payload)?)),
            103 => Ok(Self::GetArenaFdRequest(fory.deserialize(payload)?)),
            104 => Ok(Self::GetArenaFdResponse(fory.deserialize(payload)?)),
            105 => Ok(Self::DeallocateArenaRequest(fory.deserialize(payload)?)),
            106 => Ok(Self::DeallocateArenaResponse(fory.deserialize(payload)?)),
            107 => Ok(Self::PingRequest(fory.deserialize(payload)?)),
            108 => Ok(Self::PingResponse(fory.deserialize(payload)?)),
            201 => Ok(Self::AttachArenaCommand(fory.deserialize(payload)?)),
            202 => Ok(Self::ForwardPassCommand(fory.deserialize(payload)?)),
            203 => Ok(Self::ForwardPassCompleted(fory.deserialize(payload)?)),
            204 => Ok(Self::WorkerHeartbeatRequest(fory.deserialize(payload)?)),
            205 => Ok(Self::WorkerHeartbeatResponse(fory.deserialize(payload)?)),
            206 => Ok(Self::ShutdownCommand(fory.deserialize(payload)?)),
            unknown => Err(fory::Error::type_error(format!(
                "Unknown IPC type ID: {unknown}"
            ))),
        }
    }
}

// ============================================================================
// Codec Factory & Global Singleton
// ============================================================================

/// Creates and configures a new Apache Fory codec with all K3 message types pre-registered.
///
/// # Errors
/// Returns [`fory::Error`] on registration failure.
pub fn create_fory_codec() -> Result<Fory, fory::Error> {
    let mut fory = Fory::builder().xlang(false).compatible(true).build();

    fory.register::<StatusCode>(100)?;
    fory.register::<AllocateArenaRequest>(101)?;
    fory.register::<AllocateArenaResponse>(102)?;
    fory.register::<GetArenaFdRequest>(103)?;
    fory.register::<GetArenaFdResponse>(104)?;
    fory.register::<DeallocateArenaRequest>(105)?;
    fory.register::<DeallocateArenaResponse>(106)?;
    fory.register::<PingRequest>(107)?;
    fory.register::<PingResponse>(108)?;

    fory.register::<AttachArenaCommand>(201)?;
    fory.register::<ForwardPassCommand>(202)?;
    fory.register::<ForwardPassCompleted>(203)?;
    fory.register::<WorkerHeartbeatRequest>(204)?;
    fory.register::<WorkerHeartbeatResponse>(205)?;
    fory.register::<ShutdownCommand>(206)?;

    Ok(fory)
}

/// Returns a reference to the global lazily-initialized Apache Fory codec singleton.
#[must_use]
pub fn global_fory() -> &'static Fory {
    static FORY: std::sync::LazyLock<Fory> = std::sync::LazyLock::new(|| {
        create_fory_codec().expect("Failed to initialize K3 Apache Fory codec")
    });
    &FORY
}

// ============================================================================
// UDS Framing & SCM_RIGHTS Transport Protocol
// ============================================================================

/// Magic identifier for K3 UDS stream frames (`0x4B334950` = ASCII 'K3IP').
pub const FRAME_MAGIC: u32 = 0x4B33_4950;

/// Length of fixed binary frame header in bytes.
pub const FRAME_HEADER_LEN: usize = 12;

/// Maximum allowable payload size (16 MiB defensive ceiling).
pub const MAX_PAYLOAD_LEN: u32 = 16 * 1024 * 1024;

/// Frame flag: file descriptor is attached via `SCM_RIGHTS`.
pub const FLAG_FD_ATTACHED: u16 = 0x0001;

/// Frame flag: message is a response to an earlier request.
pub const FLAG_RESPONSE: u16 = 0x0002;

/// Frame flag: payload contains an error.
pub const FLAG_ERROR: u16 = 0x0004;

/// 12-byte fixed-size binary framing header for Unix Domain Socket streams.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IpcFrameHeader {
    /// Magic identifier (`0x4B334950` / 'K3IP').
    pub magic: u32,
    /// Bitflags (`FLAG_FD_ATTACHED`, `FLAG_RESPONSE`, `FLAG_ERROR`).
    pub flags: u16,
    /// Registered Apache Fory numeric type ID.
    pub type_id: u16,
    /// Length of the following serialized payload in bytes.
    pub payload_len: u32,
}

impl IpcFrameHeader {
    /// Constructs a new [`IpcFrameHeader`].
    #[must_use]
    pub const fn new(type_id: u16, payload_len: u32, flags: u16) -> Self {
        Self {
            magic: FRAME_MAGIC,
            flags,
            type_id,
            payload_len,
        }
    }

    /// Encodes header into a fixed 12-byte array.
    pub fn encode(&self, out: &mut [u8; FRAME_HEADER_LEN]) {
        out[0..4].copy_from_slice(&self.magic.to_be_bytes());
        out[4..6].copy_from_slice(&self.flags.to_be_bytes());
        out[6..8].copy_from_slice(&self.type_id.to_be_bytes());
        out[8..12].copy_from_slice(&self.payload_len.to_be_bytes());
    }

    /// Decodes header from a fixed 12-byte array.
    ///
    /// # Errors
    /// Returns [`io::Error`] if magic signature is invalid or payload exceeds limit.
    pub fn decode(src: &[u8; FRAME_HEADER_LEN]) -> io::Result<Self> {
        let magic = u32::from_be_bytes([src[0], src[1], src[2], src[3]]);
        if magic != FRAME_MAGIC {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("Invalid IPC frame magic: 0x{magic:08X} (expected 0x{FRAME_MAGIC:08X})"),
            ));
        }
        let flags = u16::from_be_bytes([src[4], src[5]]);
        let type_id = u16::from_be_bytes([src[6], src[7]]);
        let payload_len = u32::from_be_bytes([src[8], src[9], src[10], src[11]]);
        if payload_len > MAX_PAYLOAD_LEN {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("Payload length {payload_len} exceeds maximum allowed {MAX_PAYLOAD_LEN}"),
            ));
        }
        Ok(Self {
            magic,
            flags,
            type_id,
            payload_len,
        })
    }
}

/// Sends an IPC frame with optional `SCM_RIGHTS` file descriptor over a Unix domain socket.
///
/// # Errors
/// Returns [`io::Error`] on socket send failure.
#[allow(clippy::cast_possible_truncation)]
pub fn send_frame(
    sock: RawFd,
    type_id: u16,
    payload: &[u8],
    fd_to_pass: Option<RawFd>,
) -> io::Result<()> {
    let mut flags = 0_u16;
    if fd_to_pass.is_some() {
        flags |= FLAG_FD_ATTACHED;
    }
    let payload_len_u32 = u32::try_from(payload.len()).map_err(|e| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("payload len exceeds u32: {e}"),
        )
    })?;

    let header = IpcFrameHeader::new(type_id, payload_len_u32, flags);
    let mut header_buf = [0_u8; FRAME_HEADER_LEN];
    header.encode(&mut header_buf);

    let mut iov = [
        libc::iovec {
            iov_base: header_buf.as_mut_ptr().cast(),
            iov_len: FRAME_HEADER_LEN,
        },
        libc::iovec {
            iov_base: payload.as_ptr() as *mut libc::c_void,
            iov_len: payload.len(),
        },
    ];

    let mut cmsg_buffer = [0_u8; 128];
    let (cmsg_ptr, cmsg_len) = if let Some(fd) = fd_to_pass {
        // SAFETY: Preparing ancillary control buffer for SCM_RIGHTS FD transfer.
        unsafe {
            let space =
                libc::CMSG_SPACE(u32::try_from(size_of::<libc::c_int>()).unwrap_or(4)) as usize;
            if space > cmsg_buffer.len() {
                return Err(io::Error::other("CMSG space exceeds buffer capacity"));
            }
            let mut dummy_hdr: libc::msghdr = std::mem::zeroed();
            dummy_hdr.msg_control = cmsg_buffer.as_mut_ptr().cast();
            dummy_hdr.msg_controllen = space as _;

            let cmsg = libc::CMSG_FIRSTHDR(&raw const dummy_hdr);
            if cmsg.is_null() {
                return Err(io::Error::other("CMSG_FIRSTHDR failed"));
            }
            (*cmsg).cmsg_level = libc::SOL_SOCKET;
            (*cmsg).cmsg_type = libc::SCM_RIGHTS;
            (*cmsg).cmsg_len =
                libc::CMSG_LEN(u32::try_from(size_of::<libc::c_int>()).unwrap_or(4)) as _;

            std::ptr::write_unaligned(libc::CMSG_DATA(cmsg).cast::<libc::c_int>(), fd);

            (cmsg_buffer.as_mut_ptr().cast(), space)
        }
    } else {
        (std::ptr::null_mut(), 0)
    };

    // SAFETY: Construct msghdr with iov and control buffers for sendmsg.
    let mut msg: libc::msghdr = unsafe { std::mem::zeroed() };
    msg.msg_iov = iov.as_mut_ptr();
    msg.msg_iovlen = 2;
    msg.msg_control = cmsg_ptr;
    msg.msg_controllen = cmsg_len as _;

    let total_to_send = FRAME_HEADER_LEN + payload.len();
    // SAFETY: sendmsg writes iov and ancillary data to the socket descriptor.
    let sent = unsafe { libc::sendmsg(sock, &raw const msg, 0) };
    if sent < 0 {
        return Err(io::Error::last_os_error());
    }
    let sent_usize = usize::try_from(sent).unwrap_or(0);
    if sent_usize < total_to_send {
        let mut written = sent_usize;
        while written < total_to_send {
            let payload_offset = written - FRAME_HEADER_LEN;
            // SAFETY: send writes remaining payload bytes to the socket.
            let n = unsafe {
                libc::send(
                    sock,
                    payload[payload_offset..].as_ptr().cast(),
                    total_to_send - written,
                    0,
                )
            };
            if n <= 0 {
                return Err(io::Error::last_os_error());
            }
            let n_usize = usize::try_from(n).unwrap_or(0);
            written += n_usize;
        }
    }
    Ok(())
}

/// Receives an IPC frame and any `SCM_RIGHTS` file descriptor over a Unix domain socket.
///
/// # Errors
/// Returns [`io::Error`] on socket error, unexpected EOF, or protocol violation.
#[allow(clippy::cast_possible_truncation)]
pub fn recv_frame(sock: RawFd) -> io::Result<(IpcFrameHeader, Vec<u8>, Option<OwnedFd>)> {
    let mut header_buf = [0_u8; FRAME_HEADER_LEN];
    let mut iov = libc::iovec {
        iov_base: header_buf.as_mut_ptr().cast(),
        iov_len: FRAME_HEADER_LEN,
    };

    let mut cmsg_buffer = [0_u8; 128];
    // SAFETY: Construct msghdr with iov and control buffer for recvmsg.
    let mut msg: libc::msghdr = unsafe { std::mem::zeroed() };
    msg.msg_iov = &raw mut iov;
    msg.msg_iovlen = 1;
    msg.msg_control = cmsg_buffer.as_mut_ptr().cast();
    msg.msg_controllen = cmsg_buffer.len() as _;

    // SAFETY: recvmsg reads message data and ancillary control bytes.
    let bytes_read = unsafe { libc::recvmsg(sock, &raw mut msg, 0) };
    if bytes_read <= 0 {
        if bytes_read == 0 {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "Socket closed by peer",
            ));
        }
        return Err(io::Error::last_os_error());
    }

    // Extract SCM_RIGHTS FD if present
    let mut received_fd: Option<OwnedFd> = None;
    // SAFETY: Parse msghdr control messages to extract SCM_RIGHTS file descriptor.
    unsafe {
        let mut cmsg = libc::CMSG_FIRSTHDR(&raw const msg);
        while !cmsg.is_null() {
            if (*cmsg).cmsg_level == libc::SOL_SOCKET && (*cmsg).cmsg_type == libc::SCM_RIGHTS {
                let raw_fd = std::ptr::read_unaligned(libc::CMSG_DATA(cmsg).cast::<libc::c_int>());
                if raw_fd >= 0 {
                    received_fd = Some(OwnedFd::from_raw_fd(raw_fd));
                    break;
                }
            }
            cmsg = libc::CMSG_NXTHDR(&raw const msg, cmsg);
        }
    }

    // Complete reading header if partial
    let mut header_read = usize::try_from(bytes_read).unwrap_or(0);
    while header_read < FRAME_HEADER_LEN {
        // SAFETY: recv reads remaining header bytes from stream.
        let n = unsafe {
            libc::recv(
                sock,
                header_buf[header_read..].as_mut_ptr().cast(),
                FRAME_HEADER_LEN - header_read,
                0,
            )
        };
        if n <= 0 {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "Truncated frame header",
            ));
        }
        let n_usize = usize::try_from(n).unwrap_or(0);
        header_read += n_usize;
    }

    let header = IpcFrameHeader::decode(&header_buf)?;

    if (header.flags & FLAG_FD_ATTACHED != 0) && received_fd.is_none() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "Frame specified FLAG_FD_ATTACHED but no SCM_RIGHTS FD was delivered",
        ));
    }

    let payload_len = usize::try_from(header.payload_len).unwrap_or(0);
    let mut payload = vec![0_u8; payload_len];
    if payload_len > 0 {
        let mut total_read = 0;
        while total_read < payload_len {
            // SAFETY: recv reads remaining payload bytes from stream into allocated buffer.
            let n = unsafe {
                libc::recv(
                    sock,
                    payload[total_read..].as_mut_ptr().cast(),
                    payload_len - total_read,
                    0,
                )
            };
            if n <= 0 {
                return Err(io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    "Truncated frame payload",
                ));
            }
            let n_usize = usize::try_from(n).unwrap_or(0);
            total_read += n_usize;
        }
    }

    Ok((header, payload, received_fd))
}

/// Helper to create a connected bidirectional Unix stream socket pair.
///
/// # Errors
/// Returns [`io::Error`] if `socketpair` fails.
pub fn create_uds_socket_pair() -> io::Result<(OwnedFd, OwnedFd)> {
    let mut fds = [0_i32; 2];
    // SAFETY: socketpair creates two connected AF_UNIX stream sockets.
    let ret = unsafe { libc::socketpair(libc::AF_UNIX, libc::SOCK_STREAM, 0, fds.as_mut_ptr()) };
    if ret != 0 {
        return Err(io::Error::last_os_error());
    }
    // SAFETY: fds are guaranteed valid by successful socketpair call.
    unsafe { Ok((OwnedFd::from_raw_fd(fds[0]), OwnedFd::from_raw_fd(fds[1]))) }
}

/// Sends an [`IpcMessage`] with optional attached file descriptor over a [`UnixStream`].
///
/// # Errors
/// Returns [`io::Error`] if serialization or sending fails.
pub fn send_msg_frame(
    stream: &UnixStream,
    msg: &IpcMessage,
    fd_to_pass: Option<RawFd>,
) -> io::Result<()> {
    let fory = global_fory();
    let (type_id, payload) = msg.serialize(fory).map_err(|e| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            format!("Fory serialization error: {e}"),
        )
    })?;
    send_frame(stream.as_raw_fd(), type_id, &payload, fd_to_pass)
}

/// Receives an [`IpcMessage`] with optional attached file descriptor from a [`UnixStream`].
///
/// # Errors
/// Returns [`io::Error`] if receiving or deserialization fails.
pub fn recv_msg_frame(stream: &UnixStream) -> io::Result<(IpcMessage, Option<OwnedFd>)> {
    let (header, payload, opt_fd) = recv_frame(stream.as_raw_fd())?;
    let fory = global_fory();
    let msg = IpcMessage::deserialize(header.type_id, &payload, fory).map_err(|e| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            format!("Fory deserialization error: {e}"),
        )
    })?;
    Ok((msg, opt_fd))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fory_round_trip_all_message_types() {
        let fory = global_fory();

        // 1. AllocateArenaRequest / Response
        let req1 = IpcMessage::AllocateArenaRequest(AllocateArenaRequest {
            model_id: "qwen2.5-7b".to_owned(),
            arena_bytes: 1024 * 1024,
            n_ctx: 128,
            max_blocks: 8,
        });
        let (t1, b1) = req1.serialize(fory).expect("serialize req1");
        assert_eq!(t1, 101);
        let dec1 = IpcMessage::deserialize(t1, &b1, fory).expect("deserialize req1");
        assert_eq!(dec1, req1);

        let resp1 = IpcMessage::AllocateArenaResponse(AllocateArenaResponse {
            arena_id: 42,
            arena_bytes: 1024 * 1024,
            status: StatusCode::Success,
            error_message: None,
        });
        let (t2, b2) = resp1.serialize(fory).expect("serialize resp1");
        assert_eq!(t2, 102);
        let dec2 = IpcMessage::deserialize(t2, &b2, fory).expect("deserialize resp1");
        assert_eq!(dec2, resp1);

        // 2. ForwardPassCommand / Completed
        let fwd_cmd = IpcMessage::ForwardPassCommand(ForwardPassCommand {
            request_id: 1001,
            sequence_id: 50,
            token_offset: 0,
            token_count: 16,
            block_indices: vec![0, 1, 2],
            is_prefill: true,
        });
        let (t3, b3) = fwd_cmd.serialize(fory).expect("serialize fwd_cmd");
        assert_eq!(t3, 202);
        let dec3 = IpcMessage::deserialize(t3, &b3, fory).expect("deserialize fwd_cmd");
        assert_eq!(dec3, fwd_cmd);

        let fwd_comp = IpcMessage::ForwardPassCompleted(ForwardPassCompleted {
            request_id: 1001,
            sequence_id: 50,
            committed_tokens: 16,
            prefill_latency_ms: 2.34,
            decode_latency_ms: 0.0,
            next_token: 384,
            status: StatusCode::Success,
            error_message: None,
        });
        let (t4, b4) = fwd_comp.serialize(fory).expect("serialize fwd_comp");
        assert_eq!(t4, 203);
        let dec4 = IpcMessage::deserialize(t4, &b4, fory).expect("deserialize fwd_comp");
        assert_eq!(dec4, fwd_comp);

        // 3. Ping / Heartbeat / Shutdown
        let ping = IpcMessage::PingRequest(PingRequest {
            timestamp_ns: 999_888,
        });
        let (t5, b5) = ping.serialize(fory).expect("serialize ping");
        assert_eq!(t5, 107);
        let dec5 = IpcMessage::deserialize(t5, &b5, fory).expect("deserialize ping");
        assert_eq!(dec5, ping);

        let shut = IpcMessage::ShutdownCommand(ShutdownCommand { graceful: true });
        let (t6, b6) = shut.serialize(fory).expect("serialize shut");
        assert_eq!(t6, 206);
        let dec6 = IpcMessage::deserialize(t6, &b6, fory).expect("deserialize shut");
        assert_eq!(dec6, shut);
    }

    #[test]
    fn test_uds_frame_header_encode_decode() {
        let hdr = IpcFrameHeader::new(101, 256, FLAG_FD_ATTACHED | FLAG_RESPONSE);
        let mut buf = [0_u8; FRAME_HEADER_LEN];
        hdr.encode(&mut buf);

        let decoded = IpcFrameHeader::decode(&buf).expect("decode header");
        assert_eq!(decoded.magic, FRAME_MAGIC);
        assert_eq!(decoded.type_id, 101);
        assert_eq!(decoded.payload_len, 256);
        assert_eq!(decoded.flags, FLAG_FD_ATTACHED | FLAG_RESPONSE);

        // Invalid magic rejected
        let mut bad_buf = buf;
        bad_buf[0..4].copy_from_slice(&0xDEAD_BEEF_u32.to_be_bytes());
        assert!(IpcFrameHeader::decode(&bad_buf).is_err());
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_uds_frame_socketpair_roundtrip_with_and_without_fd() {
        let (fd_a, fd_b) = create_uds_socket_pair().expect("socketpair");

        // 1. Without FD
        let payload = b"HELLO_IPC_PAYLOAD";
        send_frame(fd_a.as_raw_fd(), 107, payload, None).expect("send_frame without fd");
        let (hdr, recv_payload, opt_fd) = recv_frame(fd_b.as_raw_fd()).expect("recv_frame");
        assert_eq!(hdr.type_id, 107);
        assert_eq!(recv_payload, payload);
        assert!(opt_fd.is_none());

        // 2. With FD
        let (pipe_r, pipe_w) = create_uds_socket_pair().expect("pipe pair");
        send_frame(
            fd_a.as_raw_fd(),
            201,
            b"ATTACH_CMD",
            Some(pipe_r.as_raw_fd()),
        )
        .expect("send_frame with fd");
        let (hdr2, recv_payload2, opt_fd2) =
            recv_frame(fd_b.as_raw_fd()).expect("recv_frame with fd");
        assert_eq!(hdr2.type_id, 201);
        assert_eq!(recv_payload2, b"ATTACH_CMD");
        assert!(opt_fd2.is_some());
        let transferred_fd = opt_fd2.unwrap();
        assert!(transferred_fd.as_raw_fd() >= 0);

        drop(pipe_w);
    }

    #[test]
    fn test_status_code_exit_code_consistency() {
        assert_eq!(StatusCode::Success.as_exit_code(), 0);
        assert_eq!(StatusCode::CapabilityMismatch.as_exit_code(), 10);
        assert_eq!(StatusCode::InsufficientLockableMemory.as_exit_code(), 11);
        assert_eq!(StatusCode::ContextLengthExceeded.as_exit_code(), 12);
        assert_eq!(StatusCode::AffinityError.as_exit_code(), 13);
        assert_eq!(StatusCode::TensorBoundsError.as_exit_code(), 14);

        let cap_err = K3Error::CapabilityMismatch {
            architecture: "foo".to_owned(),
            reason: "bar".to_owned(),
        };
        assert_eq!(
            StatusCode::from_k3_error(&cap_err),
            StatusCode::CapabilityMismatch
        );

        let mem_err = K3Error::InsufficientLockableMemory {
            required_bytes: 100,
            weight_bytes: 50,
            kv_bytes: 50,
            available_bytes: 10,
            rlimit_memlock_bytes: 80,
        };
        assert_eq!(
            StatusCode::from_k3_error(&mem_err),
            StatusCode::InsufficientLockableMemory
        );
    }
}
