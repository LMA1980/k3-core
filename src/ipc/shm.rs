//! Cross-platform shared memory arena abstraction backed by Linux `memfd_create`
//! or macOS `shm_open` / temporary unlinked file, with `SCM_RIGHTS` FD transfer.

use core::mem::size_of;
use std::fmt;
use std::fs::OpenOptions;
use std::io;
use std::os::fd::{AsRawFd, FromRawFd, OwnedFd, RawFd};
use std::os::unix::net::UnixStream;
use std::path::PathBuf;
use std::ptr::NonNull;
use std::sync::atomic::{AtomicU64, Ordering};

use sendfd::{RecvWithFd, SendWithFd};
use zerocopy::FromBytes;

use super::layout::{ArenaHeader, KvBlockMeta};
use crate::error::K3Error;

/// Operator memlock limit ceiling: 28 GiB (Rule 5, ADR-037).
pub const DEFAULT_LIMIT_MEMLOCK_BYTES: u64 = 28 * 1024 * 1024 * 1024;

/// System page size fallback (4096 bytes).
pub const DEFAULT_PAGE_SIZE: usize = 4096;

static SHM_COUNTER: AtomicU64 = AtomicU64::new(1);

/// Structured error type for shared memory and IPC operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum IpcError {
    /// Low-level OS operation failure (e.g. `memfd_create`, `mmap`, `ftruncate`).
    IoError {
        /// The operation that failed.
        operation: String,
        /// Detailed description or OS error string.
        reason: String,
    },
    /// Memory locking failed or requested bytes exceed the memlock ceiling.
    InsufficientLockableMemory(K3Error),
    /// Bounds or alignment check failed.
    TensorBoundsError(K3Error),
    /// Protocol or stream framing violation.
    ProtocolError(String),
}

impl fmt::Display for IpcError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::IoError { operation, reason } => {
                write!(f, "[ipc_io_error] {operation} failed: {reason}")
            }
            Self::InsufficientLockableMemory(err) | Self::TensorBoundsError(err) => {
                write!(f, "{err}")
            }
            Self::ProtocolError(reason) => write!(f, "[ipc_protocol_error] {reason}"),
        }
    }
}

impl std::error::Error for IpcError {}

impl From<K3Error> for IpcError {
    fn from(err: K3Error) -> Self {
        match err {
            K3Error::InsufficientLockableMemory { .. } => Self::InsufficientLockableMemory(err),
            K3Error::TensorBoundsError { .. } => Self::TensorBoundsError(err),
            other => Self::ProtocolError(other.to_string()),
        }
    }
}

impl From<io::Error> for IpcError {
    fn from(err: io::Error) -> Self {
        Self::IoError {
            operation: "io".to_owned(),
            reason: err.to_string(),
        }
    }
}

/// Returns the system memory page size in bytes via `sysconf(_SC_PAGESIZE)`.
#[must_use]
pub fn system_page_size() -> usize {
    // SAFETY: sysconf(_SC_PAGESIZE) is a standard POSIX query with no side effects.
    let val = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
    if val > 0 {
        usize::try_from(val).unwrap_or(DEFAULT_PAGE_SIZE)
    } else {
        DEFAULT_PAGE_SIZE
    }
}

/// Aligns a byte length up to the nearest system page boundary.
#[must_use]
pub fn align_to_page(len: usize) -> usize {
    let page_size = system_page_size();
    let rem = len % page_size;
    if rem == 0 {
        len.max(page_size)
    } else {
        len.saturating_add(page_size - rem)
    }
}

/// Configuration options for creating a [`SharedMemoryArena`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ArenaOptions {
    /// Logical name of the arena.
    pub name: String,
    /// Requested length in bytes (will be rounded up to system page size).
    pub len: usize,
    /// Whether to immediately lock memory with `mlock` upon creation.
    pub lock_memory: bool,
    /// Bounded memlock ceiling in bytes (defaults to 28 GiB, AGENTS.md Rule 5).
    pub memlock_ceiling_bytes: u64,
}

impl ArenaOptions {
    /// Creates default options for an arena of the specified name and length.
    #[must_use]
    pub fn new(name: impl Into<String>, len: usize) -> Self {
        Self {
            name: name.into(),
            len,
            lock_memory: false,
            memlock_ceiling_bytes: DEFAULT_LIMIT_MEMLOCK_BYTES,
        }
    }

    /// Sets whether memory should be locked via `mlock` upon creation.
    #[must_use]
    pub const fn with_lock(mut self, lock: bool) -> Self {
        self.lock_memory = lock;
        self
    }

    /// Sets a custom memlock ceiling in bytes.
    #[must_use]
    pub const fn with_memlock_ceiling(mut self, ceiling: u64) -> Self {
        self.memlock_ceiling_bytes = ceiling;
        self
    }
}

/// Portable shared memory arena abstraction.
///
/// Backed by Linux `memfd_create` or macOS `shm_open` / unlinked temporary file.
/// Mapped with `MAP_SHARED` and provides safe bounds-checked byte slicing.
#[derive(Debug)]
pub struct SharedMemoryArena {
    fd: OwnedFd,
    ptr: NonNull<u8>,
    len: usize,
    name: String,
    read_only: bool,
    is_locked: bool,
    path: Option<PathBuf>,
}

// SAFETY: SharedMemoryArena owns its mapped address range and file descriptor.
// Concurrent immutable access (&self) across threads is thread-safe.
unsafe impl Send for SharedMemoryArena {}
// SAFETY: Mutable access requires &mut self, preventing data races in safe Rust.
unsafe impl Sync for SharedMemoryArena {}

impl SharedMemoryArena {
    /// Creates a new shared memory arena with the given name and length.
    ///
    /// # Errors
    /// Returns [`IpcError`] on failure.
    pub fn create(name: &str, len: usize) -> Result<Self, IpcError> {
        Self::create_with_options(ArenaOptions::new(name, len))
    }

    /// Creates a new shared memory arena with explicit options.
    ///
    /// # Errors
    /// Returns [`IpcError::InsufficientLockableMemory`] if `lock_memory` is true and length
    /// exceeds the configured memlock ceiling or `mlock` fails.
    /// Returns [`IpcError::IoError`] if OS allocation or mapping fails.
    #[allow(
        clippy::too_many_lines,
        clippy::cast_possible_wrap,
        clippy::cast_possible_truncation
    )]
    pub fn create_with_options(options: ArenaOptions) -> Result<Self, IpcError> {
        let aligned_len = align_to_page(options.len);

        // Pre-flight memlock ceiling check if lock requested
        let aligned_len_u64 = u64::try_from(aligned_len).unwrap_or(u64::MAX);
        if options.lock_memory && aligned_len_u64 > options.memlock_ceiling_bytes {
            return Err(IpcError::InsufficientLockableMemory(
                K3Error::InsufficientLockableMemory {
                    required_bytes: aligned_len_u64,
                    weight_bytes: 0,
                    kv_bytes: aligned_len_u64,
                    available_bytes: 0,
                    rlimit_memlock_bytes: options.memlock_ceiling_bytes,
                },
            ));
        }

        #[cfg(target_os = "linux")]
        {
            let cname =
                std::ffi::CString::new(options.name.as_str()).map_err(|e| IpcError::IoError {
                    operation: "memfd_create".to_owned(),
                    reason: format!("invalid cstring name: {e}"),
                })?;

            // SAFETY: libc::memfd_create creates an anonymous shared memory file descriptor with CLOEXEC and ALLOW_SEALING.
            let raw_fd = unsafe {
                libc::memfd_create(cname.as_ptr(), libc::MFD_CLOEXEC | libc::MFD_ALLOW_SEALING)
            };
            if raw_fd >= 0 {
                // SAFETY: ftruncate sets the backing file size for raw_fd.
                let trunc_ret = unsafe { libc::ftruncate(raw_fd, aligned_len as libc::off_t) };
                if trunc_ret == 0 {
                    // SAFETY: mmap maps the anonymous file descriptor into process memory with MAP_SHARED.
                    let ptr = unsafe {
                        libc::mmap(
                            std::ptr::null_mut(),
                            aligned_len,
                            libc::PROT_READ | libc::PROT_WRITE,
                            libc::MAP_SHARED,
                            raw_fd,
                            0,
                        )
                    };
                    if ptr != libc::MAP_FAILED {
                        // SAFETY: raw_fd is verified valid and ownership is transferred to OwnedFd.
                        let owned_fd = unsafe { OwnedFd::from_raw_fd(raw_fd) };
                        let non_null =
                            NonNull::new(ptr.cast::<u8>()).ok_or_else(|| IpcError::IoError {
                                operation: "mmap".to_owned(),
                                reason: "null pointer returned from mmap".to_owned(),
                            })?;

                        // Zero-initialize mapped arena
                        // SAFETY: ptr is valid for writes up to aligned_len bytes.
                        unsafe {
                            std::ptr::write_bytes(non_null.as_ptr(), 0, aligned_len);
                        }

                        let mut arena = Self {
                            fd: owned_fd,
                            ptr: non_null,
                            len: aligned_len,
                            name: options.name,
                            read_only: false,
                            is_locked: false,
                            path: None,
                        };

                        if options.lock_memory {
                            arena.lock_memory(Some(options.memlock_ceiling_bytes))?;
                        }

                        return Ok(arena);
                    }
                }
                // SAFETY: Clean up file descriptor on failure path.
                unsafe {
                    libc::close(raw_fd);
                }
            }
        }

        // Portable fallback: macOS Darwin and systems without memfd_create
        let seq = SHM_COUNTER.fetch_add(1, Ordering::Relaxed);
        let pid = std::process::id();

        #[cfg(not(target_os = "linux"))]
        {
            let shm_name = format!("/k3_{pid}_{seq}");
            if let Ok(cname) = std::ffi::CString::new(shm_name) {
                // SAFETY: shm_open opens or creates a POSIX shared memory object.
                let raw_fd = unsafe {
                    libc::shm_open(
                        cname.as_ptr(),
                        libc::O_RDWR | libc::O_CREAT | libc::O_EXCL,
                        0o600,
                    )
                };
                if raw_fd >= 0 {
                    // Unlink immediately so object is automatically cleaned up when all FDs close
                    // SAFETY: shm_unlink removes the named reference immediately while open FD survives.
                    unsafe {
                        libc::shm_unlink(cname.as_ptr());
                    }

                    // SAFETY: ftruncate sets the shm capacity.
                    let trunc_ret = unsafe { libc::ftruncate(raw_fd, aligned_len as libc::off_t) };
                    if trunc_ret == 0 {
                        // SAFETY: mmap maps the shm FD into the address space.
                        let ptr = unsafe {
                            libc::mmap(
                                std::ptr::null_mut(),
                                aligned_len,
                                libc::PROT_READ | libc::PROT_WRITE,
                                libc::MAP_SHARED,
                                raw_fd,
                                0,
                            )
                        };
                        if ptr != libc::MAP_FAILED {
                            // SAFETY: raw_fd is valid and transferred to OwnedFd.
                            let owned_fd = unsafe { OwnedFd::from_raw_fd(raw_fd) };
                            let non_null = NonNull::new(ptr.cast::<u8>()).ok_or_else(|| {
                                IpcError::IoError {
                                    operation: "shm_open_mmap".to_owned(),
                                    reason: "null pointer returned from mmap".to_owned(),
                                }
                            })?;

                            // SAFETY: ptr is valid for writes up to aligned_len.
                            unsafe {
                                std::ptr::write_bytes(non_null.as_ptr(), 0, aligned_len);
                            }

                            let mut arena = Self {
                                fd: owned_fd,
                                ptr: non_null,
                                len: aligned_len,
                                name: options.name,
                                read_only: false,
                                is_locked: false,
                                path: None,
                            };

                            if options.lock_memory {
                                arena.lock_memory(Some(options.memlock_ceiling_bytes))?;
                            }

                            return Ok(arena);
                        }
                    }
                    // SAFETY: Clean up raw_fd on failure.
                    unsafe {
                        libc::close(raw_fd);
                    }
                }
            }
        }

        // Unlinked temporary file fallback (ensures operation even under restrictive permissions)
        let temp_path =
            std::env::temp_dir().join(format!("k3_shm_{}_{pid}_{seq}.shm", options.name));
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(&temp_path)
            .map_err(|err| IpcError::IoError {
                operation: "tempfile_open".to_owned(),
                reason: format!("failed to open shm tempfile {}: {err}", temp_path.display()),
            })?;

        let aligned_len_u64 = u64::try_from(aligned_len).unwrap_or(u64::MAX);
        file.set_len(aligned_len_u64)
            .map_err(|err| IpcError::IoError {
                operation: "ftruncate".to_owned(),
                reason: format!("failed to truncate shm file: {err}"),
            })?;

        let raw_fd = file.as_raw_fd();
        // SAFETY: mmap maps the tempfile into process virtual memory with MAP_SHARED.
        let ptr = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                aligned_len,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_SHARED,
                raw_fd,
                0,
            )
        };

        if ptr == libc::MAP_FAILED {
            let last_err = io::Error::last_os_error();
            let _ = std::fs::remove_file(&temp_path);
            return Err(IpcError::IoError {
                operation: "mmap".to_owned(),
                reason: format!("mmap MAP_SHARED failed: {last_err}"),
            });
        }

        // Unlink file from disk immediately: remains accessible through open FD
        let _ = std::fs::remove_file(&temp_path);

        let non_null = NonNull::new(ptr.cast::<u8>()).ok_or_else(|| IpcError::IoError {
            operation: "tempfile_mmap".to_owned(),
            reason: "null pointer returned from mmap".to_owned(),
        })?;
        // SAFETY: ptr is valid for writes up to aligned_len bytes.
        unsafe {
            std::ptr::write_bytes(non_null.as_ptr(), 0, aligned_len);
        }

        let owned_fd = file.into();
        let mut arena = Self {
            fd: owned_fd,
            ptr: non_null,
            len: aligned_len,
            name: options.name,
            read_only: false,
            is_locked: false,
            path: None,
        };

        if options.lock_memory {
            arena.lock_memory(Some(options.memlock_ceiling_bytes))?;
        }

        Ok(arena)
    }

    /// Maps an existing shared memory arena from an open [`OwnedFd`].
    ///
    /// # Errors
    /// Returns [`IpcError::IoError`] if `mmap` fails.
    pub fn from_existing_fd(
        fd: OwnedFd,
        len: usize,
        name: &str,
        read_only: bool,
    ) -> Result<Self, IpcError> {
        let prot = if read_only {
            libc::PROT_READ
        } else {
            libc::PROT_READ | libc::PROT_WRITE
        };

        // SAFETY: mmap maps the existing file descriptor into process memory with MAP_SHARED.
        let ptr = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                len,
                prot,
                libc::MAP_SHARED,
                fd.as_raw_fd(),
                0,
            )
        };

        if ptr == libc::MAP_FAILED {
            let last_err = io::Error::last_os_error();
            return Err(IpcError::IoError {
                operation: "mmap_from_fd".to_owned(),
                reason: format!("mmap failed on existing fd: {last_err}"),
            });
        }

        let non_null = NonNull::new(ptr.cast::<u8>()).ok_or_else(|| IpcError::IoError {
            operation: "mmap_from_fd".to_owned(),
            reason: "null pointer returned from mmap".to_owned(),
        })?;
        Ok(Self {
            fd,
            ptr: non_null,
            len,
            name: name.to_owned(),
            read_only,
            is_locked: false,
            path: None,
        })
    }

    /// Returns the mapped length of the arena in bytes.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.len
    }

    /// Returns `true` if the arena has length 0.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Returns the logical name of the arena.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Returns the raw file descriptor of the arena.
    #[must_use]
    pub fn as_raw_fd(&self) -> RawFd {
        self.fd.as_raw_fd()
    }

    /// Duplicates the backing file descriptor using `libc::dup`.
    ///
    /// # Errors
    /// Returns [`IpcError::IoError`] if `libc::dup` fails.
    pub fn duplicate_fd(&self) -> Result<OwnedFd, IpcError> {
        // SAFETY: dup duplicates an open file descriptor.
        let raw = unsafe { libc::dup(self.fd.as_raw_fd()) };
        if raw < 0 {
            let err = io::Error::last_os_error();
            return Err(IpcError::IoError {
                operation: "duplicate_fd".to_owned(),
                reason: format!("failed to dup fd: {err}"),
            });
        }
        // SAFETY: raw is guaranteed valid by successful dup call.
        Ok(unsafe { OwnedFd::from_raw_fd(raw) })
    }

    /// Returns `true` if this mapping is read-only.
    #[must_use]
    pub const fn is_read_only(&self) -> bool {
        self.read_only
    }

    /// Returns `true` if the memory pages are locked in RAM via `mlock`.
    #[must_use]
    pub const fn is_locked(&self) -> bool {
        self.is_locked
    }

    /// Returns the non-null base pointer to the mapped virtual address range.
    #[must_use]
    pub const fn as_non_null(&self) -> NonNull<u8> {
        self.ptr
    }

    /// Returns a raw constant pointer to the mapped memory.
    #[must_use]
    pub const fn as_ptr(&self) -> *const u8 {
        self.ptr.as_ptr()
    }

    /// Returns a raw mutable pointer to the mapped memory.
    ///
    /// # Panics
    /// Panics if the arena was mapped as read-only.
    #[must_use]
    pub fn as_mut_ptr(&mut self) -> *mut u8 {
        assert!(
            !self.read_only,
            "cannot get mutable pointer for read-only arena"
        );
        self.ptr.as_ptr()
    }

    /// Returns a shared byte slice covering the entire arena.
    #[must_use]
    pub fn as_slice(&self) -> &[u8] {
        // SAFETY: self.ptr is valid for reads up to self.len while self is live.
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }

    /// Returns a mutable byte slice covering the entire arena.
    ///
    /// # Panics
    /// Panics if the arena was mapped as read-only.
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        assert!(
            !self.read_only,
            "cannot get mutable slice for read-only arena"
        );
        // SAFETY: self.ptr is valid for reads/writes up to self.len, &mut self guarantees exclusivity.
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) }
    }

    /// Returns a subslice verified against offset and length bounds.
    ///
    /// # Errors
    /// Returns [`K3Error::TensorBoundsError`] if `offset + len` overflows or exceeds arena length.
    pub fn slice(&self, offset: usize, len: usize) -> Result<&[u8], K3Error> {
        let end = offset
            .checked_add(len)
            .ok_or_else(|| K3Error::TensorBoundsError {
                tensor_name: self.name.clone(),
                offset,
                byte_len: len,
                buffer_len: self.len,
                reason: "offset + len calculation overflowed usize".to_owned(),
            })?;

        if end > self.len {
            return Err(K3Error::TensorBoundsError {
                tensor_name: self.name.clone(),
                offset,
                byte_len: len,
                buffer_len: self.len,
                reason: format!(
                    "requested range [{offset}..{end}] exceeds arena capacity {}",
                    self.len
                ),
            });
        }

        // SAFETY: Bounds checked above.
        unsafe {
            Ok(std::slice::from_raw_parts(
                self.ptr.as_ptr().add(offset),
                len,
            ))
        }
    }

    /// Returns a mutable subslice verified against offset and length bounds.
    ///
    /// # Errors
    /// Returns [`K3Error::TensorBoundsError`] if `offset + len` overflows or exceeds arena length.
    /// Returns [`K3Error::TensorBoundsError`] if the arena is mapped as read-only.
    pub fn slice_mut(&mut self, offset: usize, len: usize) -> Result<&mut [u8], K3Error> {
        if self.read_only {
            return Err(K3Error::TensorBoundsError {
                tensor_name: self.name.clone(),
                offset,
                byte_len: len,
                buffer_len: self.len,
                reason: "cannot obtain mutable slice from read-only arena".to_owned(),
            });
        }

        let end = offset
            .checked_add(len)
            .ok_or_else(|| K3Error::TensorBoundsError {
                tensor_name: self.name.clone(),
                offset,
                byte_len: len,
                buffer_len: self.len,
                reason: "offset + len calculation overflowed usize".to_owned(),
            })?;

        if end > self.len {
            return Err(K3Error::TensorBoundsError {
                tensor_name: self.name.clone(),
                offset,
                byte_len: len,
                buffer_len: self.len,
                reason: format!(
                    "requested range [{offset}..{end}] exceeds arena capacity {}",
                    self.len
                ),
            });
        }

        // SAFETY: Bounds checked above, &mut self guarantees exclusivity.
        unsafe {
            Ok(std::slice::from_raw_parts_mut(
                self.ptr.as_ptr().add(offset),
                len,
            ))
        }
    }

    /// Returns a reference to the [`ArenaHeader`] at offset 0 of the arena.
    ///
    /// # Errors
    /// Returns [`K3Error::TensorBoundsError`] if arena length is smaller than 128 bytes.
    pub fn header(&self) -> Result<&ArenaHeader, K3Error> {
        let slice = self.slice(0, size_of::<ArenaHeader>())?;
        ArenaHeader::ref_from_bytes(slice).map_err(|e| K3Error::TensorBoundsError {
            tensor_name: "arena_header".to_owned(),
            offset: 0,
            byte_len: size_of::<ArenaHeader>(),
            buffer_len: self.len,
            reason: format!("failed to read ArenaHeader: {e}"),
        })
    }

    /// Returns a mutable reference to the [`ArenaHeader`] at offset 0 of the arena.
    ///
    /// # Errors
    /// Returns [`K3Error::TensorBoundsError`] if arena length is smaller than 128 bytes or arena is read-only.
    pub fn header_mut(&mut self) -> Result<&mut ArenaHeader, K3Error> {
        let len = self.len;
        let name = self.name.clone();
        let slice = self.slice_mut(0, size_of::<ArenaHeader>())?;
        ArenaHeader::mut_from_bytes(slice).map_err(|e| K3Error::TensorBoundsError {
            tensor_name: format!("{name}_header"),
            offset: 0,
            byte_len: size_of::<ArenaHeader>(),
            buffer_len: len,
            reason: format!("failed to obtain mutable ArenaHeader: {e}"),
        })
    }

    /// Returns a reference to the [`KvBlockMeta`] record at `block_idx`.
    ///
    /// # Errors
    /// Returns [`K3Error::TensorBoundsError`] if offset exceeds arena bounds.
    pub fn block_meta(&self, block_idx: usize) -> Result<&KvBlockMeta, K3Error> {
        let meta_offset = usize::try_from(self.header()?.meta_table_offset).map_err(|e| {
            K3Error::TensorBoundsError {
                tensor_name: "kv_block_meta".to_owned(),
                offset: 0,
                byte_len: size_of::<KvBlockMeta>(),
                buffer_len: self.len,
                reason: format!("meta_table_offset exceeds usize: {e}"),
            }
        })?;
        let entry_size = size_of::<KvBlockMeta>();
        let offset = meta_offset
            .checked_add(block_idx.checked_mul(entry_size).ok_or_else(|| {
                K3Error::TensorBoundsError {
                    tensor_name: "kv_block_meta".to_owned(),
                    offset: 0,
                    byte_len: entry_size,
                    buffer_len: self.len,
                    reason: "block index offset overflowed usize".to_owned(),
                }
            })?)
            .ok_or_else(|| K3Error::TensorBoundsError {
                tensor_name: "kv_block_meta".to_owned(),
                offset: 0,
                byte_len: entry_size,
                buffer_len: self.len,
                reason: "block index offset overflowed usize".to_owned(),
            })?;

        let slice = self.slice(offset, entry_size)?;
        KvBlockMeta::ref_from_bytes(slice).map_err(|e| K3Error::TensorBoundsError {
            tensor_name: format!("kv_block_meta[{block_idx}]"),
            offset,
            byte_len: entry_size,
            buffer_len: self.len,
            reason: format!("failed to read KvBlockMeta: {e}"),
        })
    }

    /// Returns a mutable reference to the [`KvBlockMeta`] record at `block_idx`.
    ///
    /// # Errors
    /// Returns [`K3Error::TensorBoundsError`] if offset exceeds arena bounds or arena is read-only.
    pub fn block_meta_mut(&mut self, block_idx: usize) -> Result<&mut KvBlockMeta, K3Error> {
        let meta_offset = usize::try_from(self.header()?.meta_table_offset).map_err(|e| {
            K3Error::TensorBoundsError {
                tensor_name: "kv_block_meta".to_owned(),
                offset: 0,
                byte_len: size_of::<KvBlockMeta>(),
                buffer_len: self.len,
                reason: format!("meta_table_offset exceeds usize: {e}"),
            }
        })?;
        let entry_size = size_of::<KvBlockMeta>();
        let offset = meta_offset
            .checked_add(block_idx.checked_mul(entry_size).ok_or_else(|| {
                K3Error::TensorBoundsError {
                    tensor_name: "kv_block_meta".to_owned(),
                    offset: 0,
                    byte_len: entry_size,
                    buffer_len: self.len,
                    reason: "block index offset overflowed usize".to_owned(),
                }
            })?)
            .ok_or_else(|| K3Error::TensorBoundsError {
                tensor_name: "kv_block_meta".to_owned(),
                offset: 0,
                byte_len: entry_size,
                buffer_len: self.len,
                reason: "block index offset overflowed usize".to_owned(),
            })?;

        let len = self.len;
        let slice = self.slice_mut(offset, entry_size)?;
        KvBlockMeta::mut_from_bytes(slice).map_err(|e| K3Error::TensorBoundsError {
            tensor_name: format!("kv_block_meta[{block_idx}]"),
            offset,
            byte_len: entry_size,
            buffer_len: len,
            reason: format!("failed to obtain mutable KvBlockMeta: {e}"),
        })
    }

    /// Locks the arena pages into physical RAM via `libc::mlock`.
    ///
    /// # Errors
    /// Returns [`K3Error::InsufficientLockableMemory`] if `len` exceeds the ceiling or `mlock` fails.
    pub fn lock_memory(&mut self, ceiling_bytes: Option<u64>) -> Result<(), K3Error> {
        if self.is_locked {
            return Ok(());
        }

        let ceiling = ceiling_bytes.unwrap_or(DEFAULT_LIMIT_MEMLOCK_BYTES);
        let req_bytes = u64::try_from(self.len).unwrap_or(u64::MAX);
        if req_bytes > ceiling {
            return Err(K3Error::InsufficientLockableMemory {
                required_bytes: req_bytes,
                weight_bytes: 0,
                kv_bytes: req_bytes,
                available_bytes: 0,
                rlimit_memlock_bytes: ceiling,
            });
        }

        // SAFETY: self.ptr points to self.len valid mapped bytes.
        let ret = unsafe { libc::mlock(self.ptr.as_ptr().cast(), self.len) };
        if ret != 0 {
            return Err(K3Error::InsufficientLockableMemory {
                required_bytes: req_bytes,
                weight_bytes: 0,
                kv_bytes: req_bytes,
                available_bytes: 0,
                rlimit_memlock_bytes: ceiling,
            });
        }

        self.is_locked = true;
        Ok(())
    }

    /// Unlocks the arena pages via `libc::munlock`.
    ///
    /// # Errors
    /// Returns [`IpcError::IoError`] if `munlock` fails.
    pub fn unlock_memory(&mut self) -> Result<(), IpcError> {
        if !self.is_locked {
            return Ok(());
        }

        // SAFETY: self.ptr points to self.len valid mapped bytes previously locked.
        let ret = unsafe { libc::munlock(self.ptr.as_ptr().cast(), self.len) };
        if ret != 0 {
            let last_err = io::Error::last_os_error();
            return Err(IpcError::IoError {
                operation: "munlock".to_owned(),
                reason: format!("munlock failed: {last_err}"),
            });
        }

        self.is_locked = false;
        Ok(())
    }
}

impl Drop for SharedMemoryArena {
    fn drop(&mut self) {
        if self.is_locked {
            // SAFETY: self.ptr points to self.len valid bytes being unlocked on drop.
            unsafe {
                libc::munlock(self.ptr.as_ptr().cast(), self.len);
            }
        }
        // SAFETY: self.ptr points to self.len valid mapped bytes being unmapped on drop.
        unsafe {
            libc::munmap(self.ptr.as_ptr().cast(), self.len);
        }
        if let Some(path) = &self.path {
            let _ = std::fs::remove_file(path);
        }
    }
}

/// Extension trait for passing [`SharedMemoryArena`] file descriptors across Unix Domain Sockets (`SCM_RIGHTS`).
pub trait ArenaSocketExt {
    /// Sends a payload along with this arena's file descriptor over a `UnixStream`.
    ///
    /// # Errors
    /// Returns [`IpcError`] on failure.
    fn send_arena_fd(&self, stream: &UnixStream, payload: &[u8]) -> Result<usize, IpcError>;
}

impl ArenaSocketExt for SharedMemoryArena {
    fn send_arena_fd(&self, stream: &UnixStream, payload: &[u8]) -> Result<usize, IpcError> {
        stream
            .send_with_fd(payload, &[self.as_raw_fd()])
            .map_err(|err| IpcError::IoError {
                operation: "send_arena_fd".to_owned(),
                reason: format!("failed to send arena fd over UDS: {err}"),
            })
    }
}

/// Receives a payload and an attached arena file descriptor from a `UnixStream`.
///
/// # Errors
/// Returns [`IpcError`] on failure.
pub fn recv_arena_fd(
    stream: &UnixStream,
    buf: &mut [u8],
) -> Result<(usize, Option<OwnedFd>), IpcError> {
    let mut fds = [0 as RawFd; 1];
    let (bytes_read, fds_read) =
        stream
            .recv_with_fd(buf, &mut fds)
            .map_err(|err| IpcError::IoError {
                operation: "recv_arena_fd".to_owned(),
                reason: format!("failed to receive fd over UDS: {err}"),
            })?;

    let received_fd = if fds_read > 0 && fds[0] >= 0 {
        // SAFETY: sendfd verified fds_read > 0 and fd is a valid descriptor transferred via SCM_RIGHTS.
        Some(unsafe { OwnedFd::from_raw_fd(fds[0]) })
    } else {
        None
    };

    Ok((bytes_read, received_fd))
}

/// Receives an attached file descriptor over UDS and maps it as a [`SharedMemoryArena`].
///
/// # Errors
/// Returns [`IpcError`] if no FD was attached or mapping fails.
pub fn recv_and_map_arena(
    stream: &UnixStream,
    buf: &mut [u8],
    expected_len: usize,
    name: &str,
    read_only: bool,
) -> Result<(usize, SharedMemoryArena), IpcError> {
    let (bytes_read, opt_fd) = recv_arena_fd(stream, buf)?;
    let fd = opt_fd.ok_or_else(|| IpcError::IoError {
        operation: "recv_and_map_arena".to_owned(),
        reason: "no file descriptor was attached to the received UDS message".to_owned(),
    })?;
    let arena = SharedMemoryArena::from_existing_fd(fd, expected_len, name, read_only)?;
    Ok((bytes_read, arena))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_arena_creation_and_bounds() {
        let mut arena = SharedMemoryArena::create("test_bounds", 4096).expect("arena creation");
        assert!(arena.len() >= 4096);
        assert_eq!(arena.name(), "test_bounds");
        assert!(!arena.is_read_only());
        assert!(!arena.is_locked());
        assert!(arena.as_raw_fd() >= 0);

        // Verify initially zeroed
        assert!(arena.as_slice().iter().all(|&b| b == 0));

        // Valid subslice
        let slice = arena.slice(0, 128).expect("valid subslice");
        assert_eq!(slice.len(), 128);

        // Out-of-bounds subslice
        let oob = arena.slice(arena.len() - 10, 20);
        assert!(matches!(oob, Err(K3Error::TensorBoundsError { .. })));

        // Overflow subslice
        let overflow = arena.slice(usize::MAX - 5, 10);
        assert!(matches!(overflow, Err(K3Error::TensorBoundsError { .. })));

        // Mutable slice modification
        {
            let mut_slice = arena.slice_mut(64, 4).expect("valid mutable subslice");
            mut_slice.copy_from_slice(&[0xDE, 0xAD, 0xBE, 0xEF]);
        }
        assert_eq!(
            arena.slice(64, 4).expect("read subslice"),
            &[0xDE, 0xAD, 0xBE, 0xEF]
        );
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_arena_from_existing_fd_and_shared_mutation() {
        let mut arena1 = SharedMemoryArena::create("arena1", 4096).expect("arena1");
        {
            let slice = arena1.slice_mut(0, 8).expect("slice_mut");
            slice.copy_from_slice(b"SHARED42");
        }

        let dup_fd = arena1.duplicate_fd().expect("duplicate_fd");
        let arena2 = SharedMemoryArena::from_existing_fd(dup_fd, arena1.len(), "arena2", true)
            .expect("from_existing_fd");

        assert_eq!(arena2.len(), arena1.len());
        assert!(arena2.is_read_only());
        assert_eq!(
            arena2.slice(0, 8).expect("read from mapped fd"),
            b"SHARED42"
        );
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_arena_typed_header_and_meta_access() {
        let mut arena = SharedMemoryArena::create("typed_arena", 8192).expect("arena");
        {
            let hdr = arena.header_mut().expect("header_mut");
            *hdr = ArenaHeader::default();
            hdr.n_ctx = 128;
            hdr.block_count = 8;
        }

        assert_eq!(arena.header().expect("header").n_ctx, 128);
        assert_eq!(arena.header().expect("header").block_count, 8);

        {
            let meta = arena.block_meta_mut(0).expect("block_meta_mut");
            *meta = KvBlockMeta::default();
            meta.set_tag(crate::ipc::layout::KvBlockTag::Persist);
            meta.set_committed_tokens(12);
        }

        let meta = arena.block_meta(0).expect("block_meta");
        assert_eq!(meta.tag, crate::ipc::layout::TAG_PERSIST);
        assert_eq!(meta.committed_tokens(), 12);
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_arena_uds_fd_transfer_roundtrip() {
        let (s1, s2) = UnixStream::pair().expect("unix stream pair");
        let mut arena = SharedMemoryArena::create("uds_arena", 4096).expect("create arena");
        arena
            .slice_mut(0, 4)
            .expect("slice_mut")
            .copy_from_slice(b"TEST");

        let handle = std::thread::spawn(move || {
            let mut buf = [0_u8; 32];
            let (read_bytes, received_arena) =
                recv_and_map_arena(&s2, &mut buf, 4096, "uds_child", false).expect("recv arena");
            assert_eq!(&buf[..read_bytes], b"ATTACH_CMD");
            assert_eq!(received_arena.slice(0, 4).expect("read"), b"TEST");
        });

        let sent = arena
            .send_arena_fd(&s1, b"ATTACH_CMD")
            .expect("send_arena_fd");
        assert_eq!(sent, 10);
        handle.join().expect("thread join");
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_arena_memlock_ceiling_enforcement() {
        // Exceeding configured ceiling returns InsufficientLockableMemory
        let options = ArenaOptions::new("ceiling_test", 4096)
            .with_lock(true)
            .with_memlock_ceiling(2048);

        let err = SharedMemoryArena::create_with_options(options);
        assert!(matches!(
            err,
            Err(IpcError::InsufficientLockableMemory(
                K3Error::InsufficientLockableMemory { .. }
            ))
        ));
    }
}
