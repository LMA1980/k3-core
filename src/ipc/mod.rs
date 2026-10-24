//! IPC primitives: shared memory arena abstraction, zerocopy layouts,
//! Apache Arrow zero-copy tensor views, and Apache Fory UDS protocol transport.

pub mod arrow;
pub mod layout;
pub mod protocol;
pub mod shm;

pub use arrow::{
    create_arrow_buffer_from_shm, slice_arrow_buffer, wrap_floats_as_float32_array,
    wrap_tokens_as_uint32_array, ShmArrowAllocation, ShmTensorViews,
};
pub use layout::{
    ArenaHeader, KvBlockMeta, KvBlockTag, OutputSlabHeader, TokenInputSlotHeader, ARENA_MAGIC,
    ARENA_MAGIC_BYTES, ARENA_VERSION, BLOCK_SIZE_TOKENS, CACHELINE_ALIGNMENT, SLAB_STATUS_IDLE,
    SLAB_STATUS_READ, SLAB_STATUS_WRITTEN, TAG_FREE, TAG_PERSIST, TAG_PURGE, TAG_VOLATILE,
};
pub use protocol::{
    create_fory_codec, create_uds_socket_pair, global_fory, recv_frame, recv_msg_frame,
    send_frame, send_msg_frame, AllocateArenaRequest, AllocateArenaResponse, AttachArenaCommand,
    DeallocateArenaRequest, DeallocateArenaResponse, ForwardPassCommand, ForwardPassCompleted,
    GetArenaFdRequest, GetArenaFdResponse, IpcFrameHeader, IpcMessage, PingRequest, PingResponse,
    ShutdownCommand, StatusCode, WorkerHeartbeatRequest, WorkerHeartbeatResponse, FLAG_ERROR,
    FLAG_FD_ATTACHED, FLAG_RESPONSE, FRAME_HEADER_LEN, FRAME_MAGIC, MAX_PAYLOAD_LEN,
};
pub use shm::{
    align_to_page, recv_and_map_arena, recv_arena_fd, system_page_size, ArenaOptions,
    ArenaSocketExt, IpcError, SharedMemoryArena, DEFAULT_LIMIT_MEMLOCK_BYTES, DEFAULT_PAGE_SIZE,
};
