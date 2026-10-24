//! Empirical Challenger Test Suite for Milestone 2: Apache Fory IPC & UDS Framing.
//!
//! Adversarially challenges:
//! 1. Round-trip serialization/deserialization for all 14 message schemas under `ipc::protocol`.
//! 2. All 9 `StatusCode` variants across Fory, integer conversions, and exit code mappings.
//! 3. UDS frame transmission over socketpair with and without `SCM_RIGHTS` FD attachment.
//! 4. Verification that corrupted magic (`bad_magic != 0x4B334950`) is rejected safely with `IpcError`.
//! 5. Verification that oversized payloads (> 16 MiB / `MAX_PAYLOAD_LEN`) are rejected safely with `IpcError`.
//! 6. Streaming pipeline stress: bursts of sequential frames with interleaved FD passing.
//! 7. Adversarial edge cases: missing FD with `FLAG_FD_ATTACHED`, unknown type IDs, truncated headers/payloads, and corrupted payload bytes.

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_possible_wrap,
    clippy::similar_names,
    clippy::too_many_lines,
    clippy::items_after_statements,
    clippy::undocumented_unsafe_blocks
)]

use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::os::fd::AsRawFd;
use std::os::unix::net::UnixStream;
use std::sync::atomic::{AtomicU64, Ordering};

use k3_core::ipc::protocol::{
    create_fory_codec, create_uds_socket_pair, global_fory, recv_frame, recv_msg_frame,
    send_frame, send_msg_frame, AllocateArenaRequest, AllocateArenaResponse, AttachArenaCommand,
    DeallocateArenaRequest, DeallocateArenaResponse, ForwardPassCommand, ForwardPassCompleted,
    GetArenaFdRequest, GetArenaFdResponse, IpcFrameHeader, IpcMessage, PingRequest, PingResponse,
    ShutdownCommand, StatusCode, WorkerHeartbeatRequest, WorkerHeartbeatResponse,
    FLAG_FD_ATTACHED, FRAME_HEADER_LEN, FRAME_MAGIC, MAX_PAYLOAD_LEN,
};
use k3_core::ipc::IpcError;

static TEMP_COUNTER: AtomicU64 = AtomicU64::new(1);

fn create_test_temp_file() -> (std::path::PathBuf, File) {
    let id = TEMP_COUNTER.fetch_add(1, Ordering::Relaxed);
    let path = std::env::temp_dir().join(format!("k3_test_ipc_fd_{}_{}.tmp", std::process::id(), id));
    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(true)
        .open(&path)
        .expect("create test temp file");
    (path, file)
}

// ============================================================================
// 1. All 14 Fory Message Schemas Round-Trip Verification
// ============================================================================

#[test]
fn test_all_14_fory_message_schemas_round_trip() {
    let fory = global_fory();

    // Verify create_fory_codec initializes independently
    let independent_codec = create_fory_codec().expect("create independent codec");

    // 1. AllocateArenaRequest (Type ID: 101)
    let schema_1 = IpcMessage::AllocateArenaRequest(AllocateArenaRequest {
        model_id: "qwen2.5-7b-instruct-q4_k_m".to_owned(),
        arena_bytes: 4 * 1024 * 1024 * 1024,
        n_ctx: 4096,
        max_blocks: 256,
    });
    assert_eq!(schema_1.type_id(), 101);
    let (t1, bytes1) = schema_1.serialize(fory).expect("serialize schema 1");
    assert_eq!(t1, 101);
    let dec1 = IpcMessage::deserialize(t1, &bytes1, fory).expect("deserialize schema 1");
    assert_eq!(dec1, schema_1);
    let dec1_ind = IpcMessage::deserialize(t1, &bytes1, &independent_codec).expect("deserialize schema 1 ind");
    assert_eq!(dec1_ind, schema_1);

    // 2. AllocateArenaResponse (Type ID: 102) - Success
    let schema_2a = IpcMessage::AllocateArenaResponse(AllocateArenaResponse {
        arena_id: 0xDEAD_CAFE_BEEF,
        arena_bytes: 4 * 1024 * 1024 * 1024,
        status: StatusCode::Success,
        error_message: None,
    });
    assert_eq!(schema_2a.type_id(), 102);
    let (t2a, bytes2a) = schema_2a.serialize(fory).expect("serialize schema 2a");
    assert_eq!(t2a, 102);
    let dec2a = IpcMessage::deserialize(t2a, &bytes2a, fory).expect("deserialize schema 2a");
    assert_eq!(dec2a, schema_2a);

    // 2. AllocateArenaResponse - Failure with error message
    let schema_2b = IpcMessage::AllocateArenaResponse(AllocateArenaResponse {
        arena_id: 0,
        arena_bytes: 0,
        status: StatusCode::InsufficientLockableMemory,
        error_message: Some("Requested 32 GiB exceeds 28 GiB ceiling".to_owned()),
    });
    let (t2b, bytes2b) = schema_2b.serialize(fory).expect("serialize schema 2b");
    let dec2b = IpcMessage::deserialize(t2b, &bytes2b, fory).expect("deserialize schema 2b");
    assert_eq!(dec2b, schema_2b);

    // 3. GetArenaFdRequest (Type ID: 103)
    let schema_3 = IpcMessage::GetArenaFdRequest(GetArenaFdRequest {
        arena_id: 998_877_665,
    });
    assert_eq!(schema_3.type_id(), 103);
    let (t3, bytes3) = schema_3.serialize(fory).expect("serialize schema 3");
    assert_eq!(t3, 103);
    let dec3 = IpcMessage::deserialize(t3, &bytes3, fory).expect("deserialize schema 3");
    assert_eq!(dec3, schema_3);

    // 4. GetArenaFdResponse (Type ID: 104)
    let schema_4 = IpcMessage::GetArenaFdResponse(GetArenaFdResponse {
        arena_id: 998_877_665,
        status: StatusCode::Success,
        error_message: None,
    });
    assert_eq!(schema_4.type_id(), 104);
    let (t4, bytes4) = schema_4.serialize(fory).expect("serialize schema 4");
    assert_eq!(t4, 104);
    let dec4 = IpcMessage::deserialize(t4, &bytes4, fory).expect("deserialize schema 4");
    assert_eq!(dec4, schema_4);

    // 5. DeallocateArenaRequest (Type ID: 105)
    let schema_5 = IpcMessage::DeallocateArenaRequest(DeallocateArenaRequest {
        arena_id: 42,
    });
    assert_eq!(schema_5.type_id(), 105);
    let (t5, bytes5) = schema_5.serialize(fory).expect("serialize schema 5");
    assert_eq!(t5, 105);
    let dec5 = IpcMessage::deserialize(t5, &bytes5, fory).expect("deserialize schema 5");
    assert_eq!(dec5, schema_5);

    // 6. DeallocateArenaResponse (Type ID: 106)
    let schema_6 = IpcMessage::DeallocateArenaResponse(DeallocateArenaResponse {
        arena_id: 42,
        status: StatusCode::Success,
        error_message: None,
    });
    assert_eq!(schema_6.type_id(), 106);
    let (t6, bytes6) = schema_6.serialize(fory).expect("serialize schema 6");
    assert_eq!(t6, 106);
    let dec6 = IpcMessage::deserialize(t6, &bytes6, fory).expect("deserialize schema 6");
    assert_eq!(dec6, schema_6);

    // 7. PingRequest (Type ID: 107)
    let schema_7 = IpcMessage::PingRequest(PingRequest {
        timestamp_ns: 1_700_000_000_123_456_789,
    });
    assert_eq!(schema_7.type_id(), 107);
    let (t7, bytes7) = schema_7.serialize(fory).expect("serialize schema 7");
    assert_eq!(t7, 107);
    let dec7 = IpcMessage::deserialize(t7, &bytes7, fory).expect("deserialize schema 7");
    assert_eq!(dec7, schema_7);

    // 8. PingResponse (Type ID: 108)
    let schema_8 = IpcMessage::PingResponse(PingResponse {
        timestamp_ns: 1_700_000_000_123_456_789,
    });
    assert_eq!(schema_8.type_id(), 108);
    let (t8, bytes8) = schema_8.serialize(fory).expect("serialize schema 8");
    assert_eq!(t8, 108);
    let dec8 = IpcMessage::deserialize(t8, &bytes8, fory).expect("deserialize schema 8");
    assert_eq!(dec8, schema_8);

    // 9. AttachArenaCommand (Type ID: 201)
    let schema_9 = IpcMessage::AttachArenaCommand(AttachArenaCommand {
        arena_id: 100_001,
        arena_bytes: 64 * 1024 * 1024,
        n_ctx: 512,
        max_blocks: 32,
    });
    assert_eq!(schema_9.type_id(), 201);
    let (t9, bytes9) = schema_9.serialize(fory).expect("serialize schema 9");
    assert_eq!(t9, 201);
    let dec9 = IpcMessage::deserialize(t9, &bytes9, fory).expect("deserialize schema 9");
    assert_eq!(dec9, schema_9);

    // 10. ForwardPassCommand (Type ID: 202)
    let schema_10 = IpcMessage::ForwardPassCommand(ForwardPassCommand {
        request_id: 88_990_011,
        sequence_id: 77_665_544,
        token_offset: 256,
        token_count: 32,
        block_indices: vec![0, 1, 2, 7, 15, 31, 63, 127],
        is_prefill: true,
    });
    assert_eq!(schema_10.type_id(), 202);
    let (t10, bytes10) = schema_10.serialize(fory).expect("serialize schema 10");
    assert_eq!(t10, 202);
    let dec10 = IpcMessage::deserialize(t10, &bytes10, fory).expect("deserialize schema 10");
    assert_eq!(dec10, schema_10);

    // 11. ForwardPassCompleted (Type ID: 203)
    let schema_11 = IpcMessage::ForwardPassCompleted(ForwardPassCompleted {
        request_id: 88_990_011,
        sequence_id: 77_665_544,
        committed_tokens: 288,
        prefill_latency_ms: 14.85,
        decode_latency_ms: 0.72,
        next_token: 151_643,
        status: StatusCode::Success,
        error_message: None,
    });
    assert_eq!(schema_11.type_id(), 203);
    let (t11, bytes11) = schema_11.serialize(fory).expect("serialize schema 11");
    assert_eq!(t11, 203);
    let dec11 = IpcMessage::deserialize(t11, &bytes11, fory).expect("deserialize schema 11");
    assert_eq!(dec11, schema_11);

    // 12. WorkerHeartbeatRequest (Type ID: 204)
    let schema_12 = IpcMessage::WorkerHeartbeatRequest(WorkerHeartbeatRequest {
        timestamp_ns: 50_000_000_000,
    });
    assert_eq!(schema_12.type_id(), 204);
    let (t12, bytes12) = schema_12.serialize(fory).expect("serialize schema 12");
    assert_eq!(t12, 204);
    let dec12 = IpcMessage::deserialize(t12, &bytes12, fory).expect("deserialize schema 12");
    assert_eq!(dec12, schema_12);

    // 13. WorkerHeartbeatResponse (Type ID: 205)
    let schema_13 = IpcMessage::WorkerHeartbeatResponse(WorkerHeartbeatResponse {
        timestamp_ns: 50_000_000_000,
        is_busy: true,
        completed_passes: 1_234_567,
    });
    assert_eq!(schema_13.type_id(), 205);
    let (t13, bytes13) = schema_13.serialize(fory).expect("serialize schema 13");
    assert_eq!(t13, 205);
    let dec13 = IpcMessage::deserialize(t13, &bytes13, fory).expect("deserialize schema 13");
    assert_eq!(dec13, schema_13);

    // 14. ShutdownCommand (Type ID: 206)
    let schema_14 = IpcMessage::ShutdownCommand(ShutdownCommand {
        graceful: true,
    });
    assert_eq!(schema_14.type_id(), 206);
    let (t14, bytes14) = schema_14.serialize(fory).expect("serialize schema 14");
    assert_eq!(t14, 206);
    let dec14 = IpcMessage::deserialize(t14, &bytes14, fory).expect("deserialize schema 14");
    assert_eq!(dec14, schema_14);
}

// ============================================================================
// 2. StatusCode Round-Trip & ADR Invariant Verification
// ============================================================================

#[test]
fn test_all_status_codes_round_trip_and_invariants() {
    let fory = global_fory();

    let codes = [
        (StatusCode::Success, 0),
        (StatusCode::CapabilityMismatch, 10),
        (StatusCode::InsufficientLockableMemory, 11),
        (StatusCode::ContextLengthExceeded, 12),
        (StatusCode::AffinityError, 13),
        (StatusCode::TensorBoundsError, 14),
        (StatusCode::ArenaNotFound, 20),
        (StatusCode::InternalError, 21),
        (StatusCode::ProtocolError, 22),
    ];

    for (code, expected_u32) in codes {
        assert_eq!(code.as_exit_code(), expected_u32 as u8);
        assert_eq!(StatusCode::from_u32(expected_u32), code);
        if expected_u32 == 0 {
            assert!(code.is_success());
        } else {
            assert!(!code.is_success());
        }

        // Direct Fory serialization of enum
        let bytes = fory.serialize(&code).expect("serialize StatusCode");
        let decoded: StatusCode = fory.deserialize(&bytes).expect("deserialize StatusCode");
        assert_eq!(decoded, code);
    }

    // Unrecognized values default to InternalError
    assert_eq!(StatusCode::from_u32(999), StatusCode::InternalError);
    assert_eq!(StatusCode::from_u32(1), StatusCode::InternalError);
}

// ============================================================================
// 3. UDS Frame Transmission Over SocketPair (With & Without SCM_RIGHTS FD)
// ============================================================================

#[test]
fn test_uds_frame_transmission_without_fd() {
    let (fd_a, fd_b) = create_uds_socket_pair().expect("create socketpair");
    let stream_a = UnixStream::from(fd_a);
    let stream_b = UnixStream::from(fd_b);

    // Send AllocateArenaRequest
    let req = IpcMessage::AllocateArenaRequest(AllocateArenaRequest {
        model_id: "deepseek-r1-distill-qwen-1.5b".to_owned(),
        arena_bytes: 256 * 1024 * 1024,
        n_ctx: 2048,
        max_blocks: 128,
    });
    send_msg_frame(&stream_a, &req, None).expect("send_msg_frame without fd");

    let (recv_msg, opt_fd) = recv_msg_frame(&stream_b).expect("recv_msg_frame");
    assert_eq!(recv_msg, req);
    assert!(opt_fd.is_none());

    // Send response back from B to A
    let resp = IpcMessage::AllocateArenaResponse(AllocateArenaResponse {
        arena_id: 777,
        arena_bytes: 256 * 1024 * 1024,
        status: StatusCode::Success,
        error_message: None,
    });
    send_msg_frame(&stream_b, &resp, None).expect("send response");

    let (recv_resp, opt_fd2) = recv_msg_frame(&stream_a).expect("recv response");
    assert_eq!(recv_resp, resp);
    assert!(opt_fd2.is_none());
}

#[test]
fn test_uds_frame_transmission_with_scm_rights_fd() {
    let (fd_a, fd_b) = create_uds_socket_pair().expect("create socketpair");
    let stream_a = UnixStream::from(fd_a);
    let stream_b = UnixStream::from(fd_b);

    // Create a temporary file to pass as FD
    let (temp_path, mut temp_file) = create_test_temp_file();
    let test_data = b"K3_SHARED_MEMORY_ARENA_DISCRIMINATOR_VERIFICATION_BYTES";
    temp_file.write_all(test_data).expect("write temp file");
    temp_file.flush().expect("flush temp file");

    let pass_fd = temp_file.as_raw_fd();

    let cmd = IpcMessage::AttachArenaCommand(AttachArenaCommand {
        arena_id: 12345,
        arena_bytes: 4096,
        n_ctx: 64,
        max_blocks: 4,
    });

    send_msg_frame(&stream_a, &cmd, Some(pass_fd)).expect("send_msg_frame with fd");

    let (recv_cmd, opt_fd) = recv_msg_frame(&stream_b).expect("recv_msg_frame with fd");
    assert_eq!(recv_cmd, cmd);
    assert!(opt_fd.is_some(), "Expected SCM_RIGHTS FD attachment");

    let transferred_fd = opt_fd.expect("opt_fd");
    let mut transferred_file = File::from(transferred_fd);

    // Verify reading through transferred descriptor produces exact same file content
    transferred_file.seek(SeekFrom::Start(0)).expect("seek transferred file");
    let mut read_buf = Vec::new();
    transferred_file.read_to_end(&mut read_buf).expect("read transferred file");
    assert_eq!(read_buf, test_data);

    // Mutate through transferred descriptor and verify sender observes mutation
    let append_data = b"_MUTATION_OK";
    transferred_file.write_all(append_data).expect("write via transferred fd");
    transferred_file.flush().expect("flush via transferred fd");

    temp_file.seek(SeekFrom::Start(0)).expect("seek original file");
    let mut updated_buf = Vec::new();
    temp_file.read_to_end(&mut updated_buf).expect("read original file");
    let mut expected_all = test_data.to_vec();
    expected_all.extend_from_slice(append_data);
    assert_eq!(updated_buf, expected_all);

    let _ = std::fs::remove_file(temp_path);
}

// ============================================================================
// 4. Corrupted Magic Rejection Verification
// ============================================================================

#[test]
fn test_corrupted_magic_rejected_safely() {
    let bad_magics = [
        0x0000_0000_u32,
        0xFFFF_FFFF_u32,
        0x4B33_4951_u32, // Off-by-one magic
        0xDEAD_BEEF_u32,
        0x4950_4B33_u32, // Endian reversed
    ];

    for bad_magic in bad_magics {
        let mut buf = [0_u8; FRAME_HEADER_LEN];
        buf[0..4].copy_from_slice(&bad_magic.to_be_bytes());
        buf[4..6].copy_from_slice(&0_u16.to_be_bytes());
        buf[6..8].copy_from_slice(&101_u16.to_be_bytes());
        buf[8..12].copy_from_slice(&64_u32.to_be_bytes());

        let decode_result = IpcFrameHeader::decode(&buf);
        assert!(decode_result.is_err(), "Magic 0x{bad_magic:08X} should be rejected");

        let io_err = decode_result.unwrap_err();
        assert_eq!(io_err.kind(), std::io::ErrorKind::InvalidData);
        assert!(
            io_err.to_string().contains("Invalid IPC frame magic"),
            "Error should mention invalid magic, got: {io_err}"
        );

        // Convert safely to IpcError and verify no panic
        let ipc_err = IpcError::from(io_err);
        assert!(matches!(ipc_err, IpcError::IoError { .. }));
    }

    // Also verify rejection over live socket
    let (fd_a, fd_b) = create_uds_socket_pair().expect("socketpair");

    let mut bad_header = [0_u8; FRAME_HEADER_LEN];
    bad_header[0..4].copy_from_slice(&0xBADF_00D0_u32.to_be_bytes());
    bad_header[6..8].copy_from_slice(&107_u16.to_be_bytes());
    bad_header[8..12].copy_from_slice(&0_u32.to_be_bytes());

    // Write bad header from fd_a
    let sent = unsafe { libc::write(fd_a.as_raw_fd(), bad_header.as_ptr().cast(), FRAME_HEADER_LEN) };
    assert_eq!(sent, FRAME_HEADER_LEN as isize);

    // fd_b calls recv_frame and must safely reject
    let recv_res = recv_frame(fd_b.as_raw_fd());
    assert!(recv_res.is_err());
    let err = recv_res.unwrap_err();
    assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
    assert!(err.to_string().contains("Invalid IPC frame magic"));

    let ipc_err = IpcError::from(err);
    assert!(matches!(ipc_err, IpcError::IoError { .. }));
}

// ============================================================================
// 5. Oversized Payload Rejection Verification
// ============================================================================

#[test]
fn test_oversized_payload_rejected_safely() {
    let oversized_lengths = [
        MAX_PAYLOAD_LEN + 1,
        MAX_PAYLOAD_LEN + 1024,
        100 * 1024 * 1024, // 100 MiB
        u32::MAX,          // 4 GiB
    ];

    for bad_len in oversized_lengths {
        let mut buf = [0_u8; FRAME_HEADER_LEN];
        buf[0..4].copy_from_slice(&FRAME_MAGIC.to_be_bytes());
        buf[4..6].copy_from_slice(&0_u16.to_be_bytes());
        buf[6..8].copy_from_slice(&101_u16.to_be_bytes());
        buf[8..12].copy_from_slice(&bad_len.to_be_bytes());

        let decode_result = IpcFrameHeader::decode(&buf);
        assert!(decode_result.is_err(), "Payload len {bad_len} should be rejected");

        let io_err = decode_result.unwrap_err();
        assert_eq!(io_err.kind(), std::io::ErrorKind::InvalidData);
        assert!(
            io_err.to_string().contains("exceeds maximum allowed"),
            "Error should mention exceeding limit, got: {io_err}"
        );

        // Convert safely to IpcError and verify no panic
        let ipc_err = IpcError::from(io_err);
        assert!(matches!(ipc_err, IpcError::IoError { .. }));
    }

    // Verify rejection over live socket (must not attempt massive allocation)
    let (fd_a, fd_b) = create_uds_socket_pair().expect("socketpair");

    let mut header = [0_u8; FRAME_HEADER_LEN];
    header[0..4].copy_from_slice(&FRAME_MAGIC.to_be_bytes());
    header[6..8].copy_from_slice(&101_u16.to_be_bytes());
    header[8..12].copy_from_slice(&(MAX_PAYLOAD_LEN + 100).to_be_bytes());

    let sent = unsafe { libc::write(fd_a.as_raw_fd(), header.as_ptr().cast(), FRAME_HEADER_LEN) };
    assert_eq!(sent, FRAME_HEADER_LEN as isize);

    let recv_res = recv_frame(fd_b.as_raw_fd());
    assert!(recv_res.is_err());
    let err = recv_res.unwrap_err();
    assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
    assert!(err.to_string().contains("exceeds maximum allowed"));

    let ipc_err = IpcError::from(err);
    assert!(matches!(ipc_err, IpcError::IoError { .. }));
}

// ============================================================================
// 6. Streaming Pipeline Stress (Bursts of Interleaved Frames With/Without FD)
// ============================================================================

#[test]
fn test_streaming_pipeline_burst_with_interleaved_fds() {
    let (fd_a, fd_b) = create_uds_socket_pair().expect("socketpair");
    let stream_a = UnixStream::from(fd_a);
    let stream_b = UnixStream::from(fd_b);

    let burst_count: usize = 30;

    // Sender thread sends 30 messages, passing an FD on every 5th message
    let handle = std::thread::spawn(move || {
        let mut temp_paths = Vec::new();
        for i in 0..burst_count {
            if i % 5 == 0 {
                let (path, temp) = create_test_temp_file();
                let pass_fd = temp.as_raw_fd();
                let msg = IpcMessage::AttachArenaCommand(AttachArenaCommand {
                    arena_id: i as u64,
                    arena_bytes: 4096,
                    n_ctx: 64,
                    max_blocks: 4,
                });
                send_msg_frame(&stream_a, &msg, Some(pass_fd)).expect("send with fd");
                temp_paths.push(path);
            } else {
                let msg = IpcMessage::PingRequest(PingRequest {
                    timestamp_ns: i as u64 * 1000,
                });
                send_msg_frame(&stream_a, &msg, None).expect("send without fd");
            }
        }
        for path in temp_paths {
            let _ = std::fs::remove_file(path);
        }
    });

    for i in 0..burst_count {
        let (msg, opt_fd) = recv_msg_frame(&stream_b).expect("recv burst frame");
        if i % 5 == 0 {
            assert!(opt_fd.is_some(), "Frame {i} should carry SCM_RIGHTS FD");
            match msg {
                IpcMessage::AttachArenaCommand(cmd) => assert_eq!(cmd.arena_id, i as u64),
                other => panic!("Unexpected msg {other:?} at index {i}"),
            }
        } else {
            assert!(opt_fd.is_none(), "Frame {i} should NOT carry FD");
            match msg {
                IpcMessage::PingRequest(ping) => assert_eq!(ping.timestamp_ns, i as u64 * 1000),
                other => panic!("Unexpected msg {other:?} at index {i}"),
            }
        }
    }

    handle.join().expect("join sender thread");
}

// ============================================================================
// 7. Adversarial Edge Cases
// ============================================================================

#[test]
fn test_adversarial_flag_fd_attached_without_actual_fd_rejected() {
    let (fd_a, fd_b) = create_uds_socket_pair().expect("socketpair");

    // Manually craft a frame with FLAG_FD_ATTACHED set, but pass None for fd_to_pass
    let header = IpcFrameHeader::new(101, 8, FLAG_FD_ATTACHED);
    let mut header_buf = [0_u8; FRAME_HEADER_LEN];
    header.encode(&mut header_buf);

    let dummy_payload = [0_u8; 8];

    // Write header + payload without SCM_RIGHTS cmsg
    unsafe {
        libc::write(fd_a.as_raw_fd(), header_buf.as_ptr().cast(), FRAME_HEADER_LEN);
        libc::write(fd_a.as_raw_fd(), dummy_payload.as_ptr().cast(), 8);
    }

    let recv_res = recv_frame(fd_b.as_raw_fd());
    assert!(recv_res.is_err());
    let err = recv_res.unwrap_err();
    assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
    assert!(err.to_string().contains("FLAG_FD_ATTACHED but no SCM_RIGHTS FD was delivered"));
}

#[test]
fn test_adversarial_unknown_type_id_rejected() {
    let fory = global_fory();
    let res = IpcMessage::deserialize(999, b"some_dummy_bytes", fory);
    assert!(res.is_err());
    let err_str = res.unwrap_err().to_string();
    assert!(err_str.contains("Unknown IPC type ID: 999"));
}

#[test]
fn test_adversarial_truncated_header_returns_unexpected_eof() {
    let (fd_a, fd_b) = create_uds_socket_pair().expect("socketpair");

    // Send only 6 bytes of the 12-byte header, then close sender
    let partial_header = [0x4Bu8, 0x33, 0x49, 0x50, 0x00, 0x00];
    unsafe {
        libc::write(fd_a.as_raw_fd(), partial_header.as_ptr().cast(), 6);
    }
    drop(fd_a);

    let res = recv_frame(fd_b.as_raw_fd());
    assert!(res.is_err());
    assert_eq!(res.unwrap_err().kind(), std::io::ErrorKind::UnexpectedEof);
}

#[test]
fn test_adversarial_truncated_payload_returns_unexpected_eof() {
    let (fd_a, fd_b) = create_uds_socket_pair().expect("socketpair");

    // Valid header declaring 64 bytes of payload, but only send 10 bytes then close
    let header = IpcFrameHeader::new(107, 64, 0);
    let mut header_buf = [0_u8; FRAME_HEADER_LEN];
    header.encode(&mut header_buf);

    unsafe {
        libc::write(fd_a.as_raw_fd(), header_buf.as_ptr().cast(), FRAME_HEADER_LEN);
        let partial_payload = [0xAA_u8; 10];
        libc::write(fd_a.as_raw_fd(), partial_payload.as_ptr().cast(), 10);
    }
    drop(fd_a);

    let res = recv_frame(fd_b.as_raw_fd());
    assert!(res.is_err());
    assert_eq!(res.unwrap_err().kind(), std::io::ErrorKind::UnexpectedEof);
}

#[test]
fn test_adversarial_corrupted_payload_returns_fory_error() {
    let (fd_a, fd_b) = create_uds_socket_pair().expect("socketpair");
    let stream_a = UnixStream::from(fd_a);
    let stream_b = UnixStream::from(fd_b);

    // Send a frame with Type ID 101 (AllocateArenaRequest) but random corrupted bytes as payload
    let corrupt_payload = vec![0xFF_u8; 32];
    send_frame(stream_a.as_raw_fd(), 101, &corrupt_payload, None).expect("send_frame");

    let res = recv_msg_frame(&stream_b);
    assert!(res.is_err());
    let err = res.unwrap_err();
    assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
    assert!(err.to_string().contains("Fory deserialization error"));
}

#[test]
fn test_adversarial_boundary_and_unicode_payloads() {
    let fory = global_fory();

    // 1. Extreme unicode string model ID
    let unicode_model_id = "🧠⚡_model_测试_español_кириллица_日本語_1234567890".to_owned();
    let msg_unicode = IpcMessage::AllocateArenaRequest(AllocateArenaRequest {
        model_id: unicode_model_id.clone(),
        arena_bytes: u64::MAX,
        n_ctx: u32::MAX,
        max_blocks: u32::MAX,
    });
    let (t_uni, b_uni) = msg_unicode.serialize(fory).expect("serialize unicode model_id");
    let dec_uni = IpcMessage::deserialize(t_uni, &b_uni, fory).expect("deserialize unicode");
    assert_eq!(dec_uni, msg_unicode);

    // 2. Large block indices vector (10,000 blocks)
    let large_indices: Vec<u32> = (0..10_000).collect();
    let msg_large_blocks = IpcMessage::ForwardPassCommand(ForwardPassCommand {
        request_id: u64::MAX,
        sequence_id: u64::MAX,
        token_offset: u32::MAX,
        token_count: 10_000,
        block_indices: large_indices,
        is_prefill: false,
    });
    let (t_lb, b_lb) = msg_large_blocks.serialize(fory).expect("serialize large blocks");
    let dec_lb = IpcMessage::deserialize(t_lb, &b_lb, fory).expect("deserialize large blocks");
    assert_eq!(dec_lb, msg_large_blocks);

    // 3. Extreme float latencies (subnormal and huge)
    let msg_floats = IpcMessage::ForwardPassCompleted(ForwardPassCompleted {
        request_id: 1,
        sequence_id: 1,
        committed_tokens: 0,
        prefill_latency_ms: f64::MIN_POSITIVE,
        decode_latency_ms: 1e15,
        next_token: u32::MAX,
        status: StatusCode::ProtocolError,
        error_message: Some(String::new()), // Empty string vs None
    });
    let (t_fl, b_fl) = msg_floats.serialize(fory).expect("serialize floats");
    let dec_fl = IpcMessage::deserialize(t_fl, &b_fl, fory).expect("deserialize floats");
    assert_eq!(dec_fl, msg_floats);
}

#[test]
fn test_adversarial_invalid_raw_fd_returns_io_error() {
    let (fd_a, _fd_b) = create_uds_socket_pair().expect("socketpair");

    // Passing an invalid file descriptor (-1) must fail safely at libc::sendmsg with EBADF
    let payload = b"PAYLOAD_WITH_BAD_FD";
    let res = send_frame(fd_a.as_raw_fd(), 107, payload, Some(-1));
    assert!(res.is_err(), "Passing invalid FD (-1) must return Err");
    let err = res.unwrap_err();
    assert_eq!(err.raw_os_error(), Some(libc::EBADF));
}

#[test]
fn test_adversarial_status_code_ipc_error_mapping_gap() {
    // Challenge observation:
    // When framing error occurs (bad magic), IpcFrameHeader::decode returns io::ErrorKind::InvalidData.
    let mut bad_buf = [0_u8; FRAME_HEADER_LEN];
    bad_buf[0..4].copy_from_slice(&0xDEAD_BEEF_u32.to_be_bytes());
    let decode_err = IpcFrameHeader::decode(&bad_buf).unwrap_err();

    let ipc_err = IpcError::from(decode_err);
    // Empirical verification:
    // Currently, From<io::Error> unconditionally constructs IpcError::IoError.
    // Therefore StatusCode::from_ipc_error(&ipc_err) produces InternalError (21)
    // rather than ProtocolError (22).
    let status = StatusCode::from_ipc_error(&ipc_err);
    assert_eq!(status, StatusCode::InternalError);
    assert_eq!(status.as_exit_code(), 21);

    // Meanwhile, if an explicit ProtocolError is constructed, it maps to code 22:
    let explicit_proto_err = IpcError::ProtocolError("framing violation".to_owned());
    assert_eq!(StatusCode::from_ipc_error(&explicit_proto_err), StatusCode::ProtocolError);
    assert_eq!(StatusCode::from_ipc_error(&explicit_proto_err).as_exit_code(), 22);
}

#[test]
fn test_adversarial_mismatched_type_id_deserialization() {
    let fory = global_fory();

    // Serialize a ForwardPassCommand (Type ID 202)
    let fwd_cmd = IpcMessage::ForwardPassCommand(ForwardPassCommand {
        request_id: 123,
        sequence_id: 456,
        token_offset: 0,
        token_count: 10,
        block_indices: vec![1, 2, 3],
        is_prefill: true,
    });
    let (_, bytes) = fwd_cmd.serialize(fory).expect("serialize fwd_cmd");

    // Attempt to deserialize those bytes as PingRequest (Type ID 107)
    // Should fail cleanly without crashing or undefined behavior
    let res = IpcMessage::deserialize(107, &bytes, fory);
    assert!(res.is_err(), "Deserializing mismatched type bytes should fail");
}

