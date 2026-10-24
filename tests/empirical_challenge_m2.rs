//! Empirical Challenger Test Suite for Milestone 2: `SharedMemoryArena` & Arrow Views.

#![allow(
    clippy::all,
    clippy::pedantic,
    clippy::nursery,
    clippy::undocumented_unsafe_blocks
)]

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread;

use k3_core::ipc::layout::{
    ArenaHeader, OutputSlabHeader, TokenInputSlotHeader, SLAB_STATUS_WRITTEN,
};
use k3_core::ipc::{
    wrap_floats_as_float32_array, wrap_tokens_as_uint32_array, ArenaOptions, IpcError,
    SharedMemoryArena, ShmTensorViews, DEFAULT_LIMIT_MEMLOCK_BYTES,
};
use k3_core::K3Error;
use zerocopy::IntoBytes;

#[test]
fn test_adversarial_out_of_bounds_slicing_returns_tensor_bounds_error() {
    let mut arena = SharedMemoryArena::create("oob_challenge", 4096).expect("create arena");
    let arena_len = arena.len();
    assert!(arena_len >= 4096);

    // 1. One byte past capacity
    let err_one_past = arena.slice(0, arena_len + 1);
    match err_one_past {
        Err(K3Error::TensorBoundsError {
            offset,
            byte_len,
            buffer_len,
            reason,
            ..
        }) => {
            assert_eq!(offset, 0);
            assert_eq!(byte_len, arena_len + 1);
            assert_eq!(buffer_len, arena_len);
            assert!(reason.contains("exceeds arena capacity"));
        }
        other => panic!("Expected TensorBoundsError, got {other:?}"),
    }

    // 2. Start offset at capacity with non-zero length
    let err_at_capacity = arena.slice(arena_len, 1);
    assert!(matches!(
        err_at_capacity,
        Err(K3Error::TensorBoundsError { .. })
    ));

    // 3. Overflow usize calculation (offset + len wraps)
    let err_overflow = arena.slice(usize::MAX - 10, 20);
    match err_overflow {
        Err(K3Error::TensorBoundsError { reason, .. }) => {
            assert!(reason.contains("overflowed usize"));
        }
        other => panic!("Expected overflow TensorBoundsError, got {other:?}"),
    }

    // 4. Mutable slicing out-of-bounds
    let err_mut_oob = arena.slice_mut(arena_len / 2, arena_len);
    assert!(matches!(
        err_mut_oob,
        Err(K3Error::TensorBoundsError { .. })
    ));

    // 5. Mutable slicing overflow
    let err_mut_overflow = arena.slice_mut(usize::MAX - 5, 10);
    assert!(matches!(
        err_mut_overflow,
        Err(K3Error::TensorBoundsError { .. })
    ));

    // 6. Exact boundary slice (offset == arena_len, len == 0) succeeds
    let empty_slice = arena.slice(arena_len, 0).expect("zero-length slice at end");
    assert_eq!(empty_slice.len(), 0);

    // 7. Slicing beyond arena_len even with len == 0 fails
    let beyond_empty = arena.slice(arena_len + 1, 0);
    assert!(matches!(
        beyond_empty,
        Err(K3Error::TensorBoundsError { .. })
    ));
}

#[test]
fn test_read_only_arena_rejects_all_mutations_with_tensor_bounds_error() {
    let arena_source = SharedMemoryArena::create("ro_source", 4096).expect("create rw");
    let dup_fd = arena_source.duplicate_fd().expect("dup_fd");
    let mut arena_target =
        SharedMemoryArena::from_existing_fd(dup_fd, arena_source.len(), "ro_mapped", true)
            .expect("create ro");

    assert!(arena_target.is_read_only());

    // slice_mut must fail with TensorBoundsError
    let err = arena_target.slice_mut(0, 64);
    match err {
        Err(K3Error::TensorBoundsError { reason, .. }) => {
            assert!(reason.contains("read-only"));
        }
        other => panic!("Expected TensorBoundsError for read-only slice_mut, got {other:?}"),
    }

    // header_mut must fail
    assert!(matches!(
        arena_target.header_mut(),
        Err(K3Error::TensorBoundsError { .. })
    ));

    // block_meta_mut must fail
    assert!(matches!(
        arena_target.block_meta_mut(0),
        Err(K3Error::TensorBoundsError { .. })
    ));
}

#[test]
fn test_cross_mapping_mutation_coherence_immediately_visible() {
    // Create primary mapping (read-write)
    let mut arena_primary =
        SharedMemoryArena::create("coherence_arena", 8192).expect("create primary");
    let dup_fd1 = arena_primary.duplicate_fd().expect("dup fd 1");
    let dup_fd2 = arena_primary.duplicate_fd().expect("dup fd 2");

    // Create secondary read-write mapping
    let mut arena_secondary =
        SharedMemoryArena::from_existing_fd(dup_fd1, arena_primary.len(), "secondary", false)
            .expect("create secondary");

    // Create tertiary read-only mapping
    let arena_tertiary =
        SharedMemoryArena::from_existing_fd(dup_fd2, arena_primary.len(), "tertiary", true)
            .expect("create tertiary");

    // Verify all 3 mappings have non-null base pointers
    assert!(!arena_primary.as_ptr().is_null());
    assert!(!arena_secondary.as_ptr().is_null());
    assert!(!arena_tertiary.as_ptr().is_null());

    // 1. Initial state: all zeros
    assert_eq!(
        arena_secondary.slice(128, 8).expect("sec slice"),
        &[0; 8]
    );
    assert_eq!(
        arena_tertiary.slice(128, 8).expect("tert slice"),
        &[0; 8]
    );

    // 2. Primary writes pattern to offset 128
    let pattern_a = [0x5A, 0xA5, 0x12, 0x34, 0xFE, 0xDC, 0xBA, 0x98];
    {
        let slice = arena_primary.slice_mut(128, 8).expect("slice_mut primary");
        slice.copy_from_slice(&pattern_a);
    }

    // Secondary and tertiary must IMMEDIATELY see pattern_a without re-mapping
    assert_eq!(
        arena_secondary.slice(128, 8).expect("read secondary"),
        &pattern_a,
        "Secondary mapping must immediately observe writes from primary"
    );
    assert_eq!(
        arena_tertiary.slice(128, 8).expect("read tertiary"),
        &pattern_a,
        "Tertiary mapping must immediately observe writes from primary"
    );

    // 3. Secondary writes pattern to offset 4096 (second page boundary)
    let pattern_b = [0xDE, 0xAD, 0xBE, 0xEF, 0xCA, 0xFE, 0xBA, 0xBE];
    {
        let slice = arena_secondary.slice_mut(4096, 8).expect("slice_mut secondary");
        slice.copy_from_slice(&pattern_b);
    }

    // Primary and tertiary must IMMEDIATELY observe pattern_b
    assert_eq!(
        arena_primary.slice(4096, 8).expect("read primary"),
        &pattern_b,
        "Primary mapping must immediately observe writes from secondary"
    );
    assert_eq!(
        arena_tertiary.slice(4096, 8).expect("read tertiary"),
        &pattern_b,
        "Tertiary mapping must immediately observe writes from secondary"
    );
}

#[test]
fn test_arrow_tensor_views_pointer_identity_and_live_mutation() {
    let mut arena =
        SharedMemoryArena::create("arrow_ptr_challenge", 16384).expect("create arena");
    let arena_ptr = arena.as_ptr();

    // 1. UInt32Array pointer bit-for-bit check
    let token_offset = 512;
    let token_count = 16;
    // SAFETY: token_offset is within arena bounds (16384).
    let expected_token_ptr = unsafe { arena_ptr.add(token_offset).cast::<u32>() };

    // Initialize tokens
    let initial_tokens: Vec<u32> = (1000..1016).collect();
    arena
        .slice_mut(token_offset, token_count * 4)
        .expect("slice_mut tokens")
        .copy_from_slice(initial_tokens.as_bytes());

    let arena_arc = Arc::new(arena);
    // SAFETY: expected_token_ptr points to valid initialized tokens within the arena.
    let uint_array = unsafe {
        wrap_tokens_as_uint32_array(
            Arc::clone(&arena_arc),
            expected_token_ptr,
            token_count,
        )
    }
    .expect("wrap tokens");

    // Assert underlying Arrow buffer pointer matches shared memory pointer bit-for-bit
    assert_eq!(
        uint_array.values().as_ptr(),
        expected_token_ptr,
        "UInt32Array pointer MUST match arena memory address bit-for-bit"
    );
    assert_eq!(uint_array.len(), token_count);
    for i in 0..token_count {
        assert_eq!(uint_array.value(i), 1000 + i as u32);
    }

    // 2. Float32Array pointer bit-for-bit check
    let float_offset = 1024;
    let float_count = 32;
    // SAFETY: float_offset is within arena bounds (16384).
    let expected_float_ptr = unsafe { arena_ptr.add(float_offset).cast::<f32>() };

    let initial_floats: Vec<f32> = (0..32).map(|x| x as f32 * 0.5).collect();
    // SAFETY: mut_slice points to float_count * 4 valid bytes in the arena.
    unsafe {
        let mut_slice = std::slice::from_raw_parts_mut(
            arena_ptr.add(float_offset).cast_mut(),
            float_count * 4,
        );
        mut_slice.copy_from_slice(initial_floats.as_bytes());
    }

    // SAFETY: expected_float_ptr is 4-byte aligned and references initialized float memory.
    let float_array = unsafe {
        wrap_floats_as_float32_array(
            Arc::clone(&arena_arc),
            "test_logits",
            expected_float_ptr,
            float_count,
        )
    }
    .expect("wrap floats");

    // Assert underlying Arrow buffer pointer matches shared memory pointer bit-for-bit
    assert_eq!(
        float_array.values().as_ptr(),
        expected_float_ptr,
        "Float32Array pointer MUST match arena memory address bit-for-bit"
    );
    assert_eq!(float_array.len(), float_count);
    for i in 0..float_count {
        assert!((float_array.value(i) - (i as f32 * 0.5)).abs() < f32::EPSILON);
    }

    // 3. Live in-place mutation through shared memory
    // SAFETY: Modify underlying shared memory directly via raw pointer within allocated bounds.
    unsafe {
        let raw_token_mut = arena_ptr.add(token_offset).cast_mut().cast::<u32>();
        *raw_token_mut.add(0) = 9999;
        *raw_token_mut.add(5) = 8888;
    }

    // The existing Arrow array MUST reflect the new values immediately without re-wrapping
    assert_eq!(
        uint_array.value(0),
        9999,
        "Arrow array must reflect underlying in-place shared memory modification"
    );
    assert_eq!(
        uint_array.value(5),
        8888,
        "Arrow array must reflect underlying in-place shared memory modification"
    );

    // 4. ShmTensorViews end-to-end pointer bit-for-bit check
    let mut views_arena =
        SharedMemoryArena::create("views_arena", 65536).expect("views arena");
    let views_ptr = views_arena.as_ptr();

    let token_slots_off: u64 = 2048;
    let output_slabs_off: u64 = 8192;
    {
        let hdr = views_arena.header_mut().expect("header_mut");
        *hdr = ArenaHeader::default();
        hdr.token_slots_offset = token_slots_off;
        hdr.output_slabs_offset = output_slabs_off;
    }

    // Setup token slot 0
    let token_slot_base = usize::try_from(token_slots_off).expect("token_slot_base");
    {
        let slot_hdr = TokenInputSlotHeader {
            slot_seq_id: 42,
            token_len: 4,
            start_position: 0,
        };
        views_arena
            .slice_mut(token_slot_base, 16)
            .expect("slot_hdr")
            .copy_from_slice(slot_hdr.as_bytes());
        let tokens: [u32; 4] = [101, 102, 103, 104];
        views_arena
            .slice_mut(token_slot_base + 16, 16)
            .expect("tokens")
            .copy_from_slice(tokens.as_bytes());
    }

    // Setup output slab
    let slab_base = usize::try_from(output_slabs_off).expect("slab_base");
    {
        let slab_hdr = OutputSlabHeader {
            request_id: 123,
            vocab_size: 8,
            position: 1,
            status: SLAB_STATUS_WRITTEN,
            _reserved: [0; 44],
        };
        views_arena
            .slice_mut(slab_base, 64)
            .expect("slab_hdr")
            .copy_from_slice(slab_hdr.as_bytes());
        let logits: [f32; 8] = [1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8];
        views_arena
            .slice_mut(slab_base + 64, 32)
            .expect("logits")
            .copy_from_slice(logits.as_bytes());
    }

    let views = ShmTensorViews::new(Arc::new(views_arena));
    let token_view = views.tokens_view(0).expect("token_view");
    let logits_view = views.logits_view().expect("logits_view");

    // SAFETY: token_slot_base + 16 and slab_base + 64 are aligned and bounded.
    let expected_view_token_ptr =
        unsafe { views_ptr.add(token_slot_base + 16).cast::<u32>() };
    // SAFETY: slab_base + 64 is aligned and within arena bounds.
    let expected_view_logits_ptr =
        unsafe { views_ptr.add(slab_base + 64).cast::<f32>() };

    assert_eq!(
        token_view.values().as_ptr(),
        expected_view_token_ptr,
        "ShmTensorViews tokens pointer must match arena memory address bit-for-bit"
    );
    assert_eq!(
        logits_view.values().as_ptr(),
        expected_view_logits_ptr,
        "ShmTensorViews logits pointer must match arena memory address bit-for-bit"
    );
}

#[test]
fn test_arrow_views_reject_unaligned_pointers() {
    let arena = SharedMemoryArena::create("align_test", 4096).expect("create arena");
    let arena_ptr = arena.as_ptr();
    let arena_arc = Arc::new(arena);

    // Test unaligned pointers (offset +1, +2, +3 bytes)
    for offset in [1, 2, 3, 5, 7] {
        // SAFETY: offset is within arena bounds (4096).
        let unaligned_token_ptr = unsafe { arena_ptr.add(offset).cast::<u32>() };
        // SAFETY: Testing rejection of unaligned pointer by wrap_tokens_as_uint32_array.
        let result = unsafe {
            wrap_tokens_as_uint32_array(Arc::clone(&arena_arc), unaligned_token_ptr, 4)
        };
        match result {
            Err(K3Error::TensorBoundsError { reason, .. }) => {
                assert!(reason.contains("not 4-byte aligned"), "Reason: {reason}");
            }
            other => panic!("Expected alignment TensorBoundsError for offset {offset}, got {other:?}"),
        }

        // SAFETY: offset is within arena bounds (4096).
        let unaligned_float_ptr = unsafe { arena_ptr.add(offset).cast::<f32>() };
        // SAFETY: Testing rejection of unaligned pointer by wrap_floats_as_float32_array.
        let result_f = unsafe {
            wrap_floats_as_float32_array(
                Arc::clone(&arena_arc),
                "unaligned_f32",
                unaligned_float_ptr,
                4,
            )
        };
        match result_f {
            Err(K3Error::TensorBoundsError { reason, .. }) => {
                assert!(reason.contains("not 4-byte aligned"), "Reason: {reason}");
            }
            other => panic!("Expected alignment TensorBoundsError for offset {offset}, got {other:?}"),
        }
    }
}

#[test]
fn test_memlock_ceiling_bound_invariant() {
    // 28 GiB ceiling invariant per AGENTS.md Rule 5 and ADR-037
    assert_eq!(DEFAULT_LIMIT_MEMLOCK_BYTES, 28 * 1024 * 1024 * 1024);

    // Requesting 29 GiB with lock_memory=true must be rejected without mlock syscall
    let over_ceiling_bytes = 29 * 1024 * 1024 * 1024;
    let opts = ArenaOptions::new("over_ceiling", 4096)
        .with_lock(true)
        .with_memlock_ceiling(DEFAULT_LIMIT_MEMLOCK_BYTES);

    let mut large_opts = opts;
    large_opts.len = usize::try_from(over_ceiling_bytes).unwrap_or(usize::MAX);

    let res = SharedMemoryArena::create_with_options(large_opts);
    match res {
        Err(IpcError::InsufficientLockableMemory(K3Error::InsufficientLockableMemory {
            required_bytes,
            rlimit_memlock_bytes,
            ..
        })) => {
            assert!(required_bytes >= over_ceiling_bytes);
            assert_eq!(rlimit_memlock_bytes, DEFAULT_LIMIT_MEMLOCK_BYTES);
        }
        other => panic!("Expected InsufficientLockableMemory, got {other:?}"),
    }
}

#[test]
fn test_concurrent_multi_mapping_stress() {
    let arena = SharedMemoryArena::create("stress_arena", 65536).expect("create stress arena");
    let arena_arc = Arc::new(arena);
    let stop = Arc::new(AtomicBool::new(false));

    // Spawn 4 reader threads continuously validating data coherence
    let mut handles = Vec::new();
    for thread_idx in 0..4 {
        let dup_fd = arena_arc.duplicate_fd().expect("dup fd for thread");
        let len = arena_arc.len();
        let stop_clone = Arc::clone(&stop);

        handles.push(thread::spawn(move || {
            let reader_arena =
                SharedMemoryArena::from_existing_fd(dup_fd, len, &format!("reader_{thread_idx}"), true)
                    .expect("reader arena");
            let mut read_cycles = 0;
            while !stop_clone.load(Ordering::Relaxed) && read_cycles < 20_000 {
                let slice = reader_arena.slice(0, 8).expect("reader slice");
                let val = u64::from_le_bytes(slice.try_into().expect("u64 slice"));
                // Verify value is a valid counter state (sequential writes from writer)
                assert!(val <= 20_000);
                read_cycles += 1;
            }
            read_cycles
        }));
    }

    // Writer thread updating counter in primary arena
    let dup_fd_w = arena_arc.duplicate_fd().expect("dup writer fd");
    let len_w = arena_arc.len();
    let writer_handle = thread::spawn(move || {
        let mut writer_arena =
            SharedMemoryArena::from_existing_fd(dup_fd_w, len_w, "writer", false)
                .expect("writer arena");
        for step in 1_u64..=10_000_u64 {
            let slice = writer_arena.slice_mut(0, 8).expect("writer slice");
            slice.copy_from_slice(&step.to_le_bytes());
        }
    });

    writer_handle.join().expect("writer joined");
    stop.store(true, Ordering::Relaxed);

    for handle in handles {
        let cycles = handle.join().expect("reader joined");
        assert!(cycles > 0);
    }
}
