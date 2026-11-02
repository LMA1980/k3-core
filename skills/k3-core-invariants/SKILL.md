---
name: k3-core-invariants
description: "Implement ArchRecipe traits, zero-copy TensorView, GGUF v3 parsing, SpacemiT K3 cluster affinity, and SCM_RIGHTS FD passing with k3-core."
---

# `k3-core` Hardware Topology, Zero-Copy Tensor & IPC Invariants

## 1. SpacemiT K3 Canonical Cluster Mapping & `/proc/set_ai_thread`

- **Cluster 0 (`X100`, cores `0–7`, mask `0x00FF`)**: Out-of-Order general-purpose cores (`2.15 GHz`), `VLEN = 256` (`vlenb = 32`), supports RISC-V Hypervisor (`H`) extension (`/dev/kvm`). Does **not** support `XSMTVDotII` matrix instructions (`SIGILL` if executed on Cluster 0).
- **Cluster 1 (`A100`, cores `8–15`, mask `0xFF00`)**: In-Order AI/Matrix cores (`1.80 GHz`), `VLEN = 1024` (`vlenb = 128`), supports `XSMTVDotII` matrix instructions.
- **Conditional `/proc/set_ai_thread` Registration**: On SpacemiT-patched kernels (`spacemit-hmp`), any worker thread or forked capability-probe process pinning to Cluster 1 (`A100`, cores `8–15`) **must** write its kernel thread ID (`SYS_gettid`) to `/proc/set_ai_thread` before calling `sched_setaffinity`. Treat `/proc/set_ai_thread` as conditional (write only when the procfs node exists) so vanilla Linux kernels remain supported.

## 2. Rust Integer Sizing (`usize` vs. `u32`) & Strict Lints

- Use `usize` strictly for indexing collections, slice/array lengths, and memory allocation offsets (`TensorView` byte offsets).
- Use `u32` for domain-specific model metadata, token IDs, GGUF header counters, and wire protocol fields where 32-bit width must be architecture-independent.
- Every `unsafe` block must carry a `// SAFETY:` comment (`clippy::undocumented_unsafe_blocks = "warn"`, `unsafe_op_in_unsafe_fn = "deny"`).
