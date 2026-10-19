# k3-core

Core traits, zero-copy tensor views, single-prompt execution context, and structured diagnostics for the SpacemiT K3 AI platform.

## Overview
`k3-core` defines the foundational abstractions shared across `k3-engine`, architecture recipe crates (`arch-*`), and compute kernel crates (`kernels-*`):
- **Quantization & Tensor Views**: `QuantType` (`F32`, `F16`, `BF16`, `Q4_0`, `Q8_0`, `Q4_K_M`, `Q6_K`), zero-copy `TensorView` with checked bounds calculation, and the `ModelWeights` trait.
- **Cycle-Exclusive Execution**: Single-prompt `ExecutionContext` and `ArchRecipe` (`prefill_step` / `decode_step`) enforcing strict prompt exclusivity (ADR-005) and fixed context bounds (ADR-027).
- **Structured Diagnostics**: `K3Error` covering capability mismatch (ADR-023), insufficient lockable memory (ADR-037), context length exceeded (ADR-027), core affinity errors, and tensor bounds errors.

## Architecture & Specifications
For master architecture and detailed component specifications, see the parent [model-runner](https://github.com/LMA1980/model-runner) repository:
- Master Plan: [PLAN.md](https://github.com/LMA1980/model-runner/blob/main/PLAN.md)
- Specs: [specs/runtime/](https://github.com/LMA1980/model-runner/tree/main/specs/runtime)
