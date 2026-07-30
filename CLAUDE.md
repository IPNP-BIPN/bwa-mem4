# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 🎯 Core Purpose

Project goal: Accurate DNA read alignment against reference genome. Tool optimizes read mapping throughput and accuracy by combining high-speed indexing (FM-index) with hardware-accelerated dynamic programming (Smith-Waterman). Output *must* be byte-identical to `bwa-mem2` oracle for validity checks.

## 🛠️ Development Workflow & Commands

### A. Setup
1.  **Dependencies:** Core runtime relies heavily on `rust-htslib`. Ensure the build environment has necessary system libraries (e.g., htslib development headers).
2.  **Initial Run/Smoke Test:** Always run `./scripts/check.sh` first. This script executes:
    *   `cargo fmt --check`: Checks formatting compliance.
    *   `cargo clippy --workspace ... -D warnings`: Runs full linting suite for style and best practices violations.
    *   `cargo test --workspace`: Executes unit tests across all crates.

### B. Build & Deployment
1.  **Standard Release:** Use `cargo build --release` for general compilation.
2.  **Performance Optimization (PGO):** For maximum speed, use the profiled PGO workflow:
    *   `cargo pgo build`: Instrument code.
    *   (Run full workload/benchmark).
    *   `cargo pgo optimize build`: Final optimized binary creation.

### C. Testing & Validation Gates
Testing is multi-layered and cannot rely solely on unit tests. Integration validation against an oracle (`bwa-mem2`) is mandatory.
1.  **Unit Tests:** `cargo test --workspace`.
2.  **Functional Parity Check (Crucial):** Run dedicated scripts in `./scripts/`:
    *   `./scripts/oracle_diff.sh`: Compares SAM outputs against the oracle for full record parity check.
    *   `./scripts/index_diff.sh`: Ensures index construction is byte-identical to established indices.

## 🏗️ Code Architecture (Big Picture)

The system operates as a strict, multi-stage bioinformatics pipeline:

**1. Data Flow Path:** `FASTQ Input` $\rightarrow$ **Encoding** (`bwa-core`) $\rightarrow$ **Seeding** (`bwa-seed`) $\rightarrow$ **Chaining** (`bwa-chain`) $\rightarrow$ **Extension/Scoring** (DP Kernel) $\rightarrow$ **Primary Resolution** (`bwa-mem`) $\rightarrow$ `SAM Output`.

**2. Key Crates & Responsibilities:**
*   `bwa-index`: Manages the FM-index structure, which enables fast search over the genome.
*   `bwa-seed`: Locates initial Super-Maximal Exact Matches (SMEMs). Acts as the filter/pre-processor for seeds.
*   `bwa-chain`: Groups related SMEMs into plausible genomic path segments.
*   `bwa-extend`/`bwa-neon`: Implements banded Smith-Waterman DP. Contains specialized traits (`SwBackend`) to enforce correctness across different hardware backends (Scalar/NEON/AVX2).

**3. Core Patterns:**
*   **Data Structure Efficiency:** Extensive use of bit packing and custom numeric formats ($\text{nt}4$ codes) in `bwa-core` for minimal memory footprint, critical for handling massive datasets.
*   **Modularity:** Separation of concerns is absolute: I/O (`bwa-io`), Indexing (`bwa-index`), Filtering (`bwa-seed`), and Scoring (`bwa-extend`) operate on distinct data primitives passed between crates.

## 🧩 Advanced Concepts & Libraries

*   **SIMD Traits (Performance):** The alignment kernels use Rust traits to abstract the DP calculation. This allows implementing `SwBackend` for different architectures (e.g., NEON) while using the scalar implementation as the single source of truth for functional testing (`assert_backend_matches_scalar`).
*   **IO Management:** Use `rust-htslib` for all reading/writing of binary sequence formats (BAM, CRAM). Never write raw byte streams without passing through this library wrapper.