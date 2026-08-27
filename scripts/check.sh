#!/usr/bin/env bash
# Local CI gate: format, lint, test. Mirrors .github/workflows/ci.yml.
#
# It does NOT cover .github/workflows/parity.yml, which is the check that actually protects
# byte-identity. Run that one too before proposing a change to the aligner:
#
#   python3 scripts/make_test_reads.py testdata/tiny/tiny.fa /tmp/ci --n 8000
#   IDX=testdata/tiny/tiny.fa R1=/tmp/ci_1.fq R2=/tmp/ci_2.fq \
#     bash scripts/opt_parity.sh ./target/release/bwa-mem4
#
# and remember `cargo test` does not relink target/release/bwa-mem4: `cargo build --release` first.
#
# It also does NOT cover the AVX-512 kernels, on any machine. Their byte-identity tests self-skip
# wherever `avx512bw` is absent, which is every arm64 host and most x86 ones, and a skip reports
# `ok`. On x86_64 Linux, `scripts/avx512_sde.sh` runs them for real under Intel SDE; everywhere else
# the `avx512-check` workflow does it on every pull request that touches `crates/bwa-neon/**`.
set -euo pipefail
cd "$(dirname "$0")/.."

echo "cargo: $(cargo --version)"
echo "== rustfmt =="
cargo fmt --check
echo "== clippy (-D warnings) =="
cargo clippy --workspace --all-targets -- -D warnings
# Off-by-default features are invisible to the sweep above. `stage-alloc` swaps the global
# allocator, so a break there is worth catching locally rather than the day the probe is needed.
echo "== clippy (stage-alloc feature) =="
cargo clippy -p bwa-mem4 --all-targets --features stage-alloc -- -D warnings
echo "== test =="
cargo test --workspace
echo "OK"

# --- cross-architecture check -------------------------------------------------------------------
# Development here is on Apple Silicon, so the x86_64 code paths (AVX2 kernels, the non-aarch64
# arms of every `cfg`) are never compiled locally by default. They have already shipped three
# breakages this way: dead aarch64-only items, an ungated macos-only call, and ARM inline assembly
# in a benchmark example. This catches all three classes without waiting for CI.
#
# It needs rustup's rustc, not Homebrew's: `cargo` on PATH is Homebrew's and its sysroot has no
# x86_64 std, which is why plain `cargo check --target x86_64-...` fails with "can't find crate for
# core". Skipped silently if the toolchain is not installed.
TC="$HOME/.rustup/toolchains/1.96.1-aarch64-apple-darwin"
if [ -x "$TC/bin/cargo" ] && [ -d "$TC/lib/rustlib/x86_64-apple-darwin" ]; then
  echo "== x86_64 cross-check =="
  # The x87 ABI warning is a macos-x86_64 target quirk unrelated to this code and absent on Linux.
  RUSTC="$TC/bin/rustc" PATH="$TC/bin:$PATH" "$TC/bin/cargo" clippy \
      --workspace --all-targets --target x86_64-apple-darwin -- -D warnings 2>&1 \
    | grep -E "^error" | grep -v "x87" && { echo "x86_64 check FAILED"; exit 1; }
  echo "x86_64 OK"

  # And actually RUN the x86_64 tests, under Rosetta. This works and it is not obvious that it
  # should: Rosetta 2 on this machine executes AVX2, so `is_x86_feature_detected!("avx2")` is true
  # and `avx2_matesw_*` / `avx2_u8_and_i16_match_scalar` really exercise the AVX2 kernels against
  # the scalar reference instead of skipping. That turns every AVX2 change from compile-checked
  # into verified, without an x86 box.
  #
  # `-C target-cpu=x86-64-v3` is required and replaces the workspace's `native`: `apple-m4` is not
  # an x86 CPU name, and rustc otherwise aborts with "64-bit code requested on a subtarget that
  # doesn't support it". v3 is the AVX2 level.
  #
  # AVX-512 is NOT covered: Rosetta has no `avx512bw`, so `avx512_matesw_*` take their
  # feature-detect early return and report `ok` without running the kernel. Those paths stay
  # CI-only.
  echo "== x86_64 tests under Rosetta (AVX2 kernels really run) =="
  RUSTC="$TC/bin/rustc" PATH="$TC/bin:$PATH" RUSTFLAGS="-C target-cpu=x86-64-v3" \
    "$TC/bin/cargo" test --workspace --target x86_64-apple-darwin --release >/dev/null 2>&1 \
    || { echo "x86_64 tests FAILED"; exit 1; }
  echo "x86_64 tests OK"
else
  echo "== x86_64 cross-check skipped (rustup toolchain or x86_64 std missing) =="
fi
