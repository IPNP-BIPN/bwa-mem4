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
set -euo pipefail
cd "$(dirname "$0")/.."

echo "cargo: $(cargo --version)"
echo "== rustfmt =="
cargo fmt --check
echo "== clippy (-D warnings) =="
cargo clippy --workspace --all-targets -- -D warnings
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
else
  echo "== x86_64 cross-check skipped (rustup toolchain or x86_64 std missing) =="
fi
