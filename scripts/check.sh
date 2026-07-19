#!/usr/bin/env bash
# Local CI gate: format, lint, test. Mirrors .github/workflows/ci.yml.
#
# It does NOT cover .github/workflows/parity.yml, which is the check that actually protects
# byte-identity. Run that one too before proposing a change to the aligner:
#
#   python3 scripts/make_test_reads.py testdata/tiny/tiny.fa /tmp/ci --n 8000
#   IDX=testdata/tiny/tiny.fa R1=/tmp/ci_1.fq R2=/tmp/ci_2.fq \
#     bash scripts/opt_parity.sh ./target/release/bwa-mem3
#
# and remember `cargo test` does not relink target/release/bwa-mem3: `cargo build --release` first.
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
