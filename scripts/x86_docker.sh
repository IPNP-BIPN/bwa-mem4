#!/usr/bin/env bash
# x86_64 harness in a container, for a developer working on Apple Silicon.
#
# WHAT THIS IS FOR, AND WHAT IT IS NOT FOR.
#
# It answers questions about the x86_64 code paths that do not depend on how fast the machine runs
# them: does the workspace build for x86_64-unknown-linux-gnu, do the AVX2 kernels produce the same
# SAM bytes as the x86_64 oracle, how does memory grow with `-t`. Docker Desktop on Apple Silicon
# runs amd64 images under Rosetta, which exposes `avx2` (so those kernels really execute) but not
# `avx512bw` (so those do not).
#
# It does NOT answer "how fast is bwa-mem4 on x86_64". Both binaries run under emulation, and
# emulation does not preserve ratios: it taxes instruction mixes unequally, which is precisely the
# variable under test when comparing one vectorised kernel against another. Issues #20, #27, #32 and
# #33 need a native x86_64 machine. Use this to arrive there with the build and the parity already
# settled, not to replace the measurement.
#
# Usage:
#   scripts/x86_docker.sh build     # image + release build for x86_64 (cached in a docker volume)
#   scripts/x86_docker.sh parity    # opt_parity.sh against the x86_64 bwa-mem2 oracle
#   scripts/x86_docker.sh rss       # peak-RSS vs -t, index contribution removed
#   scripts/x86_docker.sh shell     # interactive container with both binaries on PATH
set -euo pipefail
cd "$(dirname "$0")/.."
REPO="$PWD"

IMAGE=bwamem4-x86
VOL=bwamem4-x86-build
# 2.2.1 is the source release the oracle's 2.3 tag was cut from, and the image ships the ISA-specific
# binaries plus the dispatcher, so `bwa-mem2` selects `.avx2` by itself under Rosetta.
ORACLE_IMAGE=quay.io/biocontainers/bwa-mem2:2.2.1--he513fc3_0
# `-K` fixed so the batch boundaries, and therefore the output, do not depend on thread count.
CHUNK=10000000

need_docker() {
  command -v docker >/dev/null || { echo "docker not found"; exit 1; }
}

# The oracle binaries are copied out of the biocontainer once and cached next to the repo, because
# mounting two images into one container is not possible and rebuilding bwa-mem2 from source inside
# the container needs network access that a restricted environment may not have.
oracle_dir() { echo "${TMPDIR:-/tmp}/bwamem4-x86-oracle"; }

fetch_oracle() {
  local d; d="$(oracle_dir)"
  [ -x "$d/bwa-mem2.avx2" ] && return 0
  mkdir -p "$d"
  docker pull --platform linux/amd64 -q "$ORACLE_IMAGE" >/dev/null
  local c; c=$(docker create --platform linux/amd64 "$ORACLE_IMAGE" sh)
  docker cp "$c:/usr/local/bin/bwa-mem2" "$d/bwa-mem2"
  docker cp "$c:/usr/local/bin/bwa-mem2.avx2" "$d/bwa-mem2.avx2"
  docker rm "$c" >/dev/null
  chmod +x "$d"/bwa-mem2*
}

# Every run mounts: the repo read-write at /work, a named volume at /build for the target directory
# (writing a multi-gigabyte target/ through a macOS bind mount is dominated by filesystem overhead),
# and the cached oracle at /oracle.
in_container() {
  need_docker
  docker run --rm --platform linux/amd64 \
    -v "$REPO:/work" -v "$VOL:/build" -v "$(oracle_dir):/oracle" \
    -e CARGO_TARGET_DIR=/build/target \
    "$@"
}

cmd_build() {
  need_docker
  docker build --platform linux/amd64 -t "$IMAGE" -f scripts/x86_docker.Dockerfile scripts/
  docker volume create "$VOL" >/dev/null
  # x86-64-v3 is the AVX2 baseline, matching what the release workflow ships for x86_64.
  in_container "$IMAGE" bash -c '
    cd /work && RUSTFLAGS="-C target-cpu=x86-64-v3" cargo build --release
    /build/target/release/bwa-mem4 --version'
}

# Fixture shared by both measurements: the committed tiny index plus reads generated from it. Kept
# inside the volume so it survives between runs.
cmd_fixture() {
  in_container "$IMAGE" bash -c "
    set -e
    mkdir -p /build/pt && cd /work
    cp -n testdata/tiny/tiny.fa* /build/pt/ 2>/dev/null || true
    [ -f /build/pt/ci_1.fq ]  || python3 scripts/make_test_reads.py testdata/tiny/tiny.fa /build/pt/ci  --n 8000
    [ -f /build/pt/big_1.fq ] || python3 scripts/make_test_reads.py testdata/tiny/tiny.fa /build/pt/big --n 400000
    ls -la /build/pt | tail -6"
}

cmd_parity() {
  fetch_oracle; cmd_fixture
  in_container "$IMAGE" bash -c '
    cp /oracle/bwa-mem2* /usr/local/bin/ && chmod +x /usr/local/bin/bwa-mem2*
    cd /work
    IDX=/build/pt/tiny.fa R1=/build/pt/ci_1.fq R2=/build/pt/ci_2.fq M2=bwa-mem2 \
      bash scripts/opt_parity.sh /build/target/release/bwa-mem4'
}

# Peak RSS against thread count on a 200 kb reference. The small index is the point: it removes the
# index contribution from the peak and leaves the per-batch allocation, which is what #25 is about.
cmd_rss() {
  fetch_oracle; cmd_fixture
  in_container "$IMAGE" bash -c "
    cp /oracle/bwa-mem2* /usr/local/bin/ && chmod +x /usr/local/bin/bwa-mem2*
    cd /build/pt
    printf '%-10s %-8s %-12s %-10s\n' tool threads maxRSS_MB wall
    for t in 1 2 4 8 16; do
      for tool in bwa-mem2 /build/target/release/bwa-mem4; do
        out=\$( { /usr/bin/time -v \$tool mem -t \$t -K $CHUNK tiny.fa big_1.fq big_2.fq > /dev/null; } 2>&1 )
        rss=\$(echo \"\$out\" | grep 'Maximum resident' | grep -oE '[0-9]+')
        wall=\$(echo \"\$out\" | grep 'Elapsed (wall' | awk '{print \$NF}')
        printf '%-10s %-8s %-12s %-10s\n' \"\$(basename \$tool)\" \"\$t\" \"\$((rss/1024))\" \"\$wall\"
      done
    done
    echo
    echo 'Reminder: these wall times are emulated and are NOT a speed measurement.'"
}

cmd_shell() {
  fetch_oracle
  in_container -it "$IMAGE" bash -c '
    cp /oracle/bwa-mem2* /usr/local/bin/ 2>/dev/null && chmod +x /usr/local/bin/bwa-mem2*
    cd /work && exec bash'
}

case "${1:-}" in
  build)  cmd_build ;;
  parity) cmd_parity ;;
  rss)    cmd_rss ;;
  shell)  cmd_shell ;;
  *) sed -n '2,20p' "$0"; exit 1 ;;
esac
