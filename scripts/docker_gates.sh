#!/usr/bin/env bash
# Every gate this project has, run inside containers, from a developer machine that is Apple Silicon.
#
# TWO PLATFORMS, AND THE DIFFERENCE MATTERS.
#
#   ARCH=amd64 (default) — Docker Desktop runs the image under Rosetta. `avx2` is exposed, so those
#     kernels really execute; `avx512bw` is not, so those never do. Correctness, parity and memory
#     are meaningful here. WALL TIME IS NOT: emulation taxes instruction mixes unequally, which is
#     exactly the variable under test when comparing one vectorised kernel to another. Issues #20,
#     #27, #32 and #33 need native x86_64 hardware and this harness cannot stand in for it.
#
#   ARCH=arm64 — the image runs NATIVELY on Apple Silicon. Timings here are real, and comparable to
#     a host run, which is what makes `bench` worth having. The AVX2 and AVX-512 paths do not exist
#     on this target, so it is the NEON kernels that are exercised.
#
# Usage:
#   scripts/docker_gates.sh build     # image + release build (target/ cached in a docker volume)
#   scripts/docker_gates.sh check     # fmt + clippy -D warnings + workspace tests, in the container
#   scripts/docker_gates.sh parity    # opt_parity.sh against the platform's bwa-mem2 oracle
#   scripts/docker_gates.sh rss       # peak RSS vs -t, bwa-mem4 against the fork, both -K regimes
#   scripts/docker_gates.sh bench     # wall time; refuses to run under emulation
#   scripts/docker_gates.sh all       # build, check, parity, rss
#   scripts/docker_gates.sh shell     # interactive container, both binaries on PATH
#
#   ARCH=arm64 scripts/docker_gates.sh bench
set -euo pipefail
cd "$(dirname "$0")/.."
REPO="$PWD"

ARCH="${ARCH:-amd64}"
case "$ARCH" in
  amd64|arm64) ;;
  *) echo "ARCH must be amd64 or arm64"; exit 1 ;;
esac
PLATFORM="linux/$ARCH"
IMAGE="bwamem4-gates-$ARCH"
VOL="bwamem4-gates-$ARCH"
# `-C target-cpu`: the AVX2 baseline on x86_64, matching what the release workflow ships. On arm64
# the default target already implies NEON, so nothing is forced.
[ "$ARCH" = amd64 ] && RFLAGS="-C target-cpu=x86-64-v3" || RFLAGS=""
# 2.2.1 is the source release the oracle's 2.3 tag was cut from. The biocontainer ships the
# ISA-specific binaries plus the dispatcher, so `bwa-mem2` picks `.avx2` by itself. x86_64 only:
# there is no arm64 build of it, which is why `parity` on arm64 falls back to the host oracle.
ORACLE_IMAGE=quay.io/biocontainers/bwa-mem2:2.2.1--he513fc3_0
CHUNK=10000000

need_docker() { command -v docker >/dev/null || { echo "docker not found"; exit 1; }; }
oracle_dir() { echo "${TMPDIR:-/tmp}/bwamem4-oracle-$ARCH"; }

# The oracle binaries are copied out of the biocontainer once and cached, because two images cannot
# be mounted into one container and building bwa-mem2 from source needs network access a restricted
# environment may refuse.
fetch_oracle() {
  [ "$ARCH" = amd64 ] || { echo "no bwa-mem2 biocontainer for arm64; use the host oracle"; return 1; }
  local d; d="$(oracle_dir)"
  [ -x "$d/bwa-mem2.avx2" ] && return 0
  mkdir -p "$d"
  docker pull --platform "$PLATFORM" -q "$ORACLE_IMAGE" >/dev/null
  local c; c=$(docker create --platform "$PLATFORM" "$ORACLE_IMAGE" sh)
  docker cp "$c:/usr/local/bin/bwa-mem2" "$d/bwa-mem2"
  docker cp "$c:/usr/local/bin/bwa-mem2.avx2" "$d/bwa-mem2.avx2"
  docker rm "$c" >/dev/null
  chmod +x "$d"/bwa-mem2*
}

# The repo is bind-mounted read-write at /work; target/ lives in a named volume because writing
# gigabytes through a macOS bind mount is dominated by filesystem overhead.
run() {
  need_docker
  mkdir -p "$(oracle_dir)"
  docker run --rm --platform "$PLATFORM" \
    -v "$REPO:/work" -v "$VOL:/build" -v "$(oracle_dir):/oracle" \
    -e CARGO_TARGET_DIR=/build/target \
    "$@"
}

cmd_build() {
  need_docker
  docker build --platform "$PLATFORM" -t "$IMAGE" -f scripts/docker_gates.Dockerfile scripts/
  docker volume create "$VOL" >/dev/null
  run "$IMAGE" bash -c "cd /work && RUSTFLAGS='$RFLAGS' cargo build --release && /build/target/release/bwa-mem4 --version"
}

cmd_check() {
  run "$IMAGE" bash -c '
    cd /work
    cargo fmt --check
    cargo clippy --workspace --all-targets -- -D warnings
    cargo test --workspace 2>&1 | grep -E "^test result" \
      | awk -F"[ ;]" "{p+=\$4; f+=\$7} END {print \"tests: \" p \" passed, \" f \" failed\"}"'
}

# Fixture shared by the measurements: the committed tiny index plus reads generated from it, kept in
# the volume so it survives between runs. 8000 pairs for parity, 400k for the memory curve.
cmd_fixture() {
  run "$IMAGE" bash -c '
    set -e
    mkdir -p /build/pt && cd /work
    cp -n testdata/tiny/tiny.fa* /build/pt/ 2>/dev/null || true
    [ -f /build/pt/ci_1.fq ]  || python3 scripts/make_test_reads.py testdata/tiny/tiny.fa /build/pt/ci  --n 8000
    [ -f /build/pt/big_1.fq ] || python3 scripts/make_test_reads.py testdata/tiny/tiny.fa /build/pt/big --n 400000'
}

# fg-labs/bwa-mem3, the C++ fork this project is benchmarked against. Built once into the volume.
cmd_fork() {
  run "$IMAGE" bash -c '
    set -e
    [ -x /build/fork/bwa-mem3 ] && { echo "fork already built"; exit 0; }
    [ -d /build/fork/.git ] || git clone --quiet --depth 1 https://github.com/fg-labs/bwa-mem3 /build/fork
    cd /build/fork && git submodule update --init --recursive --depth 1 --quiet
    make -j"$(nproc)" >/dev/null 2>&1
    ls -la bwa-mem3'
}

cmd_parity() {
  fetch_oracle; cmd_fixture
  run "$IMAGE" bash -c '
    cp /oracle/bwa-mem2* /usr/local/bin/ && chmod +x /usr/local/bin/bwa-mem2*
    cd /work
    IDX=/build/pt/tiny.fa R1=/build/pt/ci_1.fq R2=/build/pt/ci_2.fq M2=bwa-mem2 \
      bash scripts/opt_parity.sh /build/target/release/bwa-mem4'
}

# Peak RSS against thread count on a 200 kb reference. The tiny index is the point: it contributes
# nothing to the peak, so what is left is the per-batch allocation, which is what #25 is about.
# Both -K regimes are reported because the distinction is the substance of that issue: the default
# batch is 10M x threads and therefore grows with -t, while a fixed -K isolates the per-thread state.
cmd_rss() {
  cmd_fixture; cmd_fork
  run "$IMAGE" bash -c "
    cd /build/pt
    printf '%-9s %-4s %-10s %-10s %-8s\n' regime -t mem4_MB fork_MB ratio
    for t in 1 4 8 16; do
      for regime in defaultK fixedK; do
        [ \$regime = fixedK ] && K='-K $CHUNK' || K=''
        a=\$( { /usr/bin/time -v /build/target/release/bwa-mem4 mem -t \$t \$K tiny.fa big_1.fq big_2.fq >/dev/null; } 2>&1 | grep 'Maximum resident' | grep -oE '[0-9]+')
        b=\$( { /usr/bin/time -v /build/fork/bwa-mem3      mem -t \$t \$K tiny.fa big_1.fq big_2.fq >/dev/null; } 2>&1 | grep 'Maximum resident' | grep -oE '[0-9]+')
        printf '%-9s %-4s %-10s %-10s %-8s\n' \"\$regime\" \"\$t\" \"\$((a/1024))\" \"\$((b/1024))\" \"\$(echo \"scale=2; \$a/\$b\" | bc)\"
      done
    done"
}

# Wall time. Refuses to run on amd64, on purpose: under emulation the number would look like a
# measurement and would not be one. See the header.
cmd_bench() {
  if [ "$ARCH" = amd64 ]; then
    echo "REFUSED: amd64 images run under emulation here, so wall time is not a measurement."
    echo "Use ARCH=arm64 for a native container run, or a native x86_64 machine for #20/#27/#32/#33."
    exit 1
  fi
  cmd_fixture; cmd_fork
  run "$IMAGE" bash -c "
    cd /build/pt
    printf '%-10s %-4s %-8s\n' tool -t wall_s
    for t in 1 4 8 16; do
      for tool in /build/fork/bwa-mem3 /build/target/release/bwa-mem4; do
        s=\$( { /usr/bin/time -f '%e' \$tool mem -t \$t -K $CHUNK tiny.fa big_1.fq big_2.fq >/dev/null; } 2>&1 | tail -1 )
        printf '%-10s %-4s %-8s\n' \"\$(basename \$tool)\" \"\$t\" \"\$s\"
      done
    done"
}

cmd_shell() {
  fetch_oracle || true
  run -it "$IMAGE" bash -c '
    cp /oracle/bwa-mem2* /usr/local/bin/ 2>/dev/null && chmod +x /usr/local/bin/bwa-mem2* 2>/dev/null
    cd /work && exec bash'
}

case "${1:-}" in
  build)  cmd_build ;;
  check)  cmd_check ;;
  parity) cmd_parity ;;
  fork)   cmd_fork ;;
  rss)    cmd_rss ;;
  bench)  cmd_bench ;;
  shell)  cmd_shell ;;
  all)    cmd_build; cmd_check; cmd_parity; cmd_rss ;;
  *) sed -n '2,28p' "$0"; exit 1 ;;
esac
