#!/usr/bin/env bash
# Run the AVX-512 byte-identity tests under Intel SDE, the way `.github/workflows/avx512-check.yml`
# does, on a machine that does not have AVX-512.
#
# WHY THIS EXISTS. The three tests (`avx512_matesw_u8_matches_scalar`,
# `avx512_matesw_i16_matches_scalar`, `avx512_u8_and_i16_match_scalar`) self-skip when
# `is_x86_feature_detected!("avx512bw")` is false, which is every machine most contributors have and
# every GitHub-hosted runner this project drew for months. They therefore reported `ok` for a long
# time without executing a single instruction of the kernels they name. SDE is a functional emulator
# from the people who define the ISA: it presents CPUID and XCR0 faithfully, so Rust's own detection
# selects the kernels and the tests run for real.
#
# WHAT IT CANNOT DO: price anything. SDE is roughly two orders of magnitude slower than hardware and
# has none of the port behaviour the kernel tuning is about. Speed claims need a real AVX-512 host.
#
# REQUIREMENTS: x86_64 Linux (SDE ships Linux and Windows builds; there is no macOS or aarch64 one),
# curl, tar, and permission to relax `kernel.yama.ptrace_scope`, which Pin needs to attach.
set -euo pipefail
cd "$(dirname "$0")/.."

# Pinned by URL and by content. An emulator that changed under us would turn a green run back into
# the thing this script exists to prevent: a claim with nothing behind it. Bump deliberately.
SDE_URL="https://downloadmirror.intel.com/843185/sde-external-9.48.0-2024-11-25-lin.tar.xz"
SDE_SHA256="3173d2a5369e3385226b488d8b75403951bc14af601435fe707d9f83e0b533e6"
SDE_DIR_NAME="sde-external-9.48.0-2024-11-25-lin"
# Two chips, and the pair is the point. `-skx` is the ISA FLOOR: the runtime dispatch selects these
# kernels on `avx512bw` alone, and Skylake-X has `avx512bw` without VBMI or VNNI, so an instruction
# from a newer set inside a BW-gated block faults here instead of in a user's alignment. `-spr` is
# the part the kernels were tuned against.
CHIPS="${BWA4_SDE_CHIPS:-skx spr}"
WORK="${BWA4_SDE_DIR:-${TMPDIR:-/tmp}/bwa4-sde}"

case "$(uname -s)/$(uname -m)" in
  Linux/x86_64) ;;
  *)
    echo "this script needs x86_64 Linux; Intel SDE ships no build for $(uname -s)/$(uname -m)" >&2
    echo "on another host, let CI do it: .github/workflows/avx512-check.yml runs on every PR" >&2
    exit 2
    ;;
esac

mkdir -p "$WORK"
if [ ! -x "$WORK/$SDE_DIR_NAME/sde64" ]; then
  echo "== fetching Intel SDE =="
  curl -fsSL -o "$WORK/sde.tar.xz" "$SDE_URL"
  echo "$SDE_SHA256  $WORK/sde.tar.xz" | sha256sum -c -
  tar xf "$WORK/sde.tar.xz" -C "$WORK"
fi
SDE="$WORK/$SDE_DIR_NAME/sde64"
"$SDE" -version | head -2

# Pin injects itself with ptrace, which Ubuntu's Yama LSM restricts to descendants by default.
if [ "$(cat /proc/sys/kernel/yama/ptrace_scope 2>/dev/null || echo 0)" != "0" ]; then
  echo "== relaxing kernel.yama.ptrace_scope (Pin cannot attach otherwise) =="
  sudo sysctl -w kernel.yama.ptrace_scope=0
fi

echo "== building the test binary =="
# No `-C target-cpu`: the kernels are chosen at RUNTIME, and a target-cpu promising AVX-512 to the
# whole binary would let the compiler emit those instructions in code that is NOT feature-gated,
# which would fault on a real host instead of falling back. The point is to exercise the dispatch.
BIN=$(RUSTFLAGS="" cargo test -p bwa-mem4-neon --release --no-run --message-format=json \
      | python3 -c 'import json,sys
for line in sys.stdin:
    try: m = json.loads(line)
    except ValueError: continue
    if m.get("executable") and m.get("target", {}).get("test"):
        print(m["executable"])' | head -1)
[ -n "$BIN" ] || { echo "cargo built no test binary" >&2; exit 1; }
echo "binary: $BIN"

status=0
for chip in $CHIPS; do
  echo
  echo "== $chip =="
  out="$WORK/out-$chip.txt"
  # SDE faults on any instruction the selected chip does not have, so this also proves the kernels
  # stay inside that model's ISA and not merely inside avx512bw.
  if ! "$SDE" "-$chip" -- "$BIN" avx512 --nocapture --test-threads 1 > "$out" 2>&1; then
    echo "FAILED under -$chip:"; tail -30 "$out"; status=1; continue
  fi
  # A skip reported as `ok` is the failure mode this whole thing exists to break.
  if grep -q 'skipping avx512' "$out"; then
    echo "an AVX-512 test SKIPPED under -$chip: nothing was proved"; tail -10 "$out"; status=1; continue
  fi
  ran=$(grep -c -E '^test .*avx512.* \.\.\. ok$' "$out" || true)
  echo "avx512 tests executed under -$chip: $ran"
  if [ "$ran" -lt 3 ]; then
    echo "expected 3 (matesw u8, matesw i16, batched u8+i16), got $ran"; status=1; continue
  fi
  grep -E 'test result' "$out" || true
done

exit "$status"
