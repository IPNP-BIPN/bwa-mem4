#!/usr/bin/env bash
# Metal Shading Language probe: issue #55's step 0, answered WITHOUT Xcode.
#
# The issue assumed the offline `metal` compiler, which ships with Xcode and is absent on a machine
# with only the Command Line Tools (`xcrun -sdk macosx metal` -> "not a developer tool"). It is not
# needed: MSL compiles at RUN TIME through `newLibraryWithSource:`, and Metal.framework is part of
# macOS. So the only thing required is clang, which the Command Line Tools do provide.
#
# Two probes:
#   features.m    does MSL expose saturating integer arithmetic, and under what name?
#   dp_ceiling.m  what does the rescue recurrence actually sustain on this GPU?
#
# The second is a CEILING, not a throughput: registers only, no memory traffic in the loop, the same
# discipline the NEON ceiling measurement used (see the "plafond du noyau de rescue" section of
# ROADMAP.md). A real kernel will be under it.
set -euo pipefail
cd "$(dirname "$0")/.."
out="${TMPDIR:-/tmp}/bwa4-msl"
mkdir -p "$out"
CFLAGS=(-fobjc-arc -framework Foundation -framework Metal -O2)
for probe in features dp_ceiling; do
  echo "== $probe =="
  clang "${CFLAGS[@]}" -o "$out/$probe" "scripts/msl/$probe.m"
  "$out/$probe"
  echo
done
