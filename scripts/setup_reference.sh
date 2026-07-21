#!/usr/bin/env bash
# Recreate reference/ : the exact patched bwa-mem2 source the installed oracle was built from,
# plus the fg-labs/bwa-mem3 C++ fork for study. reference/ is gitignored.
#
# Oracle = Homebrew bwa-mem2 2.3 = upstream tag v2.3 (rev 7aa5ff6c, source 2.2.1) + two patches
# (fastmap.patch, bandedSWA.cpp.patch) + sse2neon. See /opt/homebrew/.../bwa-mem2.rb.
set -euo pipefail
cd "$(dirname "$0")/.."
REF=reference
REV=7aa5ff6c3330490e5629ab9b7327683d2dce02d6
FASTMAP_URL="https://gist.githubusercontent.com/YoshitakaMo/eb6e6df7a621a9c9737bcc9363cf9bfc/raw/5936f4884daac3c961c6c9a62d9a0c676f578bbb/fastmap.patch"
FASTMAP_SHA=cbba705412b8a1139be752759490606a988eee0483a3c0ee65aa0a03c1c9c9e8
BANDED_URL="https://gist.githubusercontent.com/YoshitakaMo/c4cabc8e1e4b618047507bc354dbb51e/raw/1265ecf70a976476bd3e55d06804b94f9969310e/bandedSWA.cpp.patch"
BANDED_SHA=cdc13b153a23beb890d258eeb41d13aa0b777c1747bdefa49c399634c176cda7

mkdir -p "$REF"
if [ ! -d "$REF/bwa-mem2/.git" ]; then
  git clone --quiet https://github.com/bwa-mem2/bwa-mem2 "$REF/bwa-mem2"
fi
git -C "$REF/bwa-mem2" checkout --quiet "$REV"
git -C "$REF/bwa-mem2" restore src/fastmap.cpp src/bandedSWA.cpp

curl -fsSL -o "$REF/fastmap.patch" "$FASTMAP_URL"
curl -fsSL -o "$REF/bandedSWA.cpp.patch" "$BANDED_URL"
echo "$FASTMAP_SHA  $REF/fastmap.patch" | shasum -a 256 -c -
echo "$BANDED_SHA  $REF/bandedSWA.cpp.patch" | shasum -a 256 -c -

# dos2unix substitute (macOS has no dos2unix by default), then apply.
perl -i -pe 's/\r\n/\n/g' "$REF/bwa-mem2/src/fastmap.cpp" "$REF/bwa-mem2/src/bandedSWA.cpp"
patch -p1 "$REF/bwa-mem2/src/fastmap.cpp" "$REF/fastmap.patch"
patch -p1 "$REF/bwa-mem2/src/bandedSWA.cpp" "$REF/bandedSWA.cpp.patch"

if [ ! -d "$REF/bwa-mem4-cpp/.git" ]; then
  git clone --quiet https://github.com/fg-labs/bwa-mem3 "$REF/bwa-mem4-cpp"
fi
echo "reference/ ready (bwa-mem2 @ $REV + patches, fg-labs/bwa-mem3)"
