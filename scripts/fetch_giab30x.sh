#!/usr/bin/env bash
# Fetch ~30x of GIAB HG002 300x HiSeq (NIST), concatenated gzip, pairing preserved.
# Downloads R1/R2 chunk pairs in a fixed order, appending to combined .gz files, until the
# accumulated R1 gzip size crosses TARGET_R1_BYTES (~35 GB ≈ 320M pairs ≈ 30x).
set -uo pipefail
cd "$(dirname "$0")/.."

BASE="https://ftp-trace.ncbi.nlm.nih.gov/ReferenceSamples/giab/data/AshkenazimTrio/HG002_NA24385_son/NIST_HiSeq_HG002_Homogeneity-10953946/HG002_HiSeq300x_fastq"
FLOWCELLS="140528_D00360_0018_AH8VC6ADXX 140528_D00360_0019_BH8VDAADXX 140605_D00360_0020_AH9V1RADXX"
PROJ="Project_RM8391_RM8392"
OUT=work/giab30x
mkdir -p "$OUT"
R1OUT="$OUT/HG002_30x_R1.fastq.gz"
R2OUT="$OUT/HG002_30x_R2.fastq.gz"
TARGET_R1_BYTES=$((35 * 1024 * 1024 * 1024))
: > "$R1OUT"; : > "$R2OUT"

echo "[discover] listing R1 chunks..."
: > "$OUT/r1_urls.txt"
for FC in $FLOWCELLS; do
  P="$BASE/$FC/$PROJ"
  for S in $(curl -s --max-time 60 --retry 3 "$P/" | grep -oE 'href="Sample_[^"/]+/"' | sed 's/href="//;s/"//'); do
    curl -s --max-time 60 --retry 3 "$P/$S" \
      | grep -oE 'href="[^"]+_R1_[^"]+\.fastq\.gz"' | sed 's/href="//;s/"//' \
      | while read -r f; do echo "$P/$S$f"; done >> "$OUT/r1_urls.txt"
  done
done
sort -o "$OUT/r1_urls.txt" "$OUT/r1_urls.txt"
echo "[discover] $(wc -l < "$OUT/r1_urls.txt") R1 chunks available"

acc=0; n=0
while read -r r1url; do
  r2url="${r1url/_R1_/_R2_}"
  n=$((n+1))
  echo "[dl $n] $(basename "$r1url")"
  curl -s --max-time 1800 --retry 4 --retry-delay 5 "$r1url" >> "$R1OUT" || { echo "  R1 fail"; break; }
  curl -s --max-time 1800 --retry 4 --retry-delay 5 "$r2url" >> "$R2OUT" || { echo "  R2 fail"; break; }
  acc=$(stat -f%z "$R1OUT")
  echo "  R1 total: $((acc/1024/1024)) MB / $((TARGET_R1_BYTES/1024/1024)) MB target"
  if [ "$acc" -ge "$TARGET_R1_BYTES" ]; then echo "[done] target reached at $n chunks"; break; fi
done < "$OUT/r1_urls.txt"

echo "FETCH_DONE R1=$((acc/1024/1024))MB chunks=$n"
echo "R1=$R1OUT"
echo "R2=$R2OUT"
