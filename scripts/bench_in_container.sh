#!/usr/bin/env bash
# Inner half of `scripts/docker_gates.sh bench`: this is what actually runs inside the container.
#
# It is a separate file rather than a here-string because the loop needs quoting that does not
# survive being nested inside a `docker run bash -c '...'`, and a benchmark whose shell quoting is
# subtly wrong is worse than no benchmark.
#
# Environment (all set by the caller):
#   BENCH_REF, BENCH_R1, BENCH_R2   fixture, container paths
#   BENCH_REPS                      repetitions per (tool, thread count)
#   BENCH_THREADS                   space-separated -t values
#   BENCH_CHUNK                     -K, fixed so batch boundaries do not depend on -t
#   BENCH_LABEL                     one line describing the platform, printed verbatim
#   BENCH_TOOLS                     space-separated binaries to compare, in run order
set -uo pipefail

echo "platform: ${BENCH_LABEL}"
echo "fixture:  ${BENCH_REF}"
echo "reps:     ${BENCH_REPS} (tools alternated within each rep), median reported"
echo
printf '%-11s %-4s %-10s %-10s %-10s\n' tool -t median_s min_s max_s

for t in ${BENCH_THREADS}; do
  for tool in ${BENCH_TOOLS}; do
    times=""
    # The tools alternate inside the repetition loop rather than each running its reps back to back,
    # so page-cache state and any thermal drift land on both arms equally.
    for _ in $(seq 1 "${BENCH_REPS}"); do
      s=$( { /usr/bin/time -f '%e' "$tool" mem -t "$t" -K "${BENCH_CHUNK}" \
               "${BENCH_REF}" "${BENCH_R1}" "${BENCH_R2}" >/dev/null; } 2>&1 | tail -1 )
      times="$times $s"
    done
    sorted=$(printf '%s\n' $times | sort -n)
    med=$(printf '%s\n' "$sorted" | awk '{a[NR]=$1} END {print (NR%2)?a[(NR+1)/2]:(a[NR/2]+a[NR/2+1])/2}')
    printf '%-11s %-4s %-10s %-10s %-10s\n' \
      "$(basename "$tool")" "$t" "$med" "$(printf '%s\n' "$sorted" | head -1)" "$(printf '%s\n' "$sorted" | tail -1)"
  done
done
