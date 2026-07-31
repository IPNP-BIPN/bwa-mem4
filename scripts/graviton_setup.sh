#!/usr/bin/env bash
# Provision an AWS Graviton4 box as a GitHub Actions self-hosted runner for bench-arm.yml, and
# stage the data needed to reproduce @nh13's benchmark regime exactly.
#
# THIS SCRIPT RUNS ON THE EC2 INSTANCE, not on your laptop. It launches nothing and spends nothing
# by itself; you create the instance, then run this on it.
#
#   sudo dnf -y install git && git clone https://github.com/IPNP-BIPN/bwa-mem4 && cd bwa-mem4
#   RUNNER_TOKEN=<from GitHub> scripts/graviton_setup.sh
#
# ---------------------------------------------------------------------------------------------
# WHY THIS EXISTS
#
# The remaining first-order lever is `madvise(MADV_HUGEPAGE)` on the index arrays
# (crates/bwa-index/src/hugepage.rs), and it is structurally invisible on the macOS dev host:
# arm64 macOS uses a 16 KiB base page, Linux 4 KiB, and there is no THP equivalent on Darwin. The
# same is true of every other page-table effect. Nothing about the Graviton deficit Nils measured
# can be confirmed or refuted without a Linux arm64 box at genome scale.
#
# `gh api repos/IPNP-BIPN/bwa-mem4/actions/runners` returns total_count 0, so bench-arm.yml's
# `graviton-selfhosted` job (gated on run_graviton=true) currently queues forever.
#
# ---------------------------------------------------------------------------------------------
# THE INSTANCE (create this first, in the console or with `aws ec2 run-instances`)
#
#   type    m8g.4xlarge     Graviton4 / Neoverse-V2, 16 vCPU, 64 GiB
#                           This is Nils's exact instance type. Do NOT substitute c8g.4xlarge:
#                           it has 32 GiB, and the GATK index on tmpfs (~21 GB, which counts
#                           against RAM) plus the aligner's working set does not fit. Nils hit
#                           exactly this and had to resize 30 -> 61 GB mid-campaign, discarding
#                           every measurement taken before.
#   region  us-east-1       (his; only matters for download latency)
#   AMI     Amazon Linux 2023, arm64
#   disk    gp3, >= 250 GiB The GATK FASTA is ~3 GB, its index ~21 GB, the CRAM/BAM sources are
#                           tens of GB, plus a cargo target dir and the extracted FASTQs.
#   cost    ~$0.73/h on-demand. Stop the instance between sessions; it does not need to be up.
#
# ---------------------------------------------------------------------------------------------
# WHAT THIS SCRIPT DOES
#
#   1. installs build tooling, Rust 1.96.1, and bioconda's bwa-mem2 / bwa-mem3 / wgsim / samtools
#   2. registers the box as a GitHub Actions runner labelled `self-hosted, linux, ARM64, graviton4`
#   3. optionally stages the GATK GRCh38 reference and builds its index onto tmpfs
#
# Steps are independent and idempotent; re-running skips what is already done. Control with:
#   SKIP_RUNNER=1   do not register the Actions runner (measure by hand instead)
#   SKIP_DATA=1     do not download the reference (it is ~3 GB plus a ~21 GB index build)
#   RUNNER_TOKEN=   registration token from
#                   https://github.com/IPNP-BIPN/bwa-mem4/settings/actions/runners/new
set -uo pipefail

REPO_URL="${REPO_URL:-https://github.com/IPNP-BIPN/bwa-mem4}"
RUNNER_LABELS="${RUNNER_LABELS:-self-hosted,linux,ARM64,graviton4}"
RUNNER_VERSION="${RUNNER_VERSION:-2.330.0}"
RUST_VERSION="${RUST_VERSION:-1.96.1}"
# tmpfs, so the index is served from RAM exactly as in Nils's runs. It counts against the 64 GiB.
SHM="${SHM:-/dev/shm/bwa4}"
WORK="${WORK:-$HOME/bwa4-bench}"

say() { printf '\n=== %s ===\n' "$*"; }
have() { command -v "$1" >/dev/null 2>&1; }

# ---- 0. Sanity: is this actually a Graviton4? ------------------------------------------------
say "host"
echo "  nproc:  $(nproc)"
echo "  memory: $(awk '/MemTotal/{printf "%.1f GiB", $2/1048576}' /proc/meminfo)"
# implementer 0x41 = ARM. part 0xd40 = Neoverse-V1 (Graviton3), 0xd4f = Neoverse-V2 (Graviton4).
part=$(awk '/CPU part/{print $4; exit}' /proc/cpuinfo)
case "$part" in
  0xd4f) echo "  cpu:    Neoverse-V2 (Graviton4) -- matches the benchmark host" ;;
  0xd40) echo "  cpu:    Neoverse-V1 (Graviton3) -- NOT the benchmark host; label it graviton3" ;;
  *)     echo "  cpu:    part $part -- unrecognised, results are not comparable to Nils's table" ;;
esac
# THP state decides whether the madvise hint can do anything at all. `madvise` (the usual default)
# is the RIGHT setting for this test: it means huge pages are granted only where asked, which is
# precisely the A/B we want. `never` would silently null the lever; `always` would hide it by
# giving both arms huge pages.
thp=$(cat /sys/kernel/mm/transparent_hugepage/enabled 2>/dev/null || echo unavailable)
echo "  THP:    $thp"
case "$thp" in
  *"[never]"*)  echo "  WARNING: THP is off. The hugepage lever cannot show up. Enable it with:"
                echo "           echo madvise | sudo tee /sys/kernel/mm/transparent_hugepage/enabled" ;;
  *"[always]"*) echo "  WARNING: THP is 'always'. Both A/B arms get huge pages and the hint measures"
                echo "           nothing. Set it to 'madvise' for the comparison to mean anything." ;;
esac

# ---- 1. Toolchain -----------------------------------------------------------------------------
say "system packages"
sudo dnf -y install git gcc gcc-c++ make cmake perl-core zlib-devel bzip2 tar curl >/dev/null \
  && echo "  ok" || echo "  dnf failed (continuing; they may already be present)"

if ! have cargo; then
  say "rust $RUST_VERSION"
  curl -fsSL https://sh.rustup.rs | sh -s -- -y --default-toolchain "$RUST_VERSION" --profile minimal
fi
# shellcheck disable=SC1091
[ -f "$HOME/.cargo/env" ] && . "$HOME/.cargo/env"
echo "  $(cargo --version 2>/dev/null || echo 'cargo MISSING')"

if ! have micromamba; then
  say "micromamba + bioconda (bwa-mem2, bwa-mem3, wgsim, samtools)"
  ( cd "$HOME" && curl -fsSL https://micro.mamba.pm/api/micromamba/linux-aarch64/latest | tar -xvj bin/micromamba )
  export PATH="$HOME/bin:$PATH"
  echo 'export PATH="$HOME/bin:$PATH"' >> "$HOME/.bashrc"
fi
export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-$HOME/micromamba}"
if ! have bwa-mem2; then
  micromamba create -y -n bench -c conda-forge -c bioconda bwa-mem2 bwa-mem3 wgsim samtools >/dev/null 2>&1 \
    && echo "  bench env created" || echo "  micromamba create FAILED (bwa-mem3 may not have an aarch64 build)"
  echo 'micromamba activate bench' >> "$HOME/.bashrc"
fi

# ---- 2. GitHub Actions runner ------------------------------------------------------------------
# Registered with an explicit `graviton4` label on top of the three bench-arm.yml requires, so that
# adding a Graviton3 box later cannot silently answer a Graviton4 question.
if [ "${SKIP_RUNNER:-0}" != "1" ]; then
  say "github actions runner"
  if [ -d "$HOME/actions-runner/.runner" ]; then
    echo "  already configured"
  elif [ -z "${RUNNER_TOKEN:-}" ]; then
    echo "  SKIPPED: set RUNNER_TOKEN=<token> to register."
    echo "  Get one (valid 1 hour) at:"
    echo "    $REPO_URL/settings/actions/runners/new"
  else
    mkdir -p "$HOME/actions-runner" && cd "$HOME/actions-runner" || exit 1
    curl -fsSL -o runner.tar.gz \
      "https://github.com/actions/runner/releases/download/v${RUNNER_VERSION}/actions-runner-linux-arm64-${RUNNER_VERSION}.tar.gz"
    tar xzf runner.tar.gz && rm -f runner.tar.gz
    ./config.sh --unattended --url "$REPO_URL" --token "$RUNNER_TOKEN" \
                --labels "$RUNNER_LABELS" --name "graviton4-$(hostname -s)"
    sudo ./svc.sh install "$USER" && sudo ./svc.sh start
    echo "  registered and running as a service"
  fi
  cd - >/dev/null || true
fi

# ---- 3. Reference and index, on tmpfs -----------------------------------------------------------
# The GATK resource-bundle GRCh38 (decoys + ALTs), which is what Nils used. It is NOT the same
# reference as work/genome.fa in this repo (Ensembl primary assembly, 194 contigs, no ALTs), so its
# index is substantially larger and the two sets of numbers are not interchangeable.
if [ "${SKIP_DATA:-0}" != "1" ]; then
  say "GATK GRCh38 reference (Nils's exact reference)"
  mkdir -p "$WORK" "$SHM"
  REF="$SHM/Homo_sapiens_assembly38.fasta"
  if [ ! -s "$REF" ]; then
    echo "  downloading ~3 GB to tmpfs ..."
    curl -fsSL --retry 3 -o "$REF" \
      "https://storage.googleapis.com/gcp-public-data--broad-references/hg38/v0/Homo_sapiens_assembly38.fasta"
  fi
  ls -la "$REF"
  if [ ! -s "$REF.bwt.2bit.64" ]; then
    say "index build (bwa-mem2, so the index is the oracle's own bytes)"
    echo "  expect ~20-40 min and a large peak RSS; the index lands on tmpfs and stays resident."
    /usr/bin/time -v bwa-mem2 index "$REF" 2>&1 | tail -5
  fi
  df -h /dev/shm | tail -1
fi

# ---- 4. Build our binary ------------------------------------------------------------------------
say "build bwa-mem4"
cd "$(dirname "$0")/.." || exit 1
# RUSTFLAGS empty on purpose: .cargo/config.toml sets `-C target-cpu=native` for local dev, and the
# benchmark must time the codegen we actually ship, as both bench workflows already do.
RUSTFLAGS="" cargo build --release -p bwa-mem4 2>&1 | tail -2

say "next"
cat <<'NEXT'
  Reads. Nils's two real sets, deterministic every-Nth-pair down to 5M pairs:
    wgs  https://ftp.sra.ebi.ac.uk/vol1/run/ERR324/ERR3240114/HG00096.final.cram
    wes  https://ftp.1000genomes.ebi.ac.uk/vol1/ftp/phase3/data/HG00100/exome_alignment/HG00100.mapped.ILLUMINA.bwa.GBR.exome.20121211.bam
  Convert with `samtools collate` then `samtools fastq`, gzip both mates, then subsample.

  Then, in order of what actually needs answering:

  1. The hugepage A/B. One binary, both arms, so nothing but the hint differs:
       BWA4_NO_HUGEPAGE=1 ./target/release/bwa-mem4 mem -t16 $REF r1.fq.gz r2.fq.gz >/dev/null
                          ./target/release/bwa-mem4 mem -t16 $REF r1.fq.gz r2.fq.gz >/dev/null
     Confirm mechanistically before believing any wall delta:
       grep AnonHugePages /proc/<pid>/smaps_rollup
       perf stat -e dTLB-load-misses,LLC-load-misses -- ...

  2. Where the wall goes, which no measurement on the Mac could show at this scale:
       BWA4_STAGE_TIME=1 ./target/release/bwa-mem4 mem -t16 -K 10000000 $REF r1.fq.gz r2.fq.gz >/dev/null

  3. Head-to-head, Nils's exact regime (default -K, gzipped in, /dev/null out):
       T=16 K=160000000 IDX=$REF READS=r1.fq.gz READS2=r2.fq.gz scripts/fork_bench.sh pe 3
     Use the PGO binary for our arm (scripts/pgo.sh); a plain release build is ~15% slower and is
     not what we ship. Note PGO measured -0.4% on Graviton4 for Nils, so report both.

  4. Re-sweep the rescue chunk size on REAL reads, where mate rescue is 47-64% of PE compute
     instead of the ~10% wgsim shows:
       for P in 512 1024 2048 4096 8192 16384; do
         BWA4_RESCUE_PAIRS_PER_CHUNK=$P BWA4_STAGE_TIME=1 ./target/release/bwa-mem4 mem -t16 ...
       done

  Method rules that are not optional (each was paid for by a wrong result in this repo):
    interleave arms within a rep; warm every binary first; genome index only; print the batch
    count; quote -t with every ratio; nothing under 3% is a gain.
NEXT
