# Build environment for the x86_64 harness driven by scripts/x86_docker.sh.
#
# Pinned to the workspace's rust-version so the container and CI agree on the compiler. On Apple
# Silicon this runs under Rosetta via Docker Desktop: `avx2` is exposed, `avx512bw` is not, so the
# AVX-512 kernels compile here but never execute.
FROM --platform=linux/amd64 rust:1.96-bookworm

# rust-htslib builds htslib from source, so the C toolchain and the compression / transport
# libraries it links must be present; clang and libclang are what bindgen needs. samtools and tabix
# are for the BGZF and CRAM cases of opt_parity.sh, which skip themselves when absent.
RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential clang libclang-dev cmake pkg-config git curl ca-certificates \
      zlib1g-dev libbz2-dev liblzma-dev libcurl4-openssl-dev libssl-dev \
      autoconf automake libtool python3 procps time bc samtools tabix \
    && rm -rf /var/lib/apt/lists/*

# The repo arrives as a bind mount from macOS; keeping target/ in a named volume instead is what
# makes rebuilds take seconds rather than minutes.
ENV CARGO_TARGET_DIR=/build/target
WORKDIR /work
