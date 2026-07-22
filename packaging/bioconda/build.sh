#!/bin/bash
set -euo pipefail

# Runtime CPU dispatch means a plain build runs everywhere on the arch; override the repo's
# .cargo/config.toml `-C target-cpu=native`, which would bake in the builder's CPU and SIGILL on
# anything older.
export RUSTFLAGS=""
# hts-sys' bindgen needs to find libclang from the clangdev build dep.
export LIBCLANG_PATH="${BUILD_PREFIX}/lib"
# Let htslib's build find zlib/bzip2/xz/libdeflate from the host prefix.
export CFLAGS="${CFLAGS:-} -I${PREFIX}/include"
export LDFLAGS="${LDFLAGS:-} -L${PREFIX}/lib"

# Build and install only the binary crate; --locked pins the committed Cargo.lock.
cargo install --locked --no-track --path crates/bwa-cli --root "${PREFIX}"

# `cargo install` puts it at ${PREFIX}/bin/bwa-mem4.
