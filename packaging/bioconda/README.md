# Bioconda packaging for bwa-mem4

`meta.yaml` and `build.sh` are the Bioconda recipe. It has been **submitted** as
bioconda/bioconda-recipes#67394; Bioconda's `@BiocondaBot` builds and lints the
PR and their maintainers merge it (this repo cannot publish to the bioconda
channel itself). The recipe here is the source of truth; keep it in sync with
what lands there. For reference, the original submission steps were:

1. **Prerequisite (done):** a `v4.1.0` GitHub release exists. The recipe builds
   from its source tarball; `meta.yaml`'s `sha256` is the checksum of
   `https://github.com/IPNP-BIPN/bwa-mem4/archive/refs/tags/v4.1.0.tar.gz`
   (also in `SHA256` here). Regenerate with
   `curl -sL <url> | shasum -a 256` if the tag is ever recut.

2. **Submit:** fork `github.com/bioconda/bioconda-recipes`, then

   ```sh
   mkdir -p recipes/bwa-mem4
   cp packaging/bioconda/meta.yaml  recipes/bwa-mem4/meta.yaml
   cp packaging/bioconda/build.sh   recipes/bwa-mem4/build.sh
   ```

   and open a pull request. `@BiocondaBot` builds the recipe on Linux and macOS
   and comments the result; a maintainer merges. The package then lands on the
   `bioconda` channel and `conda install -c bioconda bwa-mem4` works, with
   Bioconda mirroring it to a Galaxy/Docker container automatically.

3. **Updating for a later release:** bump `version`, refresh `sha256`, reset
   `build.number` to 0, PR again.

Notes on the recipe:

- It builds the binary crate from source with cargo rather than installing from
  crates.io, so it does not depend on the crates.io publish ordering and gets
  the exact committed `Cargo.lock` via `--locked`.
- `hts-sys` compiles vendored htslib, hence the C/C++ compilers, `make`,
  `clangdev` (libclang, for its bindgen), and the zlib/bzip2/xz/libdeflate host
  libraries.
- `RUSTFLAGS=""` in `build.sh` overrides the repo's `-C target-cpu=native`,
  which would bake in the build host's CPU and SIGILL on older ones. The SIMD
  kernels are chosen at runtime, so nothing is lost.
