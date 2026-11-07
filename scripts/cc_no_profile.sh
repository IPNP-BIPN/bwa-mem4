#!/usr/bin/env bash
# C compiler wrapper for PGO builds. cc >= 1.2 forwards rustc's -Cprofile-generate / -Cprofile-use
# to C as -fprofile-*, so every bundled C library (htslib, zlib, zstd, libdeflate, libsais,
# mimalloc) gets instrumented by the system clang, whose raw-profile layout differs from rustc's
# LLVM. The mixed __llvm_prf_data section then corrupts every profile write (SIGSEGV at exit, or in
# mimalloc's thread init). The C stays uninstrumented, as it was before cc started inheriting.
args=()
for a in "$@"; do
  case "$a" in
    -fprofile-generate* | -fprofile-use*) ;;
    *) args+=("$a") ;;
  esac
done
exec "${PGO_REAL_CC:-cc}" "${args[@]}"
