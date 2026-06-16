#!/usr/bin/env bash

if [ -z "${CUDA_HOME:-}" ] && command -v nvcc >/dev/null 2>&1; then
  CUDA_HOME="$(dirname "$(dirname "$(readlink -f "$(command -v nvcc)")")")"
fi
if [ -n "${CUDA_HOME:-}" ]; then
  export CUDA_HOME
  export PATH="$CUDA_HOME/bin:$PATH"
  export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
fi

if [ -z "${CC:-}" ] || [ -z "${CXX:-}" ]; then
  for version in 14 13 12 11; do
    if command -v "gcc-$version" >/dev/null 2>&1 && command -v "g++-$version" >/dev/null 2>&1; then
      CC="${CC:-$(command -v "gcc-$version")}"
      CXX="${CXX:-$(command -v "g++-$version")}"
      break
    fi
  done
fi
if [ -n "${CC:-}" ]; then
  export CC
fi
if [ -n "${CXX:-}" ]; then
  export CXX
fi
if [ -z "${CUDAHOSTCXX:-}" ] && [ -n "${CXX:-}" ]; then
  CUDAHOSTCXX="$CXX"
fi
if [ -n "${CUDAHOSTCXX:-}" ]; then
  export CUDAHOSTCXX
fi
