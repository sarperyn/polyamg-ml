#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${PETSC_DIR:-}" || -z "${PETSC_ARCH:-}" ]]; then
  echo "PETSC_DIR and PETSC_ARCH must be set"
  exit 1
fi

cmake -S . -B build "$@"
cmake --build build -j
