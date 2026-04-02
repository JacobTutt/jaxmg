#!/bin/bash

set -euo pipefail

TARGET_ENV="${TARGET_ENV:-}"
TARGET_PREFIX="${TARGET_PREFIX:-/projects/u6n/jaxmg-cyclic-2d-env}"
SOURCE_ENV="${SOURCE_ENV:-bayeseor}"
PYTHON_BIN="${PYTHON_BIN:-python}"

source "$HOME/miniforge3/etc/profile.d/conda.sh"

if ! conda env list | awk '{print $1}' | grep -qx "$SOURCE_ENV"; then
  echo "Source environment $SOURCE_ENV does not exist" >&2
  exit 1
fi

if [[ -n "$TARGET_ENV" && -n "$TARGET_PREFIX" ]]; then
  echo "Set only one of TARGET_ENV or TARGET_PREFIX" >&2
  exit 1
fi

if [[ -n "$TARGET_ENV" ]]; then
  activate_target="$TARGET_ENV"
  if conda env list | awk '{print $1}' | grep -qx "$TARGET_ENV"; then
    echo "Environment $activate_target already exists"
  else
    echo "Cloning $SOURCE_ENV into $activate_target"
    conda create -y -n "$TARGET_ENV" --clone "$SOURCE_ENV"
  fi
else
  mkdir -p "$(dirname "$TARGET_PREFIX")"
  activate_target="$TARGET_PREFIX"
  if [[ -d "$TARGET_PREFIX" ]]; then
    echo "Environment $activate_target already exists"
  else
    echo "Cloning $SOURCE_ENV into $activate_target"
    conda create -y -p "$TARGET_PREFIX" --clone "$SOURCE_ENV"
  fi
fi

conda activate "$activate_target"

echo "Installing migration-only extras into $activate_target"
"$PYTHON_BIN" -m pip install --upgrade pip
"$PYTHON_BIN" -m pip install mpi4py nvidia-cusolvermp-cu12

echo "Environment ready: $activate_target"
"$PYTHON_BIN" -m pip show jax jaxlib mpi4py nvidia-cusolvermp-cu12 | sed -n '1,80p'
