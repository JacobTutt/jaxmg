#!/bin/bash

set -euo pipefail

TARGET_ENV="${TARGET_ENV:-jaxmg-cyclic-2d}"
SOURCE_ENV="${SOURCE_ENV:-bayeseor}"
PYTHON_BIN="${PYTHON_BIN:-python}"

source "$HOME/miniforge3/etc/profile.d/conda.sh"

if ! conda env list | awk '{print $1}' | grep -qx "$SOURCE_ENV"; then
  echo "Source environment $SOURCE_ENV does not exist" >&2
  exit 1
fi

if conda env list | awk '{print $1}' | grep -qx "$TARGET_ENV"; then
  echo "Environment $TARGET_ENV already exists"
else
  echo "Cloning $SOURCE_ENV into $TARGET_ENV"
  conda create -y -n "$TARGET_ENV" --clone "$SOURCE_ENV"
fi

conda activate "$TARGET_ENV"

echo "Installing migration-only extras into $TARGET_ENV"
"$PYTHON_BIN" -m pip install --upgrade pip
"$PYTHON_BIN" -m pip install mpi4py nvidia-cusolvermp-cu12

echo "Environment ready: $TARGET_ENV"
"$PYTHON_BIN" -m pip show jax jaxlib mpi4py nvidia-cusolvermp-cu12 | sed -n '1,80p'
