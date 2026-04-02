#!/bin/bash
#SBATCH --job-name=build-jaxmg-cusolvermp
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --time=01:00:00
#SBATCH --chdir=/projects/u6n
#SBATCH --output=/projects/u6n/logs/%x-%j.out
#SBATCH --error=/projects/u6n/logs/%x-%j.err

set -euo pipefail

export REPO_DIR="${REPO_DIR:-$HOME/benchmark-src/jaxmg}"
export GIT_URL="${GIT_URL:-https://github.com/JacobTutt/jaxmg}"
export GIT_REF="${GIT_REF:-cyclic_2d}"
export BUILD_DIR="${BUILD_DIR:-$REPO_DIR/build}"

source "$HOME/miniforge3/bin/activate" bayeseor
module load gcc-native/13.2 cudatoolkit/24.11_12.6 nvhpc/24.11

export CUDA_REAL_ROOT="${CUDA_REAL_ROOT:-/opt/nvidia/hpc_sdk/Linux_aarch64/24.11/cuda/12.6}"
export CUDA_COMPAT_ROOT="${CUDA_COMPAT_ROOT:-$REPO_DIR/.cuda_compat}"
export CUDA_ROOT="$CUDA_COMPAT_ROOT"
export CUDA_TARGET_INCLUDE_DIR="${CUDA_TARGET_INCLUDE_DIR:-$CUDA_REAL_ROOT/targets/sbsa-linux/include}"
export CUDA_MATH_INCLUDE_DIR="${CUDA_MATH_INCLUDE_DIR:-/opt/nvidia/hpc_sdk/Linux_aarch64/24.11/math_libs/12.6/targets/sbsa-linux/include}"
export GCC_NATIVE_ROOT="${GCC_NATIVE_ROOT:-/opt/cray/pe/gcc-native/13}"
export CUDNN_SOURCE_ROOT="${CUDNN_SOURCE_ROOT:-$HOME/miniforge3/envs/bayeseor/lib/python3.11/site-packages/nvidia/cudnn}"
export CUDNN_SOURCE_INCLUDE_DIR="${CUDNN_SOURCE_INCLUDE_DIR:-$CUDNN_SOURCE_ROOT/include}"
export CUDNN_SOURCE_LIB_DIR="${CUDNN_SOURCE_LIB_DIR:-$CUDNN_SOURCE_ROOT/lib}"
export CUDNN_COMPAT_ROOT="${CUDNN_COMPAT_ROOT:-$REPO_DIR/.cudnn_compat}"
export CUDNN_ROOT="${CUDNN_ROOT:-$CUDNN_COMPAT_ROOT}"
export CUDNN_ROOT_DIR="${CUDNN_ROOT_DIR:-$CUDNN_COMPAT_ROOT}"
export CUDNN_INCLUDE_DIR="${CUDNN_INCLUDE_DIR:-$CUDNN_COMPAT_ROOT/include}"
export CUDNN_LIBRARY="${CUDNN_LIBRARY:-$CUDNN_COMPAT_ROOT/lib}"
export CUSOLVERMG_LIB="${CUSOLVERMG_LIB:-/opt/nvidia/hpc_sdk/Linux_aarch64/24.11/math_libs/12.6/lib64/libcusolverMg.so}"
export CUSOLVERMP_LIB="${CUSOLVERMP_LIB:-/opt/nvidia/hpc_sdk/Linux_aarch64/24.11/math_libs/12.6/lib64/libcusolverMp.so}"
export NCCL_LIB="${NCCL_LIB:-/opt/nvidia/hpc_sdk/Linux_aarch64/24.11/comm_libs/12.6/lib/libnccl.so}"
export NCCL_INCLUDE_DIR="${NCCL_INCLUDE_DIR:-/opt/nvidia/hpc_sdk/Linux_aarch64/24.11/comm_libs/12.6/include}"
export JAX_VERSION="${JAX_VERSION:-0.9.1}"
export PYTHONUNBUFFERED=1

if [[ -x "$GCC_NATIVE_ROOT/bin/gcc" && -x "$GCC_NATIVE_ROOT/bin/g++" ]]; then
  export CC="$GCC_NATIVE_ROOT/bin/gcc"
  export CXX="$GCC_NATIVE_ROOT/bin/g++"
else
  export CC="$(command -v gcc)"
  export CXX="$(command -v g++)"
fi
export CUDAHOSTCXX="$CXX"
export LD_LIBRARY_PATH="$CUDNN_SOURCE_LIB_DIR:/opt/nvidia/hpc_sdk/Linux_aarch64/24.11/math_libs/12.6/lib64:$CUDA_REAL_ROOT/lib64:$CUDA_REAL_ROOT/extras/CUPTI/lib64:${LD_LIBRARY_PATH:-}"

mkdir -p "$(dirname "$REPO_DIR")"

if [[ ! -d "$REPO_DIR/.git" ]]; then
  git clone --branch "$GIT_REF" "$GIT_URL" "$REPO_DIR"
else
  git -C "$REPO_DIR" fetch origin "$GIT_REF"
  git -C "$REPO_DIR" checkout -f FETCH_HEAD
fi

rm -rf "$CUDNN_COMPAT_ROOT"
mkdir -p "$CUDNN_COMPAT_ROOT/lib"
ln -sfn "$CUDNN_SOURCE_INCLUDE_DIR" "$CUDNN_COMPAT_ROOT/include"
ln -sfn "$CUDNN_SOURCE_LIB_DIR/libcudnn.so.9" "$CUDNN_COMPAT_ROOT/lib/libcudnn.so"

rm -rf "$CUDA_COMPAT_ROOT"
mkdir -p "$CUDA_COMPAT_ROOT" "$CUDA_COMPAT_ROOT/targets" "$CUDA_COMPAT_ROOT/extras" "$CUDA_COMPAT_ROOT/include"
ln -sfn "$CUDA_REAL_ROOT/bin" "$CUDA_COMPAT_ROOT/bin"
ln -sfn "$CUDA_REAL_ROOT/lib64" "$CUDA_COMPAT_ROOT/lib64"
ln -sfn "$CUDA_REAL_ROOT/targets/sbsa-linux" "$CUDA_COMPAT_ROOT/targets/sbsa-linux"
ln -sfn "$CUDA_REAL_ROOT/extras/CUPTI" "$CUDA_COMPAT_ROOT/extras/CUPTI"
for src_dir in "$CUDA_TARGET_INCLUDE_DIR" "$CUDA_MATH_INCLUDE_DIR"; do
  find "$src_dir" -mindepth 1 -maxdepth 1 -exec ln -sfn {} "$CUDA_COMPAT_ROOT/include/" \;
done

rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

cmake .. \
  -DJAXMG_EXPERIMENTAL_CUSOLVERMP=ON \
  -DCMAKE_C_COMPILER="$CC" \
  -DCMAKE_CXX_COMPILER="$CXX" \
  -DCMAKE_CUDA_HOST_COMPILER="$CUDAHOSTCXX" \
  -DCUDA_TOOLKIT_ROOT_DIR="$CUDA_ROOT" \
  -DCUSOLVERMG_LIB="$CUSOLVERMG_LIB" \
  -DCUSOLVERMP_LIB="$CUSOLVERMP_LIB" \
  -DNCCL_LIB="$NCCL_LIB" \
  -DNCCL_INCLUDE_DIR="$NCCL_INCLUDE_DIR" \
  -DCUDNN_ROOT_DIR="$CUDNN_ROOT_DIR" \
  -DCUDNN_INCLUDE_DIR="$CUDNN_INCLUDE_DIR" \
  -DCUDNN_LIBRARY="$CUDNN_LIBRARY"

cmake --build . --target jaxmg_cusolvermp_context -j 8
