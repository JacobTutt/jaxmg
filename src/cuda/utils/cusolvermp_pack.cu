#include "include/cusolvermp_pack.h"

#include <algorithm>
#include <cstdint>
#include <sstream>
#include <string>

#if defined(JAXMG_HAVE_CUSOLVERMP) && defined(JAXMG_HAVE_NCCL)
#include <cuComplex.h>

namespace jaxmg
{

namespace
{

template <typename T>
__device__ T ZeroValueDevice()
{
  return static_cast<T>(0);
}

template <>
__device__ cuDoubleComplex ZeroValueDevice<cuDoubleComplex>()
{
  return make_cuDoubleComplex(0.0, 0.0);
}

inline std::string CudaErrorString(cudaError_t status)
{
  return std::string(cudaGetErrorName(status)) + ": " + cudaGetErrorString(status);
}

inline std::string NcclErrorString(ncclResult_t status)
{
  return std::string(ncclGetErrorString(status));
}

inline int64_t PositiveProduct(int64_t lhs, int64_t rhs)
{
  return std::max<int64_t>(lhs, 1) * std::max<int64_t>(rhs, 1);
}

template <typename T>
__global__ void PackGlobalRowMajorToLocalBlockCyclicKernel(
    const T *src_rowmajor, T *dst_colmajor, int64_t global_rows,
    int64_t global_cols, int64_t local_rows, int64_t local_cols, int block_rows,
    int block_cols, int process_row, int process_col, int nprow, int npcol)
{
  const int64_t linear_index =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total = local_rows * local_cols;
  if (linear_index >= total)
  {
    return;
  }

  const int64_t local_row = linear_index % local_rows;
  const int64_t local_col = linear_index / local_rows;
  const int64_t global_block_row = (local_row / block_rows) * nprow + process_row;
  const int64_t global_block_col = (local_col / block_cols) * npcol + process_col;
  const int64_t global_row = global_block_row * block_rows + (local_row % block_rows);
  const int64_t global_col = global_block_col * block_cols + (local_col % block_cols);

  T value = ZeroValueDevice<T>();
  if (global_row < global_rows && global_col < global_cols)
  {
    value = src_rowmajor[global_row * global_cols + global_col];
  }
  dst_colmajor[local_row + local_col * local_rows] = value;
}

template <typename T>
cudaError_t LaunchPackGlobalRowMajorToLocalBlockCyclic(
    const T *src_rowmajor, T *dst_colmajor, int64_t global_rows,
    int64_t global_cols, int64_t local_rows, int64_t local_cols, int block_rows,
    int block_cols, int process_row, int process_col, int nprow, int npcol,
    cudaStream_t stream)
{
  if (local_rows <= 0 || local_cols <= 0)
  {
    return cudaSuccess;
  }
  const int64_t total = std::max<int64_t>(local_rows, 1) * std::max<int64_t>(local_cols, 1);
  const int threads = 256;
  const int blocks = static_cast<int>((total + threads - 1) / threads);
  PackGlobalRowMajorToLocalBlockCyclicKernel<T><<<blocks, threads, 0, stream>>>(
      src_rowmajor, dst_colmajor, global_rows, global_cols, local_rows, local_cols,
      block_rows, block_cols, process_row, process_col, nprow, npcol);
  return cudaGetLastError();
}

template <typename T>
CuSolverMpGpuPackResult TryPackCuSolverMpInputsGpuImpl(
    const CuSolverMpContextSpec &spec, const CuSolverMpPotrsProblemSpec &problem,
    const void *input_a, size_t input_a_elements, const void *input_b,
    size_t input_b_elements, void *d_a, int64_t local_matrix_rows,
    int64_t local_matrix_cols, void *d_b, int64_t local_rhs_rows,
    int64_t local_rhs_cols, ncclComm_t comm, cudaStream_t stream)
{
  CuSolverMpGpuPackResult unsupported{false, ""};
  if (input_a == nullptr || input_b == nullptr || spec.process_count <= 1)
  {
    return unsupported;
  }

  const size_t full_a_elements =
      static_cast<size_t>(problem.matrix_rows * problem.matrix_cols);
  const size_t full_b_elements =
      static_cast<size_t>(problem.rhs_rows * problem.rhs_cols);
  const bool a_is_full = input_a_elements == full_a_elements;
  const bool b_is_full = input_b_elements == full_b_elements;
  const bool a_is_row_sharded =
      !a_is_full && problem.matrix_cols > 0 &&
      input_a_elements % static_cast<size_t>(problem.matrix_cols) == 0 &&
      input_a_elements * static_cast<size_t>(spec.process_count) == full_a_elements;
  const bool b_is_row_sharded =
      !b_is_full && problem.rhs_cols > 0 &&
      input_b_elements % static_cast<size_t>(problem.rhs_cols) == 0 &&
      input_b_elements * static_cast<size_t>(spec.process_count) == full_b_elements;

  if ((!a_is_full && !a_is_row_sharded) || (!b_is_full && !b_is_row_sharded))
  {
    return unsupported;
  }

  void *d_global_a = const_cast<void *>(input_a);
  void *d_global_b = const_cast<void *>(input_b);
  bool free_global_a = false;
  bool free_global_b = false;

  auto fail = [&](const std::string &message) -> CuSolverMpGpuPackResult {
    if (free_global_a)
    {
      cudaFree(d_global_a);
    }
    if (free_global_b)
    {
      cudaFree(d_global_b);
    }
    return CuSolverMpGpuPackResult{false, message};
  };

  if (a_is_row_sharded)
  {
    const size_t bytes = sizeof(T) * full_a_elements;
    cudaError_t cuda_status = cudaMalloc(&d_global_a, std::max<size_t>(bytes, 1));
    if (cuda_status != cudaSuccess)
    {
      return fail("cudaMalloc(global row-sharded A) failed: " +
                  CudaErrorString(cuda_status));
    }
    free_global_a = true;
    ncclResult_t nccl_status = ncclAllGather(
        input_a, d_global_a, sizeof(T) * input_a_elements, ncclUint8, comm, stream);
    if (nccl_status != ncclSuccess)
    {
      return fail("ncclAllGather(input A) failed: " + NcclErrorString(nccl_status));
    }
  }

  if (b_is_row_sharded)
  {
    const size_t bytes = sizeof(T) * full_b_elements;
    cudaError_t cuda_status = cudaMalloc(&d_global_b, std::max<size_t>(bytes, 1));
    if (cuda_status != cudaSuccess)
    {
      return fail("cudaMalloc(global row-sharded B) failed: " +
                  CudaErrorString(cuda_status));
    }
    free_global_b = true;
    ncclResult_t nccl_status = ncclAllGather(
        input_b, d_global_b, sizeof(T) * input_b_elements, ncclUint8, comm, stream);
    if (nccl_status != ncclSuccess)
    {
      return fail("ncclAllGather(input B) failed: " + NcclErrorString(nccl_status));
    }
  }

  cudaError_t cuda_status = LaunchPackGlobalRowMajorToLocalBlockCyclic(
      static_cast<const T *>(d_global_a), static_cast<T *>(d_a), problem.matrix_rows,
      problem.matrix_cols, local_matrix_rows, local_matrix_cols,
      problem.matrix_block_rows, problem.matrix_block_cols,
      spec.process_rank / spec.process_grid.npcol,
      spec.process_rank % spec.process_grid.npcol, spec.process_grid.nprow,
      spec.process_grid.npcol, stream);
  if (cuda_status != cudaSuccess)
  {
    return fail("pack kernel(A) launch failed: " + CudaErrorString(cuda_status));
  }

  cuda_status = LaunchPackGlobalRowMajorToLocalBlockCyclic(
      static_cast<const T *>(d_global_b), static_cast<T *>(d_b), problem.rhs_rows,
      problem.rhs_cols, local_rhs_rows, local_rhs_cols, problem.rhs_block_rows,
      problem.rhs_block_cols, spec.process_rank / spec.process_grid.npcol,
      spec.process_rank % spec.process_grid.npcol, spec.process_grid.nprow,
      spec.process_grid.npcol, stream);
  if (cuda_status != cudaSuccess)
  {
    return fail("pack kernel(B) launch failed: " + CudaErrorString(cuda_status));
  }

  cuda_status = cudaStreamSynchronize(stream);
  if (cuda_status != cudaSuccess)
  {
    return fail("cudaStreamSynchronize(after gpu pack) failed: " +
                CudaErrorString(cuda_status));
  }

  if (free_global_a)
  {
    cudaFree(d_global_a);
  }
  if (free_global_b)
  {
    cudaFree(d_global_b);
  }
  return CuSolverMpGpuPackResult{true, ""};
}

} // namespace

CuSolverMpGpuPackResult TryPackCuSolverMpInputsGpu(
    CuSolverMpScalarType scalar_type, const CuSolverMpContextSpec &spec,
    const CuSolverMpPotrsProblemSpec &problem, const void *input_a,
    size_t input_a_elements, const void *input_b, size_t input_b_elements,
    void *d_a, int64_t local_matrix_rows, int64_t local_matrix_cols, void *d_b,
    int64_t local_rhs_rows, int64_t local_rhs_cols, ncclComm_t comm,
    cudaStream_t stream)
{
  switch (scalar_type)
  {
  case CuSolverMpScalarType::kF32:
    return TryPackCuSolverMpInputsGpuImpl<float>(
        spec, problem, input_a, input_a_elements, input_b, input_b_elements, d_a,
        local_matrix_rows, local_matrix_cols, d_b, local_rhs_rows, local_rhs_cols,
        comm, stream);
  case CuSolverMpScalarType::kF64:
    return TryPackCuSolverMpInputsGpuImpl<double>(
        spec, problem, input_a, input_a_elements, input_b, input_b_elements, d_a,
        local_matrix_rows, local_matrix_cols, d_b, local_rhs_rows, local_rhs_cols,
        comm, stream);
  case CuSolverMpScalarType::kC128:
    return TryPackCuSolverMpInputsGpuImpl<cuDoubleComplex>(
        spec, problem, input_a, input_a_elements, input_b, input_b_elements, d_a,
        local_matrix_rows, local_matrix_cols, d_b, local_rhs_rows, local_rhs_cols,
        comm, stream);
  }

  return CuSolverMpGpuPackResult{false, "unsupported scalar type for gpu pack"};
}

} // namespace jaxmg

#endif
