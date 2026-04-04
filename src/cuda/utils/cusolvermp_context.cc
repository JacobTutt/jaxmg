#include "include/cusolvermp_context.h"

#if defined(JAXMG_HAVE_CUSOLVERMP) && defined(JAXMG_HAVE_NCCL)
#include <cuda_runtime_api.h>
#include <cusolverMp.h>
#include <nccl.h>
#endif

namespace jaxmg
{

CuSolverMpContextPlan BuildCuSolverMpContextPlan()
{
  return CuSolverMpContextPlan{
      .ffi_target_name = "potrs_cusolvermp",
      .library_name = "libpotrs_cusolvermp.so",
      .symbol_name = "PotrsCuSolverMpFFI",
      .distribution = "2d-block-cyclic",
      .local_layout = "column-major",
      .unsupported_operations = {"potri"},
  };
}

std::vector<std::string> CuSolverMpContextLaunchIssues(
    const CuSolverMpContextSpec &spec)
{
  std::vector<std::string> issues;

  if (spec.process_count != spec.global_device_count)
  {
    issues.push_back(
        "cuSOLVERMp expects one process per GPU, so process_count must match the global device count.");
  }
  if (spec.local_device_count != 1)
  {
    issues.push_back(
        "cuSOLVERMp expects each process to control exactly one local GPU.");
  }
  if (spec.process_rank < 0 || spec.process_rank >= spec.process_count)
  {
    issues.push_back("process_rank must satisfy 0 <= process_rank < process_count.");
  }
  if (spec.local_device_index != 0)
  {
    issues.push_back(
        "The initial cuSOLVERMp path assumes local_device_index == 0 within each process.");
  }
  if (spec.process_grid.nprow <= 0 || spec.process_grid.npcol <= 0)
  {
    issues.push_back("process_grid entries must be positive.");
  }
  if (spec.process_grid.nprow * spec.process_grid.npcol != spec.global_device_count)
  {
    issues.push_back(
        "process_grid must satisfy nprow * npcol == global_device_count.");
  }
  if (spec.mb <= 0 || spec.nb <= 0)
  {
    issues.push_back("Block sizes mb and nb must be positive.");
  }
  if (!spec.requires_mpi)
  {
    issues.push_back("The planned cuSOLVERMp backend assumes MPI bootstrap is enabled.");
  }
  if (!spec.requires_nccl)
  {
    issues.push_back("The planned cuSOLVERMp backend assumes NCCL communication is enabled.");
  }

  return issues;
}

bool CuSolverMpContextSpecIsLaunchReady(const CuSolverMpContextSpec &spec)
{
  return CuSolverMpContextLaunchIssues(spec).empty();
}

std::vector<std::string> CuSolverMpPotrsStubContractIssues(
    const CuSolverMpContextSpec &spec, int tile_size, int matrix_block_rows,
    int matrix_block_cols, int rhs_block_rows, int rhs_block_cols)
{
  std::vector<std::string> issues = CuSolverMpContextLaunchIssues(spec);

  if (tile_size <= 0)
  {
    issues.push_back("T_A must be positive.");
  }
  if (matrix_block_rows != tile_size || matrix_block_cols != tile_size)
  {
    issues.push_back("potrs_cusolvermp expects the matrix block shape to match T_A.");
  }
  if (rhs_block_rows <= 0)
  {
    issues.push_back("potrs_cusolvermp expects a positive RHS block row count.");
  }
  if (rhs_block_cols != 1)
  {
    issues.push_back("potrs_cusolvermp expects NRHS=1 in the initial cuSOLVERMp path.");
  }

  return issues;
}

std::optional<CuSolverMpRuntimeProbeResult> ProbeCuSolverMpRuntime(
    const CuSolverMpContextSpec &spec, std::string *error_message)
{
#if !(defined(JAXMG_HAVE_CUSOLVERMP) && defined(JAXMG_HAVE_NCCL))
  if (error_message != nullptr)
  {
    *error_message =
        "real cuSOLVERMp/NCCL support was not compiled into this library";
  }
  return std::nullopt;
#else
  if (spec.process_count != 1)
  {
    if (error_message != nullptr)
    {
      *error_message =
          "the initial real cuSOLVERMp probe only supports a single-rank launch";
    }
    return std::nullopt;
  }

  auto fail = [error_message](const std::string &message)
      -> std::optional<CuSolverMpRuntimeProbeResult> {
    if (error_message != nullptr)
    {
      *error_message = message;
    }
    return std::nullopt;
  };

  auto cuda_error_string = [](cudaError_t status) -> std::string {
    return cudaGetErrorString(status);
  };
  auto nccl_error_string = [](ncclResult_t status) -> std::string {
    return ncclGetErrorString(status);
  };
  auto cusolver_error_string = [](cusolverStatus_t status) -> std::string {
    return std::to_string(static_cast<int>(status));
  };

  cudaStream_t stream = nullptr;
  ncclComm_t comm = nullptr;
  cusolverMpHandle_t handle = nullptr;
  cusolverMpGrid_t grid = nullptr;

  cudaError_t cuda_status = cudaSetDevice(spec.local_device_index);
  if (cuda_status != cudaSuccess)
  {
    return fail("cudaSetDevice failed: " + cuda_error_string(cuda_status));
  }

  int device_id = -1;
  cuda_status = cudaGetDevice(&device_id);
  if (cuda_status != cudaSuccess)
  {
    return fail("cudaGetDevice failed: " + cuda_error_string(cuda_status));
  }

  ncclUniqueId comm_id;
  ncclResult_t nccl_status = ncclGetUniqueId(&comm_id);
  if (nccl_status != ncclSuccess)
  {
    return fail("ncclGetUniqueId failed: " + nccl_error_string(nccl_status));
  }

  nccl_status = ncclCommInitRank(&comm, spec.process_count, comm_id, spec.process_rank);
  if (nccl_status != ncclSuccess)
  {
    return fail("ncclCommInitRank failed: " + nccl_error_string(nccl_status));
  }

  cuda_status = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
  if (cuda_status != cudaSuccess)
  {
    ncclCommDestroy(comm);
    return fail("cudaStreamCreateWithFlags failed: " + cuda_error_string(cuda_status));
  }

  cusolverStatus_t cusolver_status = cusolverMpCreate(&handle, device_id, stream);
  if (cusolver_status != CUSOLVER_STATUS_SUCCESS)
  {
    cudaStreamDestroy(stream);
    ncclCommDestroy(comm);
    return fail("cusolverMpCreate failed with status " +
                cusolver_error_string(cusolver_status));
  }

  int cusolvermp_version = 0;
  cusolver_status = cusolverMpGetVersion(handle, &cusolvermp_version);
  if (cusolver_status != CUSOLVER_STATUS_SUCCESS)
  {
    cusolverMpDestroy(handle);
    cudaStreamDestroy(stream);
    ncclCommDestroy(comm);
    return fail("cusolverMpGetVersion failed with status " +
                cusolver_error_string(cusolver_status));
  }

  cusolver_status = cusolverMpCreateDeviceGrid(
      handle, &grid, comm, spec.process_grid.nprow, spec.process_grid.npcol,
      CUSOLVERMP_GRID_MAPPING_COL_MAJOR);
  if (cusolver_status != CUSOLVER_STATUS_SUCCESS)
  {
    cusolverMpDestroy(handle);
    cudaStreamDestroy(stream);
    ncclCommDestroy(comm);
    return fail("cusolverMpCreateDeviceGrid failed with status " +
                cusolver_error_string(cusolver_status));
  }

  int nccl_version = 0;
  nccl_status = ncclGetVersion(&nccl_version);
  if (nccl_status != ncclSuccess)
  {
    cusolverMpDestroyGrid(grid);
    cusolverMpDestroy(handle);
    cudaStreamDestroy(stream);
    ncclCommDestroy(comm);
    return fail("ncclGetVersion failed: " + nccl_error_string(nccl_status));
  }

  cusolverMpDestroyGrid(grid);
  cusolverMpDestroy(handle);
  cudaStreamDestroy(stream);
  ncclCommDestroy(comm);

  return CuSolverMpRuntimeProbeResult{
      .cuda_device_id = device_id,
      .nccl_version = nccl_version,
      .cusolvermp_version = cusolvermp_version,
  };
#endif
}

} // namespace jaxmg
