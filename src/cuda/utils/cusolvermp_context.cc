#include "include/cusolvermp_context.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <thread>
#include <vector>

#if defined(JAXMG_HAVE_CUSOLVERMP) && defined(JAXMG_HAVE_NCCL)
#include <cuda_runtime_api.h>
#include <cusolverMp.h>
#include <nccl.h>
#endif

namespace jaxmg
{

namespace
{

std::string GetEnvOrDefault(const char *name, const std::string &default_value)
{
  const char *value = std::getenv(name);
  return (value != nullptr && value[0] != '\0') ? std::string(value) : default_value;
}

std::string ResolveNcclBootstrapPath(const CuSolverMpContextSpec &spec)
{
  const char *explicit_path = std::getenv("JAXMG_CUSOLVERMP_BOOTSTRAP_FILE");
  if (explicit_path != nullptr && explicit_path[0] != '\0')
  {
    return std::string(explicit_path);
  }

  std::string dir = GetEnvOrDefault("JAXMG_CUSOLVERMP_BOOTSTRAP_DIR", "/tmp");
  std::string token = GetEnvOrDefault(
      "JAXMG_CUSOLVERMP_BOOTSTRAP_TOKEN",
      GetEnvOrDefault("SLURM_JOB_ID", "local"));
  return dir + "/jaxmg_cusolvermp_nccl_" + token + "_" +
         std::to_string(spec.process_count) + ".bin";
}

bool EnvFlagEnabled(const char *name)
{
  const char *value = std::getenv(name);
  if (value == nullptr || value[0] == '\0')
  {
    return false;
  }
  return std::string(value) != "0";
}

bool StopAtStage(const char *stage)
{
  const char *value = std::getenv("JAXMG_CUSOLVERMP_STOP_STAGE");
  return value != nullptr && std::string(value) == stage;
}

#if defined(JAXMG_HAVE_CUSOLVERMP) && defined(JAXMG_HAVE_NCCL)
bool WriteNcclUniqueId(const std::string &path, const ncclUniqueId &comm_id)
{
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  if (!out)
  {
    return false;
  }
  out.write(reinterpret_cast<const char *>(&comm_id), sizeof(comm_id));
  out.close();
  return out.good();
}

bool ReadNcclUniqueIdWithRetry(
    const std::string &path, ncclUniqueId *comm_id, int timeout_ms)
{
  auto deadline =
      std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
  while (std::chrono::steady_clock::now() < deadline)
  {
    std::ifstream in(path, std::ios::binary);
    if (in)
    {
      in.read(reinterpret_cast<char *>(comm_id), sizeof(*comm_id));
      if (in.gcount() == sizeof(*comm_id))
      {
        return true;
      }
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
  }
  return false;
}
#endif

int OwnerCoord(int global_index, int block_size, int nprocs, int src_proc = 0)
{
  return (src_proc + global_index / block_size) % nprocs;
}

int LocalIndexFromGlobal(
    int global_index, int block_size, int proc_coord, int nprocs,
    int src_proc = 0)
{
  int block_index = global_index / block_size;
  int block_offset = (proc_coord - src_proc + nprocs) % nprocs;
  int local_block_index = (block_index - block_offset) / nprocs;
  return local_block_index * block_size + (global_index % block_size);
}

std::vector<float> RowMajorToColumnMajor(
    const std::vector<float> &row_major, int64_t rows, int64_t cols)
{
  std::vector<float> column_major(static_cast<size_t>(rows * cols), 0.0f);
  for (int64_t row = 0; row < rows; ++row)
  {
    for (int64_t col = 0; col < cols; ++col)
    {
      column_major[static_cast<size_t>(row + col * rows)] =
          row_major[static_cast<size_t>(row * cols + col)];
    }
  }
  return column_major;
}

std::vector<float> BuildSyntheticDenseSpdRowMajor(int64_t n)
{
  std::vector<float> lower(static_cast<size_t>(n * n), 0.0f);
  for (int64_t i = 0; i < n; ++i)
  {
    lower[static_cast<size_t>(i * n + i)] = 2.0f + static_cast<float>(i);
    for (int64_t j = 0; j < i; ++j)
    {
      lower[static_cast<size_t>(i * n + j)] =
          0.05f * static_cast<float>((i + 1) + (j + 1));
    }
  }

  std::vector<float> a(static_cast<size_t>(n * n), 0.0f);
  for (int64_t i = 0; i < n; ++i)
  {
    for (int64_t j = 0; j < n; ++j)
    {
      float sum = 0.0f;
      int64_t kmax = std::min(i, j);
      for (int64_t k = 0; k <= kmax; ++k)
      {
        sum += lower[static_cast<size_t>(i * n + k)] *
               lower[static_cast<size_t>(j * n + k)];
      }
      a[static_cast<size_t>(i * n + j)] = sum;
    }
  }
  return a;
}

std::string MultiRankDebugPrefix(
    const CuSolverMpContextSpec &spec, const CuSolverMpPotrsProblemSpec &problem,
    int64_t local_matrix_rows, int64_t local_matrix_cols, int64_t local_rhs_rows,
    int64_t local_rhs_cols, bool using_input_buffers)
{
  std::ostringstream oss;
  oss << "rank=" << spec.process_rank << "/" << spec.process_count
      << " grid=(" << spec.process_grid.nprow << "," << spec.process_grid.npcol
      << ") local_A=(" << local_matrix_rows << "," << local_matrix_cols
      << ") local_B=(" << local_rhs_rows << "," << local_rhs_cols
      << ") global_A=(" << problem.matrix_rows << "," << problem.matrix_cols
      << ") global_B=(" << problem.rhs_rows << "," << problem.rhs_cols
      << ") block_A=(" << problem.matrix_block_rows << "," << problem.matrix_block_cols
      << ") block_B=(" << problem.rhs_block_rows << "," << problem.rhs_block_cols
      << ") input_buffers=" << (using_input_buffers ? "yes" : "no");
  return oss.str();
}

} // namespace

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
  if (spec.local_device_index < 0)
  {
    issues.push_back("local_device_index must be non-negative.");
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
    const CuSolverMpContextSpec &spec, const CuSolverMpPotrsProblemSpec &problem,
    const void *input_a, const void *input_b, std::string *error_message)
{
#if !(defined(JAXMG_HAVE_CUSOLVERMP) && defined(JAXMG_HAVE_NCCL))
  if (error_message != nullptr)
  {
    *error_message =
        "real cuSOLVERMp/NCCL support was not compiled into this library";
  }
  return std::nullopt;
#else
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
  auto debug_log = [&spec](const char *stage) {
    std::fprintf(stderr, "rank=%d stage=%s\n", spec.process_rank, stage);
    std::fflush(stderr);
  };

  cudaStream_t stream = nullptr;
  ncclComm_t comm = nullptr;
  cusolverMpHandle_t handle = nullptr;
  cusolverMpGrid_t grid = nullptr;
  cusolverMpMatrixDescriptor_t desc_a = nullptr;
  cusolverMpMatrixDescriptor_t desc_b = nullptr;
  void *d_a = nullptr;
  void *d_b = nullptr;
  void *d_work = nullptr;
  int *d_info = nullptr;

  cudaError_t cuda_status = cudaSetDevice(spec.local_device_index);
  if (cuda_status != cudaSuccess)
  {
    return fail("cudaSetDevice failed: " + cuda_error_string(cuda_status));
  }
  bool potrf_only = EnvFlagEnabled("JAXMG_CUSOLVERMP_POTRF_ONLY");

  int device_id = -1;
  cuda_status = cudaGetDevice(&device_id);
  if (cuda_status != cudaSuccess)
  {
    return fail("cudaGetDevice failed: " + cuda_error_string(cuda_status));
  }

  ncclUniqueId comm_id;
  ncclResult_t nccl_status = ncclSuccess;
  std::string bootstrap_path = ResolveNcclBootstrapPath(spec);
  if (spec.process_rank == 0)
  {
    nccl_status = ncclGetUniqueId(&comm_id);
    if (nccl_status != ncclSuccess)
    {
      return fail("ncclGetUniqueId failed: " + nccl_error_string(nccl_status));
    }
    if (!WriteNcclUniqueId(bootstrap_path, comm_id))
    {
      return fail("failed to write NCCL bootstrap file: " + bootstrap_path);
    }
  }
  else
  {
    int timeout_ms = std::stoi(
        GetEnvOrDefault("JAXMG_CUSOLVERMP_BOOTSTRAP_TIMEOUT_MS", "10000"));
    if (!ReadNcclUniqueIdWithRetry(bootstrap_path, &comm_id, timeout_ms))
    {
      return fail("timed out waiting for NCCL bootstrap file: " + bootstrap_path);
    }
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

  int process_row = spec.process_rank % spec.process_grid.nprow;
  int process_col = spec.process_rank / spec.process_grid.nprow;
  int64_t local_matrix_rows = cusolverMpNUMROC(
      problem.matrix_rows, problem.matrix_block_rows, process_row, 0,
      spec.process_grid.nprow);
  int64_t local_matrix_cols = cusolverMpNUMROC(
      problem.matrix_cols, problem.matrix_block_cols, process_col, 0,
      spec.process_grid.npcol);
  int64_t local_rhs_rows = cusolverMpNUMROC(
      problem.rhs_rows, problem.rhs_block_rows, process_row, 0,
      spec.process_grid.nprow);
  int64_t local_rhs_cols = cusolverMpNUMROC(
      problem.rhs_cols, problem.rhs_block_cols, process_col, 0,
      spec.process_grid.npcol);

  cusolver_status = cusolverMpCreateMatrixDesc(
      &desc_a, grid, CUDA_R_32F, problem.matrix_rows, problem.matrix_cols,
      problem.matrix_block_rows, problem.matrix_block_cols, 0, 0,
      std::max<int64_t>(1, local_matrix_rows));
  if (cusolver_status != CUSOLVER_STATUS_SUCCESS)
  {
    cusolverMpDestroyGrid(grid);
    cusolverMpDestroy(handle);
    cudaStreamDestroy(stream);
    ncclCommDestroy(comm);
    return fail("cusolverMpCreateMatrixDesc(A) failed with status " +
                cusolver_error_string(cusolver_status));
  }

  cusolver_status = cusolverMpCreateMatrixDesc(
      &desc_b, grid, CUDA_R_32F, problem.rhs_rows, problem.rhs_cols,
      problem.rhs_block_rows, problem.rhs_block_cols, 0, 0,
      std::max<int64_t>(1, local_rhs_rows));
  if (cusolver_status != CUSOLVER_STATUS_SUCCESS)
  {
    cusolverMpDestroyMatrixDesc(desc_a);
    cusolverMpDestroyGrid(grid);
    cusolverMpDestroy(handle);
    cudaStreamDestroy(stream);
    ncclCommDestroy(comm);
    return fail("cusolverMpCreateMatrixDesc(B) failed with status " +
                cusolver_error_string(cusolver_status));
  }

  size_t matrix_bytes = static_cast<size_t>(
      std::max<int64_t>(1, local_matrix_rows) *
      std::max<int64_t>(1, local_matrix_cols) * sizeof(float));
  cuda_status = cudaMalloc(&d_a, matrix_bytes);
  if (cuda_status != cudaSuccess)
  {
    cusolverMpDestroyMatrixDesc(desc_b);
    cusolverMpDestroyMatrixDesc(desc_a);
    cusolverMpDestroyGrid(grid);
    cusolverMpDestroy(handle);
    cudaStreamDestroy(stream);
    ncclCommDestroy(comm);
    return fail("cudaMalloc(A) failed: " + cuda_error_string(cuda_status));
  }

  size_t rhs_bytes = static_cast<size_t>(
      std::max<int64_t>(1, local_rhs_rows) *
      std::max<int64_t>(1, local_rhs_cols) * sizeof(float));
  cuda_status = cudaMalloc(&d_b, rhs_bytes);
  if (cuda_status != cudaSuccess)
  {
    cudaFree(d_a);
    cusolverMpDestroyMatrixDesc(desc_b);
    cusolverMpDestroyMatrixDesc(desc_a);
      cusolverMpDestroyGrid(grid);
      cusolverMpDestroy(handle);
      cudaStreamDestroy(stream);
      ncclCommDestroy(comm);
      return fail("cudaMalloc(B) failed: " + cuda_error_string(cuda_status));
  }

  constexpr int64_t kSubmatrixStart = 1;

  size_t potrf_workspace_device_bytes = 0;
  size_t potrf_workspace_host_bytes = 0;
  cusolver_status = cusolverMpPotrf_bufferSize(
      handle, CUBLAS_FILL_MODE_LOWER, problem.matrix_rows, d_a, kSubmatrixStart,
      kSubmatrixStart, desc_a, CUDA_R_32F, &potrf_workspace_device_bytes,
      &potrf_workspace_host_bytes);
  if (cusolver_status != CUSOLVER_STATUS_SUCCESS)
  {
    cudaFree(d_b);
    cudaFree(d_a);
    cusolverMpDestroyMatrixDesc(desc_b);
    cusolverMpDestroyMatrixDesc(desc_a);
    cusolverMpDestroyGrid(grid);
    cusolverMpDestroy(handle);
    cudaStreamDestroy(stream);
    ncclCommDestroy(comm);
    return fail("cusolverMpPotrf_bufferSize failed with status " +
                cusolver_error_string(cusolver_status));
  }

  size_t potrs_workspace_device_bytes = 0;
  size_t potrs_workspace_host_bytes = 0;
  if (!potrf_only)
  {
    cusolver_status = cusolverMpPotrs_bufferSize(
        handle, CUBLAS_FILL_MODE_LOWER, problem.matrix_rows, problem.rhs_cols,
        d_a, kSubmatrixStart, kSubmatrixStart, desc_a, d_b, kSubmatrixStart,
        kSubmatrixStart, desc_b, CUDA_R_32F, &potrs_workspace_device_bytes,
        &potrs_workspace_host_bytes);
    if (cusolver_status != CUSOLVER_STATUS_SUCCESS)
    {
      cudaFree(d_b);
      cudaFree(d_a);
      cusolverMpDestroyMatrixDesc(desc_b);
      cusolverMpDestroyMatrixDesc(desc_a);
      cusolverMpDestroyGrid(grid);
      cusolverMpDestroy(handle);
      cudaStreamDestroy(stream);
      ncclCommDestroy(comm);
      return fail("cusolverMpPotrs_bufferSize failed with status " +
                  cusolver_error_string(cusolver_status));
    }
  }
  debug_log("after_workspace_queries");
  if (StopAtStage("after_workspace_queries"))
  {
    return fail("stopped after workspace queries");
  }
  std::fprintf(stderr,
               "rank=%d local_sizes A=(%lld,%lld) B=(%lld,%lld)\n",
               spec.process_rank, static_cast<long long>(local_matrix_rows),
               static_cast<long long>(local_matrix_cols),
               static_cast<long long>(local_rhs_rows),
               static_cast<long long>(local_rhs_cols));
  std::fflush(stderr);
  if (StopAtStage("before_host_vector_alloc"))
  {
    return fail("stopped before host vector alloc");
  }

  debug_log("before_host_vector_alloc");
  std::vector<float> host_a(
      static_cast<size_t>(std::max<int64_t>(1, local_matrix_rows) *
                          std::max<int64_t>(1, local_matrix_cols)),
      0.0f);
  std::vector<float> host_b(
      static_cast<size_t>(std::max<int64_t>(1, local_rhs_rows) *
                          std::max<int64_t>(1, local_rhs_cols)),
      1.0f);
  debug_log("after_host_vector_alloc");
  std::vector<float> host_input_a_rowmajor;
  std::vector<float> host_input_b;
  bool use_input_buffers = input_a != nullptr && input_b != nullptr &&
                           !EnvFlagEnabled("JAXMG_CUSOLVERMP_USE_SYNTHETIC_LOCAL");
  auto debug_prefix = [&]() {
    return MultiRankDebugPrefix(spec, problem, local_matrix_rows,
                                local_matrix_cols, local_rhs_rows,
                                local_rhs_cols, use_input_buffers);
  };
  if (use_input_buffers)
  {
    host_input_a_rowmajor.resize(static_cast<size_t>(problem.matrix_rows * problem.matrix_cols));
    host_input_b.resize(static_cast<size_t>(problem.rhs_rows * problem.rhs_cols));

    cuda_status = cudaMemcpy(host_input_a_rowmajor.data(), input_a,
                             sizeof(float) * host_input_a_rowmajor.size(),
                             cudaMemcpyDeviceToHost);
    if (cuda_status != cudaSuccess)
    {
      cudaFree(d_b);
      cudaFree(d_a);
      cusolverMpDestroyMatrixDesc(desc_b);
      cusolverMpDestroyMatrixDesc(desc_a);
      cusolverMpDestroyGrid(grid);
      cusolverMpDestroy(handle);
      cudaStreamDestroy(stream);
      ncclCommDestroy(comm);
      return fail("cudaMemcpy(input A) failed: " + cuda_error_string(cuda_status));
    }

    cuda_status = cudaMemcpy(host_input_b.data(), input_b,
                             sizeof(float) * host_input_b.size(),
                             cudaMemcpyDeviceToHost);
    if (cuda_status != cudaSuccess)
    {
      cudaFree(d_b);
      cudaFree(d_a);
      cusolverMpDestroyMatrixDesc(desc_b);
      cusolverMpDestroyMatrixDesc(desc_a);
      cusolverMpDestroyGrid(grid);
      cusolverMpDestroy(handle);
      cudaStreamDestroy(stream);
      ncclCommDestroy(comm);
      return fail("cudaMemcpy(input b) failed: " + cuda_error_string(cuda_status));
    }

    if (spec.process_count > 1)
    {
      std::vector<float> host_input_a_colmajor =
          RowMajorToColumnMajor(host_input_a_rowmajor, problem.matrix_rows,
                                problem.matrix_cols);

      cusolver_status = cusolverMpMatrixScatterH2D(
          handle, problem.matrix_rows, problem.matrix_cols, d_a, kSubmatrixStart,
          kSubmatrixStart, desc_a, 0, host_input_a_colmajor.data(),
          problem.matrix_rows);
      if (cusolver_status != CUSOLVER_STATUS_SUCCESS)
      {
        cudaFree(d_b);
        cudaFree(d_a);
        cusolverMpDestroyMatrixDesc(desc_b);
        cusolverMpDestroyMatrixDesc(desc_a);
        cusolverMpDestroyGrid(grid);
        cusolverMpDestroy(handle);
        cudaStreamDestroy(stream);
        ncclCommDestroy(comm);
        return fail("cusolverMpMatrixScatterH2D(A) failed with status " +
                    cusolver_error_string(cusolver_status) + " [" +
                    debug_prefix() + "]");
      }
      debug_log("after_scatter_a");

      cusolver_status = cusolverMpMatrixScatterH2D(
          handle, problem.rhs_rows, problem.rhs_cols, d_b, kSubmatrixStart,
          kSubmatrixStart, desc_b, 0, host_input_b.data(),
          problem.rhs_rows);
      if (cusolver_status != CUSOLVER_STATUS_SUCCESS)
      {
        cudaFree(d_b);
        cudaFree(d_a);
        cusolverMpDestroyMatrixDesc(desc_b);
        cusolverMpDestroyMatrixDesc(desc_a);
        cusolverMpDestroyGrid(grid);
        cusolverMpDestroy(handle);
        cudaStreamDestroy(stream);
        ncclCommDestroy(comm);
        return fail("cusolverMpMatrixScatterH2D(B) failed with status " +
                    cusolver_error_string(cusolver_status) + " [" +
                    debug_prefix() + "]");
      }
      debug_log("after_scatter_b");
      cuda_status = cudaStreamSynchronize(stream);
      if (cuda_status != cudaSuccess)
      {
        cudaFree(d_b);
        cudaFree(d_a);
        cusolverMpDestroyMatrixDesc(desc_b);
        cusolverMpDestroyMatrixDesc(desc_a);
        cusolverMpDestroyGrid(grid);
        cusolverMpDestroy(handle);
        cudaStreamDestroy(stream);
        ncclCommDestroy(comm);
        return fail("cudaStreamSynchronize(after scatter) failed: " +
                    cuda_error_string(cuda_status) + " [" + debug_prefix() + "]");
      }
      debug_log("after_scatter_sync");
      if (StopAtStage("after_scatter_sync"))
      {
        return fail("stopped after scatter sync");
      }
    }
    else
    {
      for (int64_t col = 0; col < problem.matrix_cols; ++col)
      {
        if (OwnerCoord(col, problem.matrix_block_cols, spec.process_grid.npcol) !=
            process_col)
        {
          continue;
        }
        int local_col = LocalIndexFromGlobal(
            col, problem.matrix_block_cols, process_col, spec.process_grid.npcol);
        for (int64_t row = 0; row < problem.matrix_rows; ++row)
        {
          if (OwnerCoord(row, problem.matrix_block_rows, spec.process_grid.nprow) !=
              process_row)
          {
            continue;
          }
          int local_row = LocalIndexFromGlobal(
              row, problem.matrix_block_rows, process_row, spec.process_grid.nprow);
          host_a[static_cast<size_t>(local_row + local_col * local_matrix_rows)] =
              host_input_a_rowmajor[static_cast<size_t>(row * problem.matrix_cols + col)];
        }
      }

      for (int64_t col = 0; col < problem.rhs_cols; ++col)
      {
        if (OwnerCoord(col, problem.rhs_block_cols, spec.process_grid.npcol) !=
            process_col)
        {
          continue;
        }
        int local_col = LocalIndexFromGlobal(
            col, problem.rhs_block_cols, process_col, spec.process_grid.npcol);
        for (int64_t row = 0; row < problem.rhs_rows; ++row)
        {
          if (OwnerCoord(row, problem.rhs_block_rows, spec.process_grid.nprow) !=
              process_row)
          {
            continue;
          }
          int local_row = LocalIndexFromGlobal(
              row, problem.rhs_block_rows, process_row, spec.process_grid.nprow);
          host_b[static_cast<size_t>(local_row + local_col * local_rhs_rows)] =
              host_input_b[static_cast<size_t>(row * problem.rhs_cols + col)];
        }
      }
    }
  }
  else
  {
    debug_log("before_synthetic_local_build");
    if (EnvFlagEnabled("JAXMG_CUSOLVERMP_USE_DENSE_SYNTHETIC"))
    {
      std::vector<float> synthetic_a_rowmajor =
          BuildSyntheticDenseSpdRowMajor(problem.matrix_rows);
      for (int64_t col = 0; col < problem.matrix_cols; ++col)
      {
        if (OwnerCoord(col, problem.matrix_block_cols, spec.process_grid.npcol) !=
            process_col)
        {
          continue;
        }
        int local_col = LocalIndexFromGlobal(
            col, problem.matrix_block_cols, process_col, spec.process_grid.npcol);
        for (int64_t row = 0; row < problem.matrix_rows; ++row)
        {
          if (OwnerCoord(row, problem.matrix_block_rows, spec.process_grid.nprow) !=
              process_row)
          {
            continue;
          }
          int local_row = LocalIndexFromGlobal(
              row, problem.matrix_block_rows, process_row, spec.process_grid.nprow);
          host_a[static_cast<size_t>(local_row + local_col * local_matrix_rows)] =
              synthetic_a_rowmajor[static_cast<size_t>(row * problem.matrix_cols + col)];
        }
      }
    }
    else
    {
      for (int64_t global_i = 0;
           global_i < std::min(problem.matrix_rows, problem.matrix_cols); ++global_i)
      {
        if (OwnerCoord(global_i, problem.matrix_block_rows, spec.process_grid.nprow) !=
                process_row ||
            OwnerCoord(global_i, problem.matrix_block_cols, spec.process_grid.npcol) !=
                process_col)
        {
          continue;
        }

        int local_row = LocalIndexFromGlobal(
            global_i, problem.matrix_block_rows, process_row, spec.process_grid.nprow);
        int local_col = LocalIndexFromGlobal(
            global_i, problem.matrix_block_cols, process_col, spec.process_grid.npcol);
        host_a[static_cast<size_t>(local_row + local_col * local_matrix_rows)] =
            static_cast<float>(global_i + 1);
      }
    }

    if (local_rhs_cols > 0)
    {
      for (int64_t global_i = 0; global_i < problem.rhs_rows; ++global_i)
      {
        if (OwnerCoord(global_i, problem.rhs_block_rows, spec.process_grid.nprow) !=
            process_row)
        {
          continue;
        }
        int local_row = LocalIndexFromGlobal(
            global_i, problem.rhs_block_rows, process_row, spec.process_grid.nprow);
        host_b[static_cast<size_t>(local_row)] = 1.0f;
      }
    }
    debug_log("after_synthetic_local_build");
    if (StopAtStage("after_synthetic_local_build"))
    {
      return fail("stopped after synthetic local build");
    }
  }

  if (!(use_input_buffers && spec.process_count > 1))
  {
    debug_log("before_cudaMemcpy_local_buffers");
    cuda_status = cudaMemcpy(d_a, host_a.data(), matrix_bytes, cudaMemcpyHostToDevice);
    if (cuda_status != cudaSuccess)
    {
      cudaFree(d_b);
      cudaFree(d_a);
      cusolverMpDestroyMatrixDesc(desc_b);
      cusolverMpDestroyMatrixDesc(desc_a);
      cusolverMpDestroyGrid(grid);
      cusolverMpDestroy(handle);
      cudaStreamDestroy(stream);
      ncclCommDestroy(comm);
      return fail("cudaMemcpy(A) failed: " + cuda_error_string(cuda_status));
    }

    cuda_status = cudaMemcpy(d_b, host_b.data(), rhs_bytes, cudaMemcpyHostToDevice);
    if (cuda_status != cudaSuccess)
    {
      cudaFree(d_b);
      cudaFree(d_a);
      cusolverMpDestroyMatrixDesc(desc_b);
      cusolverMpDestroyMatrixDesc(desc_a);
      cusolverMpDestroyGrid(grid);
      cusolverMpDestroy(handle);
      cudaStreamDestroy(stream);
      ncclCommDestroy(comm);
      return fail("cudaMemcpy(B) failed: " + cuda_error_string(cuda_status));
    }
    debug_log("after_cudaMemcpy_local_buffers");
    if (StopAtStage("after_device_buffer_copies"))
    {
      return fail("stopped after device buffer copies");
    }
  }

  size_t work_bytes = potrf_only ? potrf_workspace_device_bytes
                                 : std::max(potrf_workspace_device_bytes,
                                            potrs_workspace_device_bytes);
  if (work_bytes > 0)
  {
    debug_log("before_cudaMalloc_workspace");
    cuda_status = cudaMalloc(&d_work, work_bytes);
    if (cuda_status != cudaSuccess)
    {
      cudaFree(d_b);
      cudaFree(d_a);
      cusolverMpDestroyMatrixDesc(desc_b);
      cusolverMpDestroyMatrixDesc(desc_a);
      cusolverMpDestroyGrid(grid);
      cusolverMpDestroy(handle);
      cudaStreamDestroy(stream);
      ncclCommDestroy(comm);
      return fail("cudaMalloc(workspace) failed: " + cuda_error_string(cuda_status));
    }
    debug_log("after_cudaMalloc_workspace");
    if (StopAtStage("after_workspace_alloc"))
    {
      return fail("stopped after workspace alloc");
    }
  }

  size_t host_work_bytes = potrf_only ? potrf_workspace_host_bytes
                                      : std::max(potrf_workspace_host_bytes,
                                                 potrs_workspace_host_bytes);
  std::vector<char> h_work(host_work_bytes);

  debug_log("before_cudaMalloc_info");
  cuda_status = cudaMalloc(reinterpret_cast<void **>(&d_info), sizeof(int));
  if (cuda_status != cudaSuccess)
  {
    if (d_work != nullptr)
    {
      cudaFree(d_work);
    }
    cudaFree(d_b);
    cudaFree(d_a);
    cusolverMpDestroyMatrixDesc(desc_b);
    cusolverMpDestroyMatrixDesc(desc_a);
    cusolverMpDestroyGrid(grid);
    cusolverMpDestroy(handle);
    cudaStreamDestroy(stream);
    ncclCommDestroy(comm);
    return fail("cudaMalloc(info) failed: " + cuda_error_string(cuda_status));
  }
  debug_log("after_cudaMalloc_info");
  if (StopAtStage("after_info_alloc"))
  {
    return fail("stopped after info alloc");
  }

  debug_log("before_cusolverMpPotrf");
  cusolver_status = cusolverMpPotrf(
      handle, CUBLAS_FILL_MODE_LOWER, problem.matrix_rows, d_a, kSubmatrixStart,
      kSubmatrixStart, desc_a, CUDA_R_32F, d_work, potrf_workspace_device_bytes,
      h_work.empty() ? nullptr : h_work.data(), potrf_workspace_host_bytes, d_info);
  if (cusolver_status != CUSOLVER_STATUS_SUCCESS)
  {
    cudaFree(d_info);
    if (d_work != nullptr)
    {
      cudaFree(d_work);
    }
    cudaFree(d_b);
    cudaFree(d_a);
    cusolverMpDestroyMatrixDesc(desc_b);
    cusolverMpDestroyMatrixDesc(desc_a);
    cusolverMpDestroyGrid(grid);
    cusolverMpDestroy(handle);
    cudaStreamDestroy(stream);
    ncclCommDestroy(comm);
    return fail("cusolverMpPotrf failed with status " +
                cusolver_error_string(cusolver_status) + " [" + debug_prefix() + "]");
  }
  debug_log("after_cusolverMpPotrf");

  int potrf_info = 0;
  cuda_status = cudaMemcpy(&potrf_info, d_info, sizeof(int), cudaMemcpyDeviceToHost);
  if (cuda_status != cudaSuccess)
  {
    cudaFree(d_info);
    if (d_work != nullptr)
    {
      cudaFree(d_work);
    }
    cudaFree(d_b);
    cudaFree(d_a);
    cusolverMpDestroyMatrixDesc(desc_b);
    cusolverMpDestroyMatrixDesc(desc_a);
    cusolverMpDestroyGrid(grid);
    cusolverMpDestroy(handle);
    cudaStreamDestroy(stream);
    ncclCommDestroy(comm);
    return fail("cudaMemcpy(potrf info) failed: " + cuda_error_string(cuda_status));
  }
  if (potrf_info != 0)
  {
    cudaFree(d_info);
    if (d_work != nullptr)
    {
      cudaFree(d_work);
    }
    cudaFree(d_b);
    cudaFree(d_a);
    cusolverMpDestroyMatrixDesc(desc_b);
    cusolverMpDestroyMatrixDesc(desc_a);
    cusolverMpDestroyGrid(grid);
    cusolverMpDestroy(handle);
    cudaStreamDestroy(stream);
    ncclCommDestroy(comm);
    return fail("cusolverMpPotrf completed with info=" + std::to_string(potrf_info) +
                " [" + debug_prefix() + "]");
  }
  if (StopAtStage("after_potrf_info") || potrf_only)
  {
    cudaFree(d_info);
    if (d_work != nullptr)
    {
      cudaFree(d_work);
    }
    cudaFree(d_b);
    cudaFree(d_a);
    cusolverMpDestroyMatrixDesc(desc_b);
    cusolverMpDestroyMatrixDesc(desc_a);
    cusolverMpDestroyGrid(grid);
    cusolverMpDestroy(handle);
    cudaStreamDestroy(stream);
    ncclCommDestroy(comm);
    return fail(potrf_only ? "stopped after potrf-only run"
                           : "stopped after potrf info");
  }

  cusolver_status = cusolverMpPotrs(
      handle, CUBLAS_FILL_MODE_LOWER, problem.matrix_rows, problem.rhs_cols, d_a,
      kSubmatrixStart, kSubmatrixStart, desc_a, d_b, kSubmatrixStart,
      kSubmatrixStart, desc_b, CUDA_R_32F, d_work, potrs_workspace_device_bytes,
      h_work.empty() ? nullptr : h_work.data(), potrs_workspace_host_bytes, d_info);
  if (cusolver_status != CUSOLVER_STATUS_SUCCESS)
  {
    cudaFree(d_info);
    if (d_work != nullptr)
    {
      cudaFree(d_work);
    }
    cudaFree(d_b);
    cudaFree(d_a);
    cusolverMpDestroyMatrixDesc(desc_b);
    cusolverMpDestroyMatrixDesc(desc_a);
    cusolverMpDestroyGrid(grid);
    cusolverMpDestroy(handle);
    cudaStreamDestroy(stream);
    ncclCommDestroy(comm);
    return fail("cusolverMpPotrs failed with status " +
                cusolver_error_string(cusolver_status) + " [" + debug_prefix() + "]");
  }

  int potrs_info = 0;
  cuda_status = cudaMemcpy(&potrs_info, d_info, sizeof(int), cudaMemcpyDeviceToHost);
  if (cuda_status != cudaSuccess)
  {
    cudaFree(d_info);
    if (d_work != nullptr)
    {
      cudaFree(d_work);
    }
    cudaFree(d_b);
    cudaFree(d_a);
    cusolverMpDestroyMatrixDesc(desc_b);
    cusolverMpDestroyMatrixDesc(desc_a);
    cusolverMpDestroyGrid(grid);
    cusolverMpDestroy(handle);
    cudaStreamDestroy(stream);
    ncclCommDestroy(comm);
    return fail("cudaMemcpy(potrs info) failed: " + cuda_error_string(cuda_status));
  }
  if (potrs_info != 0)
  {
    cudaFree(d_info);
    if (d_work != nullptr)
    {
      cudaFree(d_work);
    }
    cudaFree(d_b);
    cudaFree(d_a);
    cusolverMpDestroyMatrixDesc(desc_b);
    cusolverMpDestroyMatrixDesc(desc_a);
    cusolverMpDestroyGrid(grid);
    cusolverMpDestroy(handle);
    cudaStreamDestroy(stream);
    ncclCommDestroy(comm);
    return fail("cusolverMpPotrs completed with info=" + std::to_string(potrs_info) +
                " [" + debug_prefix() + "]");
  }

  cuda_status = cudaMemcpy(host_b.data(), d_b, rhs_bytes, cudaMemcpyDeviceToHost);
  if (cuda_status != cudaSuccess)
  {
    cudaFree(d_info);
    if (d_work != nullptr)
    {
      cudaFree(d_work);
    }
    cudaFree(d_b);
    cudaFree(d_a);
    cusolverMpDestroyMatrixDesc(desc_b);
    cusolverMpDestroyMatrixDesc(desc_a);
    cusolverMpDestroyGrid(grid);
    cusolverMpDestroy(handle);
    cudaStreamDestroy(stream);
    ncclCommDestroy(comm);
    return fail("cudaMemcpy(solution) failed: " + cuda_error_string(cuda_status));
  }

  float max_abs_error = 0.0f;
  float residual_max_abs_error = 0.0f;
  if (!host_input_a_rowmajor.empty())
  {
    for (int64_t row = 0; row < problem.matrix_rows; ++row)
    {
      float accum = 0.0f;
      for (int64_t col = 0; col < problem.matrix_cols; ++col)
      {
        accum += host_input_a_rowmajor[static_cast<size_t>(row * problem.matrix_cols + col)] *
                 host_b[static_cast<size_t>(col)];
      }
      residual_max_abs_error = std::max(
          residual_max_abs_error,
          std::fabs(accum - host_input_b[static_cast<size_t>(row)]));
    }
  }
  else
  {
    if (local_rhs_cols > 0)
    {
      for (int64_t global_i = 0; global_i < problem.rhs_rows; ++global_i)
      {
        if (OwnerCoord(global_i, problem.rhs_block_rows, spec.process_grid.nprow) !=
            process_row)
        {
          continue;
        }
        int local_row = LocalIndexFromGlobal(
            global_i, problem.rhs_block_rows, process_row, spec.process_grid.nprow);
        float expected = 1.0f / static_cast<float>(global_i + 1);
        max_abs_error = std::max(
            max_abs_error,
            std::fabs(host_b[static_cast<size_t>(local_row)] - expected));
      }
    }
  }

  int nccl_version = 0;
  nccl_status = ncclGetVersion(&nccl_version);
  if (nccl_status != ncclSuccess)
  {
    cudaFree(d_info);
    if (d_work != nullptr)
    {
      cudaFree(d_work);
    }
    cudaFree(d_b);
    cudaFree(d_a);
    cusolverMpDestroyMatrixDesc(desc_b);
    cusolverMpDestroyMatrixDesc(desc_a);
    cusolverMpDestroyGrid(grid);
    cusolverMpDestroy(handle);
    cudaStreamDestroy(stream);
    ncclCommDestroy(comm);
    return fail("ncclGetVersion failed: " + nccl_error_string(nccl_status));
  }

  cudaFree(d_info);
  if (d_work != nullptr)
  {
    cudaFree(d_work);
  }
  cudaFree(d_b);
  cudaFree(d_a);
  cusolverMpDestroyMatrixDesc(desc_b);
  cusolverMpDestroyMatrixDesc(desc_a);
  cusolverMpDestroyGrid(grid);
  cusolverMpDestroy(handle);
  cudaStreamDestroy(stream);
  ncclCommDestroy(comm);

  return CuSolverMpRuntimeProbeResult{
      .cuda_device_id = device_id,
      .nccl_version = nccl_version,
      .cusolvermp_version = cusolvermp_version,
      .local_matrix_rows = local_matrix_rows,
      .local_matrix_cols = local_matrix_cols,
      .local_rhs_rows = local_rhs_rows,
      .local_rhs_cols = local_rhs_cols,
      .potrf_workspace_device_bytes = potrf_workspace_device_bytes,
      .potrf_workspace_host_bytes = potrf_workspace_host_bytes,
      .potrs_workspace_device_bytes = potrs_workspace_device_bytes,
      .potrs_workspace_host_bytes = potrs_workspace_host_bytes,
      .potrf_info = potrf_info,
      .potrs_info = potrs_info,
      .solution_max_abs_error = max_abs_error,
      .residual_max_abs_error = residual_max_abs_error,
      .solved_rhs = host_b,
  };
#endif
}

} // namespace jaxmg
