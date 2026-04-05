#ifndef JAXMG_CUSOLVERMP_CONTEXT_H_
#define JAXMG_CUSOLVERMP_CONTEXT_H_

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace jaxmg
{

struct CuSolverMpProcessGrid
{
  int nprow;
  int npcol;
};

struct CuSolverMpContextSpec
{
  int process_rank;
  int process_count;
  int local_device_count;
  int local_device_index;
  int global_device_count;
  CuSolverMpProcessGrid process_grid;
  int mb;
  int nb;
  bool requires_mpi;
  bool requires_nccl;
};

struct CuSolverMpContextPlan
{
  std::string ffi_target_name;
  std::string library_name;
  std::string symbol_name;
  std::string distribution;
  std::string local_layout;
  std::vector<std::string> unsupported_operations;
};

struct CuSolverMpPotrsProblemSpec
{
  int64_t matrix_rows;
  int64_t matrix_cols;
  int64_t rhs_rows;
  int64_t rhs_cols;
  int64_t matrix_block_rows;
  int64_t matrix_block_cols;
  int64_t rhs_block_rows;
  int64_t rhs_block_cols;
};

struct CuSolverMpRuntimeProbeResult
{
  int cuda_device_id;
  int nccl_version;
  int cusolvermp_version;
  int64_t local_matrix_rows;
  int64_t local_matrix_cols;
  int64_t local_rhs_rows;
  int64_t local_rhs_cols;
};

CuSolverMpContextPlan BuildCuSolverMpContextPlan();

bool CuSolverMpContextSpecIsLaunchReady(const CuSolverMpContextSpec &spec);

std::vector<std::string> CuSolverMpContextLaunchIssues(
    const CuSolverMpContextSpec &spec);

std::vector<std::string> CuSolverMpPotrsStubContractIssues(
    const CuSolverMpContextSpec &spec, int tile_size, int matrix_block_rows,
    int matrix_block_cols, int rhs_block_rows, int rhs_block_cols);

std::optional<CuSolverMpRuntimeProbeResult> ProbeCuSolverMpRuntime(
    const CuSolverMpContextSpec &spec, const CuSolverMpPotrsProblemSpec &problem,
    std::string *error_message);

} // namespace jaxmg

#endif // JAXMG_CUSOLVERMP_CONTEXT_H_
