#include "include/cusolvermp_context.h"

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

} // namespace jaxmg
