#ifndef JAXMG_CUSOLVERMP_PACK_H_
#define JAXMG_CUSOLVERMP_PACK_H_

#include <cstddef>
#include <cstdint>
#include <string>

struct CUstream_st;
using cudaStream_t = CUstream_st *;

struct ncclComm;
using ncclComm_t = ncclComm *;

namespace jaxmg
{

struct CuSolverMpGpuPackResult
{
  bool applied;
  std::string error_message;
};

CuSolverMpGpuPackResult TryPackCuSolverMpInputsGpu(
    int scalar_type_tag, int process_rank, int process_count, int nprow,
    int npcol, int64_t matrix_rows, int64_t matrix_cols,
    int64_t rhs_rows, int64_t rhs_cols, int matrix_block_rows,
    int matrix_block_cols, int rhs_block_rows, int rhs_block_cols,
    const void *input_a, size_t input_a_elements, const void *input_b,
    size_t input_b_elements, void *d_a, int64_t local_matrix_rows,
    int64_t local_matrix_cols, void *d_b, int64_t local_rhs_rows,
    int64_t local_rhs_cols, ncclComm_t comm, cudaStream_t stream);

} // namespace jaxmg

#endif // JAXMG_CUSOLVERMP_PACK_H_
