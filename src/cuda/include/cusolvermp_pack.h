#ifndef JAXMG_CUSOLVERMP_PACK_H_
#define JAXMG_CUSOLVERMP_PACK_H_

#include "include/cusolvermp_context.h"

#include <string>

#if defined(JAXMG_HAVE_CUSOLVERMP) && defined(JAXMG_HAVE_NCCL)
#include <cuda_runtime_api.h>
#include <nccl.h>
#endif

namespace jaxmg
{

struct CuSolverMpGpuPackResult
{
  bool applied;
  std::string error_message;
};

#if defined(JAXMG_HAVE_CUSOLVERMP) && defined(JAXMG_HAVE_NCCL)
CuSolverMpGpuPackResult TryPackCuSolverMpInputsGpu(
    CuSolverMpScalarType scalar_type, const CuSolverMpContextSpec &spec,
    const CuSolverMpPotrsProblemSpec &problem, const void *input_a,
    size_t input_a_elements, const void *input_b, size_t input_b_elements,
    void *d_a, int64_t local_matrix_rows, int64_t local_matrix_cols, void *d_b,
    int64_t local_rhs_rows, int64_t local_rhs_cols, ncclComm_t comm,
    cudaStream_t stream);
#endif

} // namespace jaxmg

#endif // JAXMG_CUSOLVERMP_PACK_H_
