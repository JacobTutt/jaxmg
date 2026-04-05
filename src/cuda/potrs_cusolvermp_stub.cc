// C++
#include <cstdint>
#include <string>
// Abseil
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
// Jaxlib
#include "jaxlib/ffi_helpers.h"
#include "jaxlib/gpu/vendor.h"
// JAXMg
#include "include/cusolvermp_context.h"
// XLA
#include "xla/ffi/api/ffi.h"

namespace jax
{
namespace JAX_GPU_NAMESPACE
{
namespace ffi = ::xla::ffi;

struct CuSolverMpPotrsStubAttrs
{
  int64_t tile_size;
  std::string_view context_json;
  int64_t process_rank;
  int64_t process_count;
  int64_t local_device_count;
  int64_t local_device_index;
  int64_t global_device_count;
  int64_t process_grid_nprow;
  int64_t process_grid_npcol;
  int64_t matrix_block_rows;
  int64_t matrix_block_cols;
  int64_t matrix_padded_rows;
  int64_t matrix_padded_cols;
  int64_t rhs_block_rows;
  int64_t rhs_block_cols;
  int64_t rhs_padded_rows;
  int64_t rhs_padded_cols;
};

} // namespace JAX_GPU_NAMESPACE
} // namespace jax

XLA_FFI_REGISTER_STRUCT_ATTR_DECODING(
    ::jax::JAX_GPU_NAMESPACE::CuSolverMpPotrsStubAttrs,
    StructMember<int64_t>("T_A"),
    StructMember<std::string_view>("context_json"),
    StructMember<int64_t>("process_rank"),
    StructMember<int64_t>("process_count"),
    StructMember<int64_t>("local_device_count"),
    StructMember<int64_t>("local_device_index"),
    StructMember<int64_t>("global_device_count"),
    StructMember<int64_t>("process_grid_nprow"),
    StructMember<int64_t>("process_grid_npcol"),
    StructMember<int64_t>("matrix_block_rows"),
    StructMember<int64_t>("matrix_block_cols"),
    StructMember<int64_t>("matrix_padded_rows"),
    StructMember<int64_t>("matrix_padded_cols"),
    StructMember<int64_t>("rhs_block_rows"),
    StructMember<int64_t>("rhs_block_cols"),
    StructMember<int64_t>("rhs_padded_rows"),
    StructMember<int64_t>("rhs_padded_cols"));

namespace jax
{
namespace JAX_GPU_NAMESPACE
{

ffi::Error PotrsCuSolverMpDispatch(
    gpuStream_t stream, ffi::ScratchAllocator scratch, ffi::AnyBuffer a,
    ffi::AnyBuffer b, CuSolverMpPotrsStubAttrs attrs,
    ffi::Result<ffi::AnyBuffer> out_a, ffi::Result<ffi::AnyBuffer> out_b,
    ffi::Result<ffi::Buffer<ffi::S32>> status)
{
  (void)stream;
  (void)scratch;
  (void)a;
  (void)b;
  (void)out_a;
  (void)out_b;
  (void)status;

  jaxmg::CuSolverMpContextSpec spec{
      .process_rank = static_cast<int>(attrs.process_rank),
      .process_count = static_cast<int>(attrs.process_count),
      .local_device_count = static_cast<int>(attrs.local_device_count),
      .local_device_index = static_cast<int>(attrs.local_device_index),
      .global_device_count = static_cast<int>(attrs.global_device_count),
      .process_grid =
          {
              .nprow = static_cast<int>(attrs.process_grid_nprow),
              .npcol = static_cast<int>(attrs.process_grid_npcol),
          },
      .mb = static_cast<int>(attrs.matrix_block_rows),
      .nb = static_cast<int>(attrs.matrix_block_cols),
      .requires_mpi = true,
      .requires_nccl = true,
  };

  auto issues = jaxmg::CuSolverMpPotrsStubContractIssues(
      spec, static_cast<int>(attrs.tile_size),
      static_cast<int>(attrs.matrix_block_rows),
      static_cast<int>(attrs.matrix_block_cols),
      static_cast<int>(attrs.rhs_block_rows),
      static_cast<int>(attrs.rhs_block_cols));
  if (!issues.empty())
  {
    return ffi::Error::InvalidArgument(
        absl::StrFormat("potrs_cusolvermp stub contract validation failed: %s",
                        absl::StrJoin(issues, " | ")));
  }

  constexpr size_t kPreviewChars = 160;
  std::string preview =
      attrs.context_json.size() <= kPreviewChars
          ? std::string(attrs.context_json)
          : absl::StrFormat("%s...", attrs.context_json.substr(0, kPreviewChars));

  jaxmg::CuSolverMpPotrsProblemSpec problem{
      .matrix_rows = attrs.matrix_padded_rows,
      .matrix_cols = attrs.matrix_padded_cols,
      .rhs_rows = attrs.rhs_padded_rows,
      .rhs_cols = attrs.rhs_padded_cols,
      .matrix_block_rows = attrs.matrix_block_rows,
      .matrix_block_cols = attrs.matrix_block_cols,
      .rhs_block_rows = attrs.rhs_block_rows,
      .rhs_block_cols = attrs.rhs_block_cols,
  };

#if defined(JAXMG_HAVE_CUSOLVERMP) && defined(JAXMG_HAVE_NCCL)
  std::string probe_error;
  auto probe = jaxmg::ProbeCuSolverMpRuntime(
      spec, problem, a.untyped_data(), b.untyped_data(), &probe_error);
  if (!probe.has_value())
  {
    return ffi::Error::InvalidArgument(
        absl::StrFormat(
            "potrs_cusolvermp reached native code but the real cuSOLVERMp "
            "runtime probe failed: %s",
            probe_error));
  }

  return ffi::Error::InvalidArgument(
      absl::StrFormat(
          "potrs_cusolvermp reached native code and completed a real "
          "cuSOLVERMp runtime probe with device=%d, nccl_version=%d, "
          "cusolvermp_version=%d, T_A=%d, rank=%d/%d, process_grid=(%d,%d), "
          "matrix_block=(%d,%d), rhs_block=(%d,%d), local_matrix=(%d,%d), "
          "local_rhs=(%d,%d), potrf_workspace=(device=%zu,host=%zu), "
          "potrs_workspace=(device=%zu,host=%zu), potrf_info=%d, "
          "potrs_info=%d, solution_max_abs_error=%g, residual_max_abs_error=%g, "
          "context_json=%s, but the solver execution path is not implemented yet.",
          probe->cuda_device_id, probe->nccl_version, probe->cusolvermp_version,
          static_cast<int>(attrs.tile_size), static_cast<int>(attrs.process_rank),
          static_cast<int>(attrs.process_count),
          static_cast<int>(attrs.process_grid_nprow),
          static_cast<int>(attrs.process_grid_npcol),
          static_cast<int>(attrs.matrix_block_rows),
          static_cast<int>(attrs.matrix_block_cols),
          static_cast<int>(attrs.rhs_block_rows),
          static_cast<int>(attrs.rhs_block_cols), probe->local_matrix_rows,
          probe->local_matrix_cols, probe->local_rhs_rows, probe->local_rhs_cols,
          probe->potrf_workspace_device_bytes, probe->potrf_workspace_host_bytes,
          probe->potrs_workspace_device_bytes, probe->potrs_workspace_host_bytes,
          probe->potrf_info, probe->potrs_info, probe->solution_max_abs_error,
          probe->residual_max_abs_error, preview));
#else
  return ffi::Error::InvalidArgument(
      absl::StrFormat(
          "potrs_cusolvermp stub reached native code with T_A=%d, "
          "rank=%d/%d, process_grid=(%d,%d), matrix_block=(%d,%d), "
          "rhs_block=(%d,%d), "
          "context_json=%s, but the real cuSOLVERMp implementation is not "
          "linked yet.",
          static_cast<int>(attrs.tile_size), static_cast<int>(attrs.process_rank),
          static_cast<int>(attrs.process_count),
          static_cast<int>(attrs.process_grid_nprow),
          static_cast<int>(attrs.process_grid_npcol),
          static_cast<int>(attrs.matrix_block_rows),
          static_cast<int>(attrs.matrix_block_cols),
          static_cast<int>(attrs.rhs_block_rows),
          static_cast<int>(attrs.rhs_block_cols), preview));
#endif
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(PotrsCuSolverMpFFI, PotrsCuSolverMpDispatch,
                              ffi::Ffi::Bind()
                                  .Ctx<ffi::PlatformStream<gpuStream_t>>()
                                  .Ctx<ffi::ScratchAllocator>()
                                  .Arg<ffi::AnyBuffer>() // A
                                  .Arg<ffi::AnyBuffer>() // b
                                  .Attrs<CuSolverMpPotrsStubAttrs>()
                                  .Ret<ffi::AnyBuffer>() // A_out
                                  .Ret<ffi::AnyBuffer>() // b_out
                                  .Ret<ffi::Buffer<ffi::S32>>() // status
);

} // namespace JAX_GPU_NAMESPACE
} // namespace jax
