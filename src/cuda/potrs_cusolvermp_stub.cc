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
  int64_t rhs_block_rows;
  int64_t rhs_block_cols;
};

XLA_FFI_REGISTER_STRUCT_ATTR_DECODING(
    CuSolverMpPotrsStubAttrs, ffi::StructMember<int64_t>("T_A"),
    ffi::StructMember<std::string_view>("context_json"),
    ffi::StructMember<int64_t>("process_rank"),
    ffi::StructMember<int64_t>("process_count"),
    ffi::StructMember<int64_t>("local_device_count"),
    ffi::StructMember<int64_t>("local_device_index"),
    ffi::StructMember<int64_t>("global_device_count"),
    ffi::StructMember<int64_t>("process_grid_nprow"),
    ffi::StructMember<int64_t>("process_grid_npcol"),
    ffi::StructMember<int64_t>("matrix_block_rows"),
    ffi::StructMember<int64_t>("matrix_block_cols"),
    ffi::StructMember<int64_t>("rhs_block_rows"),
    ffi::StructMember<int64_t>("rhs_block_cols"));

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
