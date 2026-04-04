// C++
#include <cstdint>
#include <string>
// Abseil
#include "absl/strings/str_format.h"
// Jaxlib
#include "jaxlib/ffi_helpers.h"
#include "jaxlib/gpu/vendor.h"
// XLA
#include "xla/ffi/api/ffi.h"

namespace jax
{
namespace JAX_GPU_NAMESPACE
{
namespace ffi = ::xla::ffi;

ffi::Error PotrsCuSolverMpDispatch(
    gpuStream_t stream, ffi::ScratchAllocator scratch, ffi::AnyBuffer a,
    ffi::AnyBuffer b, ffi::Dictionary attrs,
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

  auto tile_size = attrs.get<int64_t>("T_A");
  if (tile_size.has_error())
  {
    return tile_size.error();
  }

  auto context_json = attrs.get<std::string_view>("context_json");
  if (context_json.has_error())
  {
    return context_json.error();
  }

  auto process_rank = attrs.get<int64_t>("process_rank");
  if (process_rank.has_error())
  {
    return process_rank.error();
  }

  auto process_count = attrs.get<int64_t>("process_count");
  if (process_count.has_error())
  {
    return process_count.error();
  }

  auto process_grid_nprow = attrs.get<int64_t>("process_grid_nprow");
  if (process_grid_nprow.has_error())
  {
    return process_grid_nprow.error();
  }

  auto process_grid_npcol = attrs.get<int64_t>("process_grid_npcol");
  if (process_grid_npcol.has_error())
  {
    return process_grid_npcol.error();
  }

  auto matrix_block_rows = attrs.get<int64_t>("matrix_block_rows");
  if (matrix_block_rows.has_error())
  {
    return matrix_block_rows.error();
  }

  auto matrix_block_cols = attrs.get<int64_t>("matrix_block_cols");
  if (matrix_block_cols.has_error())
  {
    return matrix_block_cols.error();
  }

  constexpr size_t kPreviewChars = 160;
  std::string preview =
      context_json->size() <= kPreviewChars
          ? std::string(*context_json)
          : absl::StrFormat("%s...", context_json->substr(0, kPreviewChars));

  return ffi::Error::InvalidArgument(
      absl::StrFormat(
          "potrs_cusolvermp stub reached native code with T_A=%d, "
          "rank=%d/%d, process_grid=(%d,%d), matrix_block=(%d,%d), "
          "context_json=%s, but the real cuSOLVERMp implementation is not "
          "linked yet.",
          static_cast<int>(*tile_size), static_cast<int>(*process_rank),
          static_cast<int>(*process_count), static_cast<int>(*process_grid_nprow),
          static_cast<int>(*process_grid_npcol), static_cast<int>(*matrix_block_rows),
          static_cast<int>(*matrix_block_cols), preview));
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(PotrsCuSolverMpFFI, PotrsCuSolverMpDispatch,
                              ffi::Ffi::Bind()
                                  .Ctx<ffi::PlatformStream<gpuStream_t>>()
                                  .Ctx<ffi::ScratchAllocator>()
                                  .Arg<ffi::AnyBuffer>() // A
                                  .Arg<ffi::AnyBuffer>() // b
                                  .Attrs<ffi::Dictionary>()
                                  .Ret<ffi::AnyBuffer>() // A_out
                                  .Ret<ffi::AnyBuffer>() // b_out
                                  .Ret<ffi::Buffer<ffi::S32>>() // status
);

} // namespace JAX_GPU_NAMESPACE
} // namespace jax
