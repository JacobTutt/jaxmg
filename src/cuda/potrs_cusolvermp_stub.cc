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

template <typename T>
ffi::ErrorOr<T> GetAttrOrError(const ffi::Dictionary &attrs, std::string_view name)
{
  return attrs.get<T>(name);
}

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

  auto tile_size = GetAttrOrError<int64_t>(attrs, "T_A");
  if (tile_size.has_error())
  {
    return tile_size.error();
  }

  auto context_json = GetAttrOrError<std::string_view>(attrs, "context_json");
  if (context_json.has_error())
  {
    return context_json.error();
  }

  auto process_rank = GetAttrOrError<int64_t>(attrs, "process_rank");
  if (process_rank.has_error())
  {
    return process_rank.error();
  }

  auto process_count = GetAttrOrError<int64_t>(attrs, "process_count");
  if (process_count.has_error())
  {
    return process_count.error();
  }

  auto local_device_count = GetAttrOrError<int64_t>(attrs, "local_device_count");
  if (local_device_count.has_error())
  {
    return local_device_count.error();
  }

  auto local_device_index = GetAttrOrError<int64_t>(attrs, "local_device_index");
  if (local_device_index.has_error())
  {
    return local_device_index.error();
  }

  auto global_device_count = GetAttrOrError<int64_t>(attrs, "global_device_count");
  if (global_device_count.has_error())
  {
    return global_device_count.error();
  }

  auto process_grid_nprow = GetAttrOrError<int64_t>(attrs, "process_grid_nprow");
  if (process_grid_nprow.has_error())
  {
    return process_grid_nprow.error();
  }

  auto process_grid_npcol = GetAttrOrError<int64_t>(attrs, "process_grid_npcol");
  if (process_grid_npcol.has_error())
  {
    return process_grid_npcol.error();
  }

  auto matrix_block_rows = GetAttrOrError<int64_t>(attrs, "matrix_block_rows");
  if (matrix_block_rows.has_error())
  {
    return matrix_block_rows.error();
  }

  auto matrix_block_cols = GetAttrOrError<int64_t>(attrs, "matrix_block_cols");
  if (matrix_block_cols.has_error())
  {
    return matrix_block_cols.error();
  }

  auto rhs_block_rows = GetAttrOrError<int64_t>(attrs, "rhs_block_rows");
  if (rhs_block_rows.has_error())
  {
    return rhs_block_rows.error();
  }

  auto rhs_block_cols = GetAttrOrError<int64_t>(attrs, "rhs_block_cols");
  if (rhs_block_cols.has_error())
  {
    return rhs_block_cols.error();
  }

  if (*process_count <= 0 || *global_device_count <= 0)
  {
    return ffi::Error::InvalidArgument(
        "potrs_cusolvermp stub expected positive process/device counts.");
  }
  if (*process_rank < 0 || *process_rank >= *process_count)
  {
    return ffi::Error::InvalidArgument(
        "potrs_cusolvermp stub expected 0 <= process_rank < process_count.");
  }
  if (*local_device_count != 1 || *local_device_index != 0)
  {
    return ffi::Error::InvalidArgument(
        "potrs_cusolvermp stub expected one process per GPU with local_device_index == 0.");
  }
  if (*process_count != *global_device_count)
  {
    return ffi::Error::InvalidArgument(
        "potrs_cusolvermp stub expected process_count == global_device_count.");
  }
  if (*process_grid_nprow <= 0 || *process_grid_npcol <= 0)
  {
    return ffi::Error::InvalidArgument(
        "potrs_cusolvermp stub expected positive process-grid dimensions.");
  }
  if (*process_grid_nprow * *process_grid_npcol != *global_device_count)
  {
    return ffi::Error::InvalidArgument(
        "potrs_cusolvermp stub expected nprow * npcol == global_device_count.");
  }
  if (*matrix_block_rows != *tile_size || *matrix_block_cols != *tile_size)
  {
    return ffi::Error::InvalidArgument(
        "potrs_cusolvermp stub expected matrix block shape to match T_A.");
  }
  if (*rhs_block_cols != 1)
  {
    return ffi::Error::InvalidArgument(
        "potrs_cusolvermp stub expected the initial cuSOLVERMp path to use NRHS=1.");
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
          "rhs_block=(%d,%d), "
          "context_json=%s, but the real cuSOLVERMp implementation is not "
          "linked yet.",
          static_cast<int>(*tile_size), static_cast<int>(*process_rank),
          static_cast<int>(*process_count), static_cast<int>(*process_grid_nprow),
          static_cast<int>(*process_grid_npcol), static_cast<int>(*matrix_block_rows),
          static_cast<int>(*matrix_block_cols), static_cast<int>(*rhs_block_rows),
          static_cast<int>(*rhs_block_cols), preview));
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
