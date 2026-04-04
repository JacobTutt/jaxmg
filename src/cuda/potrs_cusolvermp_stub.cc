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
    ffi::AnyBuffer b, int64_t tile_size, std::string_view context_json,
    ffi::Result<ffi::AnyBuffer> out_a, ffi::Result<ffi::AnyBuffer> out_b,
    ffi::Result<ffi::Buffer<ffi::S32>> status)
{
  (void)stream;
  (void)scratch;
  (void)a;
  (void)b;
  (void)tile_size;
  (void)out_a;
  (void)out_b;
  (void)status;

  constexpr size_t kPreviewChars = 160;
  std::string preview =
      context_json.size() <= kPreviewChars
          ? std::string(context_json)
          : absl::StrFormat("%s...", context_json.substr(0, kPreviewChars));

  return ffi::Error::InvalidArgument(
      absl::StrFormat(
          "potrs_cusolvermp stub reached native code with T_A=%d and "
          "context_json=%s, but the real cuSOLVERMp implementation is not "
          "linked yet.",
          static_cast<int>(tile_size), preview));
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(PotrsCuSolverMpFFI, PotrsCuSolverMpDispatch,
                              ffi::Ffi::Bind()
                                  .Ctx<ffi::PlatformStream<gpuStream_t>>()
                                  .Ctx<ffi::ScratchAllocator>()
                                  .Arg<ffi::AnyBuffer>()        // A
                                  .Arg<ffi::AnyBuffer>()        // b
                                  .Attr<int64_t>("T_A")         // tile size
                                  .Attr<std::string_view>("context_json")
                                  .Ret<ffi::AnyBuffer>()        // A_out
                                  .Ret<ffi::AnyBuffer>()        // b_out
                                  .Ret<ffi::Buffer<ffi::S32>>() // status
);

} // namespace JAX_GPU_NAMESPACE
} // namespace jax
