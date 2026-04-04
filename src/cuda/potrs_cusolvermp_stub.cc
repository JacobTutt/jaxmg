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
    ffi::AnyBuffer b, int64_t tile_size, ffi::Result<ffi::AnyBuffer> out_a,
    ffi::Result<ffi::AnyBuffer> out_b, ffi::Result<ffi::Buffer<ffi::S32>> status)
{
  (void)stream;
  (void)scratch;
  (void)a;
  (void)b;
  (void)tile_size;
  (void)out_a;
  (void)out_b;
  (void)status;

  return ffi::Error::InvalidArgument(
      "potrs_cusolvermp stub reached native code, but the real cuSOLVERMp implementation is not linked yet.");
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(PotrsCuSolverMpFFI, PotrsCuSolverMpDispatch,
                              ffi::Ffi::Bind()
                                  .Ctx<ffi::PlatformStream<gpuStream_t>>()
                                  .Ctx<ffi::ScratchAllocator>()
                                  .Arg<ffi::AnyBuffer>()        // A
                                  .Arg<ffi::AnyBuffer>()        // b
                                  .Attr<int64_t>("T_A")         // tile size
                                  .Ret<ffi::AnyBuffer>()        // A_out
                                  .Ret<ffi::AnyBuffer>()        // b_out
                                  .Ret<ffi::Buffer<ffi::S32>>() // status
);

} // namespace JAX_GPU_NAMESPACE
} // namespace jax
