// C++
#include <cstdint>
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

ffi::Error SyevdCuSolverMpDispatch(
    gpuStream_t stream, ffi::ScratchAllocator scratch, ffi::AnyBuffer a,
    int64_t tile_size, ffi::Result<ffi::AnyBuffer> eigenvalues,
    ffi::Result<ffi::AnyBuffer> V, ffi::Result<ffi::Buffer<ffi::S32>> status)
{
  (void)stream;
  (void)scratch;
  (void)a;
  (void)tile_size;
  (void)eigenvalues;
  (void)V;
  (void)status;

  return ffi::Error::InvalidArgument(
      "syevd_cusolvermp stub reached native code, but the real cuSOLVERMp implementation is not linked yet.");
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(SyevdCuSolverMpFFI, SyevdCuSolverMpDispatch,
                              ffi::Ffi::Bind()
                                  .Ctx<ffi::PlatformStream<gpuStream_t>>()
                                  .Ctx<ffi::ScratchAllocator>()
                                  .Arg<ffi::AnyBuffer>()        // A
                                  .Attr<int64_t>("T_A")         // tile size
                                  .Ret<ffi::AnyBuffer>()        // eigenvalues
                                  .Ret<ffi::AnyBuffer>()        // V
                                  .Ret<ffi::Buffer<ffi::S32>>() // status
);

} // namespace JAX_GPU_NAMESPACE
} // namespace jax
