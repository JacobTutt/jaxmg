#include "absl/status/status.h"
#include "absl/strings/str_format.h"

#include "cusolvermp_context.h"

namespace jax
{
    namespace JAX_GPU_NAMESPACE
    {
        CusolverMpContextConfig MakeCusolverMpContextConfig(
            int64_t process_count,
            int64_t process_index,
            int64_t local_device_count,
            int64_t nprow,
            int64_t npcol)
        {
            CusolverMpContextConfig config;
            config.process_count = process_count;
            config.process_index = process_index;
            config.local_device_count = local_device_count;
            config.nprow = nprow;
            config.npcol = npcol;
            config.process_row = process_index / npcol;
            config.process_col = process_index % npcol;
            return config;
        }

        std::string DescribeCusolverMpContext(const CusolverMpContextConfig &config)
        {
            return absl::StrFormat(
                "world=%d rank=%d local_devices=%d grid=%dx%d coords=(%d,%d)",
                config.process_count,
                config.process_index,
                config.local_device_count,
                config.nprow,
                config.npcol,
                config.process_row,
                config.process_col);
        }
    }
}
