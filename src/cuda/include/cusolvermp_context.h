#pragma once

#include <cstdint>
#include <string>

namespace jax
{
    namespace JAX_GPU_NAMESPACE
    {
        struct CusolverMpContextConfig
        {
            int64_t process_count = 0;
            int64_t process_index = 0;
            int64_t local_device_count = 0;
            int64_t process_row = 0;
            int64_t process_col = 0;
            int64_t nprow = 0;
            int64_t npcol = 0;
        };

        struct CusolverMpContextHandles
        {
            void *mpi_comm = nullptr;
            void *nccl_comm = nullptr;
            void *cuda_stream = nullptr;
            void *cusolvermp_handle = nullptr;
            void *device_grid = nullptr;
        };

        struct CusolverMpContext
        {
            CusolverMpContextConfig config;
            CusolverMpContextHandles handles;
        };

        CusolverMpContextConfig MakeCusolverMpContextConfig(
            int64_t process_count,
            int64_t process_index,
            int64_t local_device_count,
            int64_t nprow,
            int64_t npcol);

        std::string DescribeCusolverMpContext(const CusolverMpContextConfig &config);
    }
}
