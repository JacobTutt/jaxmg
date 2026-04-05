cmake_minimum_required(VERSION 3.25.2)
project(JaxSolve)

# Get CPM
set(CPM_DOWNLOAD_VERSION 0.42.0)
set(CPM_DOWNLOAD_LOCATION "${CMAKE_BINARY_DIR}/cmake/CPM_${CPM_DOWNLOAD_VERSION}.cmake")
if(NOT (EXISTS ${CPM_DOWNLOAD_LOCATION}))
    message(STATUS "Downloading CPM.cmake")
    file(DOWNLOAD https://github.com/TheLartians/CPM.cmake/releases/download/v${CPM_DOWNLOAD_VERSION}/CPM.cmake ${CPM_DOWNLOAD_LOCATION})
endif()
include(${CPM_DOWNLOAD_LOCATION})

# Abseil requires C++20
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}" ${CMAKE_MODULE_PATH})

# Static linking against C++20 standard library
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -static-libgcc -static-libstdc++")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -static-libgcc -static-libstdc++")
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -static-libgcc -static-libstdc++")
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libc++ -static-libgcc")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -stdlib=libc++ -static-libgcc")
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -stdlib=libc++ -static-libgcc")
endif()

# Abseill headers
CPMAddPackage(
    NAME absl
    GIT_REPOSITORY https://github.com/abseil/abseil-cpp
    GIT_TAG 20250512.1
)

# Import CUDAToolkit (Requires CUDA_HOME to be set)
find_package(CUDAToolkit REQUIRED)
find_library(CUSOLVERMG_LIB cusolverMg
    HINTS "$ENV{CONDA_PREFIX}/lib" "${CUDAToolkit_LIBRARY_DIR}" "${CUDA_TOOLKIT_ROOT_DIR}/lib64" "/usr/local/cuda/lib64")
find_path(CUSOLVERMP_INCLUDE_DIR cusolverMp.h
    HINTS "$ENV{CONDA_PREFIX}/include" "${CUDAToolkit_INCLUDE_DIRS}" "/usr/local/cuda/include")
find_library(CUSOLVERMP_LIB cusolverMp
    HINTS "$ENV{CONDA_PREFIX}/lib" "${CUDAToolkit_LIBRARY_DIR}" "${CUDA_TOOLKIT_ROOT_DIR}/lib64" "/usr/local/cuda/lib64")
find_path(CUDA_MATH_INCLUDE_DIR cublas_v2.h
    HINTS
        "/opt/nvidia/hpc_sdk/Linux_aarch64/24.11/math_libs/${CUDAToolkit_VERSION_MAJOR}.${CUDAToolkit_VERSION_MINOR}/targets/sbsa-linux/include"
        "/opt/nvidia/hpc_sdk/Linux_aarch64/24.11/math_libs/12.6/targets/sbsa-linux/include"
        "$ENV{CONDA_PREFIX}/lib/python3.11/site-packages/nvidia/cublas/include")
find_path(NCCL_INCLUDE_DIR nccl.h
    HINTS "$ENV{CONDA_PREFIX}/include" "/usr/local/include")
find_library(NCCL_LIB nccl
    HINTS "$ENV{CONDA_PREFIX}/lib" "/usr/local/lib" "/usr/local/cuda/lib64")
set(HAVE_CUSOLVERMP OFF)
if(CUSOLVERMP_INCLUDE_DIR AND CUSOLVERMP_LIB AND NCCL_INCLUDE_DIR AND NCCL_LIB)
    set(HAVE_CUSOLVERMP ON)
endif()
find_package(CUDNN)
message(STATUS "CUDA include directory: $ENV{CUDA_ROOT}")
if(CUDNN_FOUND)
    message(STATUS "CUDNN include directory: $ENV{CUDNN_ROOT}")
else()
    message(STATUS "CUDNN not found; Mg targets may not build in this configuration")
endif()
if(HAVE_CUSOLVERMP)
    message(STATUS "cuSOLVERMp include directory: ${CUSOLVERMP_INCLUDE_DIR}")
    message(STATUS "cuSOLVERMp library: ${CUSOLVERMP_LIB}")
    message(STATUS "NCCL include directory: ${NCCL_INCLUDE_DIR}")
    message(STATUS "NCCL library: ${NCCL_LIB}")
else()
    message(STATUS "cuSOLVERMp/NCCL not found; building mp targets in stub-only mode")
endif()
if(CUSOLVERMG_LIB)
    message(STATUS "cuSolverMg library: ${CUSOLVERMG_LIB}")
else()
    message(STATUS "cuSolverMg library not found; Mg targets may not build in this configuration")
    set(CUSOLVERMG_LIB "")
endif()
# Enable CUDA 
enable_language(CUDA)
## Set backend install subdirectory based on CUDA major version
string(REGEX MATCH "^([0-9]+)\.([0-9]+)" _cuda_ver_match "${CMAKE_CUDA_COMPILER_VERSION}")
set(CUDA_MAJOR "${CMAKE_MATCH_1}")
set(BIN_DST src/jaxmg/cu${CUDA_MAJOR})
message(STATUS "Library install destination: ${CMAKE_SOURCE_DIR}/${BIN_DST}")

# Ensure CUDA uses C++20 (required by this project)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Get Jaxlib
# Allow overriding the JAX version via -DJAX_VERSION=0.x.y or env JAX_VERSION; default to 0.6.2
if(NOT DEFINED JAX_VERSION OR JAX_VERSION STREQUAL "")
    if(DEFINED ENV{JAX_VERSION} AND NOT "$ENV{JAX_VERSION}" STREQUAL "")
        set(JAX_VERSION "$ENV{JAX_VERSION}")
    else()
        set(JAX_VERSION "0.6.2")
    endif()
endif()
set(JAX_GIT_TAG "jax-v${JAX_VERSION}")
message(STATUS "Using JAX version: ${JAX_VERSION} (tag: ${JAX_GIT_TAG})")
CPMAddPackage(
    NAME jax
    GIT_REPOSITORY https://github.com/jax-ml/jax.git
    GIT_TAG ${JAX_GIT_TAG}
    DOWNLOAD_ONLY YES
)
message(STATUS "Jax source directory: ${jax_SOURCE_DIR}")
# https://github.com/openxla/xla/archive/8f3d7099bc30c2912357c770ca25172fad9ae302.zip
CPMAddPackage(
    NAME XLA
    GIT_REPOSITORY https://github.com/openxla/xla.git
    GIT_TAG 8f3d7099bc30c2912357c770ca25172fad9ae302
    DOWNLOAD_ONLY YES
)
message(STATUS "XLA source directory: ${XLA_SOURCE_DIR}")

# JAX headers expect a CUDA tree rooted at third_party/gpus/cuda with an
# include/ subtree. On systems like Isambard, the real toolkit headers live
# under targets/<arch>/include rather than directly under CUDA_ROOT/include, so
# build the minimal expected directory layout explicitly.
file(MAKE_DIRECTORY ${CMAKE_SOURCE_DIR}/third_party/gpus)
set(CUDA_THIRD_PARTY_ROOT ${CMAKE_SOURCE_DIR}/third_party/gpus/cuda)
file(REMOVE_RECURSE ${CUDA_THIRD_PARTY_ROOT})
file(MAKE_DIRECTORY ${CUDA_THIRD_PARTY_ROOT})
file(MAKE_DIRECTORY ${CUDA_THIRD_PARTY_ROOT}/include)
list(GET CUDAToolkit_INCLUDE_DIRS 0 CUDA_PRIMARY_INCLUDE_DIR)
set(CUDA_TOOLKIT_HEADERS
    cooperative_groups.h
    cuComplex.h
    cuda.h
    cuda_fp8.h
)
foreach(header IN LISTS CUDA_TOOLKIT_HEADERS)
    execute_process(
        COMMAND ln -sfn ${CUDA_PRIMARY_INCLUDE_DIR}/${header} ${CUDA_THIRD_PARTY_ROOT}/include/${header}
        RESULT_VARIABLE ln_result
    )
    if(NOT ln_result EQUAL 0)
        message(FATAL_ERROR "Failed to create CUDA toolkit header symlink: ${CUDA_PRIMARY_INCLUDE_DIR}/${header}")
    endif()
endforeach()
if(CUDA_MATH_INCLUDE_DIR)
    set(CUDA_MATH_HEADERS
        cublas_v2.h
        cufft.h
        cusolverDn.h
        cusolver_common.h
        cusparse.h
    )
    foreach(header IN LISTS CUDA_MATH_HEADERS)
        execute_process(
            COMMAND ln -sfn ${CUDA_MATH_INCLUDE_DIR}/${header} ${CUDA_THIRD_PARTY_ROOT}/include/${header}
            RESULT_VARIABLE ln_result
        )
        if(NOT ln_result EQUAL 0)
            message(FATAL_ERROR "Failed to create CUDA math header symlink: ${CUDA_MATH_INCLUDE_DIR}/${header}")
        endif()
    endforeach()
endif()
if(DEFINED ENV{CUDA_ROOT} AND NOT "$ENV{CUDA_ROOT}" STREQUAL "" AND EXISTS "$ENV{CUDA_ROOT}/extras")
    execute_process(
        COMMAND ln -sfn $ENV{CUDA_ROOT}/extras ${CUDA_THIRD_PARTY_ROOT}/extras
        RESULT_VARIABLE ln_result
    )
    if(NOT ln_result EQUAL 0)
        message(FATAL_ERROR "Failed to create CUDA extras symlink: $ENV{CUDA_ROOT}/extras")
    endif()
endif()

if(CUDNN_FOUND)
    message(STATUS "Resolved cuDNN include dir: ${CUDNN_INCLUDE_PATH}")
    execute_process(
        COMMAND ln -sfn ${CUDNN_INCLUDE_PATH} ${CMAKE_SOURCE_DIR}/third_party/gpus/cudnn
        RESULT_VARIABLE ln_result
    )
    if(NOT ln_result EQUAL 0)
        message(FATAL_ERROR "Failed to create symlink for CUDNN include: ${CUDNN_INCLUDE_PATH}")
    endif()
endif()
file(MAKE_DIRECTORY ${CMAKE_SOURCE_DIR}/xla/ffi)
execute_process(
    COMMAND ln -sfn ${XLA_SOURCE_DIR}/xla/ffi/api ${CMAKE_SOURCE_DIR}/xla/ffi
    RESULT_VARIABLE ln_result
)

if(NOT ln_result EQUAL 0)
    message(FATAL_ERROR "Failed to create symlink for XLA_DIR: ${XLA_SOURCE_DIR}")
endif()

# Set JAX_GPU_CUDA to 1 (this is required for jaxlib to compile correctly)
add_definitions(-DJAX_GPU_CUDA=1)
# Suppress host CUDA warning
add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:-diag-suppress=940,2473,2810>)
# Suppress host C++ warnings from GCC/Clang (only for C++ sources)
add_compile_options(
  $<$<COMPILE_LANGUAGE:CXX>:-Wno-return-type>
  $<$<COMPILE_LANGUAGE:CXX>:-Wno-attributes>
)
# Add libraries
add_library(cyclic SHARED
    src/cuda/cyclic.cu
    src/cuda/utils/jax_utils.cc
    src/cuda/utils/shm.cc
    ${jax_SOURCE_DIR}/jaxlib/gpu/gpu_kernel_helpers.cc
)

add_library(potrs SHARED
    src/cuda/potrs.cu
    src/cuda/utils/jax_utils.cc
    src/cuda/utils/shm.cc
    ${jax_SOURCE_DIR}/jaxlib/gpu/gpu_kernel_helpers.cc
)
add_library(potrs_cusolvermp SHARED
    src/cuda/potrs_cusolvermp_stub.cc
    src/cuda/utils/cusolvermp_context.cc
    ${jax_SOURCE_DIR}/jaxlib/gpu/gpu_kernel_helpers.cc
)
add_library(potrs_mp SHARED
    src/cuda/potrs_mp.cu
    src/cuda/utils/jax_utils.cc
    src/cuda/utils/shm.cc
    src/cuda/utils/ipc_utils.cc
    ${jax_SOURCE_DIR}/jaxlib/gpu/gpu_kernel_helpers.cc
)
add_library(potri SHARED
    src/cuda/potri.cu
    src/cuda/utils/jax_utils.cc
    src/cuda/utils/shm.cc
    ${jax_SOURCE_DIR}/jaxlib/gpu/gpu_kernel_helpers.cc
)
add_library(potri_mp SHARED
    src/cuda/potri_mp.cu
    src/cuda/utils/jax_utils.cc
    src/cuda/utils/shm.cc
    src/cuda/utils/ipc_utils.cc
    ${jax_SOURCE_DIR}/jaxlib/gpu/gpu_kernel_helpers.cc
)
add_library(syevd SHARED
    src/cuda/syevd.cu
    src/cuda/utils/jax_utils.cc
    src/cuda/utils/shm.cc
    ${jax_SOURCE_DIR}/jaxlib/gpu/gpu_kernel_helpers.cc
)
add_library(syevd_cusolvermp SHARED
    src/cuda/syevd_cusolvermp_stub.cc
    src/cuda/utils/cusolvermp_context.cc
    ${jax_SOURCE_DIR}/jaxlib/gpu/gpu_kernel_helpers.cc
)
add_library(syevd_mp SHARED
    src/cuda/syevd_mp.cu
    src/cuda/utils/jax_utils.cc
    src/cuda/utils/shm.cc
    src/cuda/utils/ipc_utils.cc
    ${jax_SOURCE_DIR}/jaxlib/gpu/gpu_kernel_helpers.cc
)
add_library(syevd_no_V SHARED
    src/cuda/syevd_no_V.cu
    src/cuda/utils/jax_utils.cc
    src/cuda/utils/shm.cc
    ${jax_SOURCE_DIR}/jaxlib/gpu/gpu_kernel_helpers.cc
)
add_library(syevd_no_V_mp SHARED
    src/cuda/syevd_no_V_mp.cu
    src/cuda/utils/jax_utils.cc
    src/cuda/utils/shm.cc
    src/cuda/utils/ipc_utils.cc
    ${jax_SOURCE_DIR}/jaxlib/gpu/gpu_kernel_helpers.cc
)

# Make sure we can see "thirdparty/gpus" and jax
set(SRC_CUDA_INCLUDE ${CMAKE_SOURCE_DIR}/src/cuda/include)
set(SRC_CUDA_ROOT ${CMAKE_SOURCE_DIR}/src/cuda)
set(JAXMG_GPU_INCLUDE_DIRS ${CMAKE_SOURCE_DIR} ${jax_SOURCE_DIR} ${SRC_CUDA_INCLUDE} ${SRC_CUDA_ROOT} ${CUDA_PRIMARY_INCLUDE_DIR})
if(CUDA_MATH_INCLUDE_DIR)
    list(APPEND JAXMG_GPU_INCLUDE_DIRS ${CUDA_MATH_INCLUDE_DIR})
endif()
target_include_directories(cyclic PUBLIC ${JAXMG_GPU_INCLUDE_DIRS})
target_include_directories(potrs PUBLIC ${JAXMG_GPU_INCLUDE_DIRS})
target_include_directories(potrs_cusolvermp PUBLIC ${JAXMG_GPU_INCLUDE_DIRS})
target_include_directories(potrs_mp PUBLIC ${JAXMG_GPU_INCLUDE_DIRS})
target_include_directories(potri PUBLIC ${JAXMG_GPU_INCLUDE_DIRS})
target_include_directories(potri_mp PUBLIC ${JAXMG_GPU_INCLUDE_DIRS})
target_include_directories(syevd PUBLIC ${JAXMG_GPU_INCLUDE_DIRS})
target_include_directories(syevd_cusolvermp PUBLIC ${JAXMG_GPU_INCLUDE_DIRS})
if(HAVE_CUSOLVERMP)
target_include_directories(potrs_cusolvermp BEFORE PUBLIC ${CUSOLVERMP_INCLUDE_DIR} ${NCCL_INCLUDE_DIR})
target_include_directories(syevd_cusolvermp BEFORE PUBLIC ${CUSOLVERMP_INCLUDE_DIR} ${NCCL_INCLUDE_DIR})
target_compile_definitions(potrs_cusolvermp PRIVATE JAXMG_HAVE_CUSOLVERMP=1 JAXMG_HAVE_NCCL=1)
target_compile_definitions(syevd_cusolvermp PRIVATE JAXMG_HAVE_CUSOLVERMP=1 JAXMG_HAVE_NCCL=1)
endif()
target_include_directories(syevd_mp PUBLIC ${JAXMG_GPU_INCLUDE_DIRS})
target_include_directories(syevd_no_V PUBLIC ${JAXMG_GPU_INCLUDE_DIRS})
target_include_directories(syevd_no_V_mp PUBLIC ${JAXMG_GPU_INCLUDE_DIRS})

# Set the CUDA architecture
set_target_properties(cyclic PROPERTIES
    CUDA_ARCHITECTURES "all"
)
set_target_properties(potrs PROPERTIES
    CUDA_ARCHITECTURES "all"
)
set_target_properties(potrs_cusolvermp PROPERTIES
    CUDA_ARCHITECTURES "all"
)
set_target_properties(potrs_mp PROPERTIES
    CUDA_ARCHITECTURES "all"
)
set_target_properties(potri PROPERTIES
    CUDA_ARCHITECTURES "all"
)
set_target_properties(potri_mp PROPERTIES
    CUDA_ARCHITECTURES "all"
)
set_target_properties(syevd PROPERTIES
    CUDA_ARCHITECTURES "all"
)
set_target_properties(syevd_cusolvermp PROPERTIES
    CUDA_ARCHITECTURES "all"
)
set_target_properties(syevd_mp PROPERTIES
    CUDA_ARCHITECTURES "all"
)
set_target_properties(syevd_no_V PROPERTIES
    CUDA_ARCHITECTURES "all"
)
set_target_properties(syevd_no_V_mp PROPERTIES
    CUDA_ARCHITECTURES "all"
)

# Link against CUDA libraries and absl
target_link_libraries(cyclic
    absl::strings
    absl::status
    absl::statusor
    CUDA::cusolver
    CUDA::cupti
    ${CUSOLVERMG_LIB}
    cuda
)
target_link_libraries(potrs
    absl::strings
    absl::status
    absl::statusor
    CUDA::cusolver
    CUDA::cupti
    ${CUSOLVERMG_LIB}
    cuda
)
target_link_libraries(potrs_cusolvermp
    absl::strings
    absl::status
    absl::statusor
    CUDA::cudart
    CUDA::cuda_driver
    CUDA::cupti
)
if(HAVE_CUSOLVERMP)
target_link_libraries(potrs_cusolvermp
    ${CUSOLVERMP_LIB}
    ${NCCL_LIB}
)
endif()
target_link_libraries(potrs_mp
    absl::strings
    absl::status
    absl::statusor
    CUDA::cusolver
    CUDA::cupti
    ${CUSOLVERMG_LIB}
    cuda
)

target_link_libraries(potri
    absl::strings
    absl::status
    absl::statusor
    CUDA::cusolver
    CUDA::cupti
    ${CUSOLVERMG_LIB}
    cuda
)
target_link_libraries(potri_mp
    absl::strings
    absl::status
    absl::statusor
    CUDA::cusolver
    CUDA::cupti
    ${CUSOLVERMG_LIB}
    cuda
)
target_link_libraries(syevd
    absl::strings
    absl::status
    absl::statusor
    CUDA::cusolver
    CUDA::cupti
    ${CUSOLVERMG_LIB}
    cuda
)
target_link_libraries(syevd_cusolvermp
    absl::strings
    absl::status
    absl::statusor
    CUDA::cudart
    CUDA::cuda_driver
    CUDA::cupti
)
if(HAVE_CUSOLVERMP)
target_link_libraries(syevd_cusolvermp
    ${CUSOLVERMP_LIB}
    ${NCCL_LIB}
)
endif()
target_link_libraries(syevd_mp
    absl::strings
    absl::status
    absl::statusor
    CUDA::cusolver
    CUDA::cupti
    ${CUSOLVERMG_LIB}
    cuda
)

target_link_libraries(syevd_no_V
    absl::strings
    absl::status
    absl::statusor
    CUDA::cusolver
    CUDA::cupti
    ${CUSOLVERMG_LIB}
    cuda
)
target_link_libraries(syevd_no_V_mp
    absl::strings
    absl::status
    absl::statusor
    CUDA::cusolver
    CUDA::cupti
    ${CUSOLVERMG_LIB}
    cuda
)
# Make install targets
install(TARGETS cyclic
    LIBRARY DESTINATION ${CMAKE_SOURCE_DIR}/${BIN_DST}
)
install(TARGETS potrs
    LIBRARY DESTINATION ${CMAKE_SOURCE_DIR}/${BIN_DST}
)
install(TARGETS potrs_cusolvermp
    LIBRARY DESTINATION ${CMAKE_SOURCE_DIR}/${BIN_DST}
)
install(TARGETS potrs_mp
    LIBRARY DESTINATION ${CMAKE_SOURCE_DIR}/${BIN_DST}
)
install(TARGETS potri
    LIBRARY DESTINATION ${CMAKE_SOURCE_DIR}/${BIN_DST}
)
install(TARGETS potri_mp
    LIBRARY DESTINATION ${CMAKE_SOURCE_DIR}/${BIN_DST}
)
install(TARGETS syevd
    LIBRARY DESTINATION ${CMAKE_SOURCE_DIR}/${BIN_DST}
)
install(TARGETS syevd_cusolvermp
    LIBRARY DESTINATION ${CMAKE_SOURCE_DIR}/${BIN_DST}
)
install(TARGETS syevd_mp
    LIBRARY DESTINATION ${CMAKE_SOURCE_DIR}/${BIN_DST}
)
install(TARGETS syevd_no_V
    LIBRARY DESTINATION ${CMAKE_SOURCE_DIR}/${BIN_DST}
)
install(TARGETS syevd_no_V_mp
    LIBRARY DESTINATION ${CMAKE_SOURCE_DIR}/${BIN_DST}
)

# module load gcc/11.5.0 cudnn/9.2.0.82-12 cuda/12.8.0

# cmake ..
# cmake --build . --target install -j10
# OR
# cmake --build . --target cyclic && cmake --install .
# cmake --build . --target potrs_mp && cmake --install .
# cmake --build . --target potrs && cmake --install .
# cmake --build . --target potri && cmake --install .
# cmake --build . --target syevd && cmake --install .
# cmake --build . --target syevd_no_V && cmake --install .
