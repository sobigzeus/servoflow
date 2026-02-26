cmake_minimum_required(VERSION 3.22)
project(servoflow LANGUAGES CXX CUDA VERSION 0.1.0)

# ── Language standards ────────────────────────────────────────────────────────
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# ── Build type ────────────────────────────────────────────────────────────────
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# ── Options ───────────────────────────────────────────────────────────────────
option(SF_BUILD_TESTS       "Build unit tests"                   ON)
option(SF_BUILD_BENCHMARKS  "Build benchmarks"                   ON)
option(SF_BUILD_EXAMPLES    "Build examples"                     ON)
option(SF_USE_FLASH_ATTN    "Use FlashAttention v2 for attention" ON)
option(SF_ENABLE_CUDA       "Build CUDA backend"                 ON)
option(SF_BUILD_PYTHON      "Build Python bindings (pybind11)"   OFF)

# ── CUDA setup ────────────────────────────────────────────────────────────────
if(SF_ENABLE_CUDA)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)

    # Default architectures: Ampere (RTX 3090) + Turing (T4) + Ada (RTX 4090)
    # Users can override with -DSF_CUDA_ARCHS="80;86;89"
    if(NOT DEFINED SF_CUDA_ARCHS)
        set(SF_CUDA_ARCHS "75;80;86;89" CACHE STRING "CUDA architectures to target")
    endif()
    set(CMAKE_CUDA_ARCHITECTURES ${SF_CUDA_ARCHS})

    # Compiler flags for device code.
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --use_fast_math -Xcompiler -Wall --expt-relaxed-constexpr")
    set(CMAKE_CUDA_FLAGS_RELEASE "-O3 -DNDEBUG")
endif()

# ── Compiler flags ─────────────────────────────────────────────────────────────
add_compile_options(
    $<$<COMPILE_LANGUAGE:CXX>:-Wall>
    $<$<COMPILE_LANGUAGE:CXX>:-Wextra>
    $<$<COMPILE_LANGUAGE:CXX>:-Wpedantic>
    $<$<COMPILE_LANGUAGE:CXX>:-Wno-unused-parameter>
)

# ── External dependencies ─────────────────────────────────────────────────────
include(FetchContent)

# nlohmann/json — header-only JSON parser for model configs.
FetchContent_Declare(json
    URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz
    URL_HASH SHA256=d6c65aca6b1ed68e7a182f4757257b107ae403032760ed6ef121c9d55e81757d
)
FetchContent_MakeAvailable(json)

# GoogleTest — unit testing.
if(SF_BUILD_TESTS)
    FetchContent_Declare(googletest
        URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.tar.gz
        URL_HASH SHA256=8ad598c73ad796e0d8280b082cebd82a630d73e73cd3c70057938a6501bba5d7
    )
    set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)
endif()

# ── Core library sources ──────────────────────────────────────────────────────
set(SF_CORE_SOURCES
    src/backend/backend.cpp
    src/sampling/flow_matching.cpp
    src/engine/inference_engine.cpp
    src/loader/safetensors.cpp
    src/models/rdt1b/config.cpp
    src/models/rdt1b/rdt1b.cpp
)

set(SF_CUDA_SOURCES
    src/backend/cuda/cuda_backend.cu
    src/backend/cuda/ops/attention.cu
    src/backend/cuda/ops/elementwise.cu
    src/models/rdt1b/dit_block.cu
)

# FlashAttention v2 — integrated from source.
if(SF_USE_FLASH_ATTN AND SF_ENABLE_CUDA)
    message(STATUS "Integrating FlashAttention v2 from source")
    
    # Add FlashAttention sources
    # file(GLOB FLASH_ATTN_SOURCES "src/backend/cuda/flash_attn/*.cu")
    # Optimize build time by only compiling needed kernels
    set(FLASH_ATTN_SOURCES 
        src/backend/cuda/flash_attn/flash_fwd_hdim32_fp16_causal_sm80.cu
        src/backend/cuda/flash_attn/flash_fwd_hdim32_fp16_sm80.cu
        src/backend/cuda/flash_attn/flash_fwd_hdim64_fp16_causal_sm80.cu
        src/backend/cuda/flash_attn/flash_fwd_hdim64_fp16_sm80.cu
        src/backend/cuda/flash_attn/flash_fwd_hdim128_fp16_causal_sm80.cu
        src/backend/cuda/flash_attn/flash_fwd_hdim128_fp16_sm80.cu
    )
    list(APPEND SF_CUDA_SOURCES 
        src/backend/cuda/flash_attn_adapter.cu
        ${FLASH_ATTN_SOURCES}
    )
    
    set(SF_HAVE_FLASH_ATTN ON)
endif()

# ── servoflow library target ──────────────────────────────────────────────────
if(SF_ENABLE_CUDA)
    add_library(servoflow SHARED ${SF_CORE_SOURCES} ${SF_CUDA_SOURCES})
else()
    add_library(servoflow SHARED ${SF_CORE_SOURCES})
endif()

target_include_directories(servoflow
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(servoflow
    PUBLIC  nlohmann_json::nlohmann_json
)

if(SF_ENABLE_CUDA)
    target_link_libraries(servoflow
        PUBLIC
            CUDA::cudart
            CUDA::cublas
            CUDA::curand
    )
    target_compile_definitions(servoflow PUBLIC SF_CUDA_ENABLED)
    set_target_properties(servoflow PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
    )
    
    if(SF_HAVE_FLASH_ATTN)
        set(CUTLASS_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/flash-attention/csrc/cutlass/include")
        if(NOT EXISTS ${CUTLASS_INCLUDE_DIR})
            message(FATAL_ERROR "Cutlass include directory not found at ${CUTLASS_INCLUDE_DIR}. Please run 'git submodule update --init --recursive'")
        endif()
        
        target_include_directories(servoflow PRIVATE
            ${CUTLASS_INCLUDE_DIR}
            ${CMAKE_CURRENT_SOURCE_DIR}/src/backend/cuda
        )
        target_compile_definitions(servoflow PUBLIC SF_USE_FLASH_ATTN)
    endif()
endif()

# ── Tests ─────────────────────────────────────────────────────────────────────
if(SF_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# ── Benchmarks ────────────────────────────────────────────────────────────────
if(SF_BUILD_BENCHMARKS)
    add_subdirectory(benchmarks)
endif()

# ── Examples ──────────────────────────────────────────────────────────────────
if(SF_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

# ── Install rules ─────────────────────────────────────────────────────────────
include(GNUInstallDirs)

# Export nlohmann_json alongside servoflow so that the install set is complete.
install(TARGETS nlohmann_json
    EXPORT servoflow-targets
)

install(TARGETS servoflow
    EXPORT servoflow-targets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(EXPORT servoflow-targets
    FILE servoflow-config.cmake
    NAMESPACE sf::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/servoflow
)

# ── Summary ───────────────────────────────────────────────────────────────────
message(STATUS "")
message(STATUS "ServoFlow ${PROJECT_VERSION} build configuration:")
message(STATUS "  Build type:        ${CMAKE_BUILD_TYPE}")
message(STATUS "  CUDA backend:      ${SF_ENABLE_CUDA}")
message(STATUS "  CUDA archs:        ${SF_CUDA_ARCHS}")
message(STATUS "  FlashAttention:    ${SF_HAVE_FLASH_ATTN}")
message(STATUS "  Tests:             ${SF_BUILD_TESTS}")
message(STATUS "  Benchmarks:        ${SF_BUILD_BENCHMARKS}")
message(STATUS "")
