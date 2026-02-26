# ── Stage 1: builder ─────────────────────────────────────────────────────────
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive

# System packages: CMake 3.28 from Kitware PPA + GCC 12 + build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates gnupg wget git \
        build-essential gcc-12 g++-12 \
        ninja-build \
    && wget -qO - https://apt.kitware.com/keys/kitware-archive-latest.asc \
        | gpg --dearmor -o /usr/share/keyrings/kitware-archive-keyring.gpg \
    && echo 'deb [signed-by=/usr/share/keyrings/kitware-archive-keyring.gpg] https://apt.kitware.com/ubuntu/ jammy main' \
        > /etc/apt/sources.list.d/kitware.list \
    && apt-get update && apt-get install -y --no-install-recommends cmake \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Use GCC 12 as default
RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-12 120 \
 && update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-12 120

WORKDIR /workspace/servoflow

# Copy project files
COPY . .

# Build arguments
ARG CMAKE_BUILD_TYPE=Release
ARG SF_CUDA_ARCHS=86
ARG SF_BUILD_TESTS=ON
ARG SF_BUILD_BENCHMARKS=ON
ARG SF_BUILD_EXAMPLES=ON

RUN cmake -B build -G Ninja \
        -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE} \
        -DSF_CUDA_ARCHS="${SF_CUDA_ARCHS}" \
        -DSF_BUILD_TESTS=${SF_BUILD_TESTS} \
        -DSF_BUILD_BENCHMARKS=${SF_BUILD_BENCHMARKS} \
        -DSF_BUILD_EXAMPLES=${SF_BUILD_EXAMPLES} \
        -DSF_USE_FLASH_ATTN=ON \
    && cmake --build build -j6

# ── Stage 2: runtime ─────────────────────────────────────────────────────────
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04 AS runtime

WORKDIR /workspace/servoflow

# Copy compiled artifacts from builder
COPY --from=builder /workspace/servoflow/build/libservoflow.so /usr/local/lib/
COPY --from=builder /workspace/servoflow/build/tests           ./build/tests
COPY --from=builder /workspace/servoflow/build/benchmarks      ./build/benchmarks
COPY --from=builder /workspace/servoflow/build/examples        ./build/examples
COPY --from=builder /workspace/servoflow/include               ./include

RUN ldconfig

CMD ["/bin/bash"]
