#pragma once
#include <tuple>
#include "flash.h"

namespace at {
namespace cuda {
namespace philox {

__host__ __device__ inline std::tuple<uint64_t, uint64_t> unpack(const FLASH_NAMESPACE::PhiloxCudaState& state) {
    return std::make_tuple(state.seed, state.offset);
}

} // namespace philox
} // namespace cuda
} // namespace at
