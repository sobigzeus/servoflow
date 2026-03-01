#include <gtest/gtest.h>
#include "servoflow/backend/backend.h"
#include "servoflow/core/tensor.h"
#include <vector>
#include <cmath>

using namespace sf;

TEST(QuantizationTest, Int8DeQuantizePerTensor) {
    if (!BackendRegistry::instance().has(DeviceType::CUDA)) {
        GTEST_SKIP() << "CUDA backend not available";
    }
    auto backend = get_backend(DeviceType::CUDA, 0);

    // Shape: [2, 4]
    // Data: -10, -5, 0, 5, 10, 20, 30, 40
    // Scale: 0.5
    // Expected: -5, -2.5, 0, 2.5, 5, 10, 15, 20

    std::vector<int8_t> h_input = {-10, -5, 0, 5, 10, 20, 30, 40};
    float scale_val = 0.5f;

    Tensor input = backend->alloc({2, 4}, DType::Int8);
    Tensor scale = backend->alloc({1}, DType::Float16);
    Tensor output = backend->alloc({2, 4}, DType::Float16);

    // Copy input
    backend->copy(input, Tensor(std::make_shared<Storage>(h_input.data(), h_input.size(), kCPU, nullptr), {2, 4}, DType::Int8));
    
    // Copy scale (need to convert float to half on host or fill)
    // We can use cast or fill. fill only takes float but converts to dtype.
    backend->fill(scale, scale_val);

    backend->dequantize(input, scale, output);
    backend->sync_device();

    // Verify
    Tensor h_output = backend->alloc_pinned({2, 4}, DType::Float32);
    backend->cast(output, h_output); // cast back to fp32 for reading
    backend->sync_device();

    const float* out_ptr = static_cast<const float*>(h_output.raw_data_ptr());
    for (size_t i = 0; i < h_input.size(); ++i) {
        float expected = h_input[i] * scale_val;
        EXPECT_NEAR(out_ptr[i], expected, 1e-2);
    }
}

TEST(QuantizationTest, Int8DeQuantizePerChannel) {
    if (!BackendRegistry::instance().has(DeviceType::CUDA)) {
        GTEST_SKIP() << "CUDA backend not available";
    }
    auto backend = get_backend(DeviceType::CUDA, 0);

    // Shape: [2, 4] -> 2 channels (rows)
    // Data: 
    // Row 0: 10, 10, 10, 10 (scale 0.5 -> 5)
    // Row 1: 20, 20, 20, 20 (scale 2.0 -> 40)

    std::vector<int8_t> h_input = {
        10, 10, 10, 10,
        20, 20, 20, 20
    };
    
    Tensor input = backend->alloc({2, 4}, DType::Int8);
    Tensor scale = backend->alloc({2}, DType::Float16); // Per-channel scale
    Tensor output = backend->alloc({2, 4}, DType::Float16);

    backend->copy(input, Tensor(std::make_shared<Storage>(h_input.data(), h_input.size(), kCPU, nullptr), {2, 4}, DType::Int8));
    
    // Fill scale manually via host staging
    // We need a helper to fill FP16 data from host
    // Let's alloc a pinned FP32 buffer, cast to FP16 device
    Tensor h_scale_f32 = backend->alloc_pinned({2}, DType::Float32);
    float* s_ptr = static_cast<float*>(h_scale_f32.raw_data_ptr());
    s_ptr[0] = 0.5f;
    s_ptr[1] = 2.0f;
    
    Tensor d_scale_f32 = backend->alloc({2}, DType::Float32);
    backend->copy(d_scale_f32, h_scale_f32);
    backend->cast(d_scale_f32, scale); // F32 -> F16

    backend->dequantize(input, scale, output);
    backend->sync_device();

    // Verify
    Tensor h_output = backend->alloc_pinned({2, 4}, DType::Float32);
    backend->cast(output, h_output);
    backend->sync_device();

    const float* out_ptr = static_cast<const float*>(h_output.raw_data_ptr());
    for (int i = 0; i < 4; ++i) EXPECT_NEAR(out_ptr[i], 5.0f, 1e-2);
    for (int i = 4; i < 8; ++i) EXPECT_NEAR(out_ptr[i], 40.0f, 1e-2);
}
