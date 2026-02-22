// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "servoflow/core/tensor.h"
#include "servoflow/backend/backend.h"
#include <functional>
#include <vector>

namespace sf {

// Denoising function type:
//   (noisy_action, timestep, condition_cache) → predicted_velocity
// The model fills this in; the sampler calls it at each step.
using DenoiseFn = std::function<void(
    const Tensor& x_t,           // noisy action [B, T_action, action_dim]
    float         t,             // timestep in [0, 1]
    const Tensor& condition,     // pre-computed condition embedding [B, S, D]
    Tensor&       velocity_out,  // output: predicted velocity
    StreamHandle  stream
)>;

// Noise schedule for ODE / SDE samplers.
struct Schedule {
    int   num_steps    = 10;    // number of denoising steps
    float t_start      = 1.f;  // start time (pure noise)
    float t_end        = 0.f;  // end time (clean action)

    // Returns uniformly spaced timesteps from t_start to t_end.
    std::vector<float> linspace() const;
};

// ─────────────────────────────────────────────────────────────────────────────
// ISampler: interface for all sampling strategies.
// ─────────────────────────────────────────────────────────────────────────────
class ISampler {
public:
    virtual ~ISampler() = default;

    // Sample clean action given a noisy start and a denoising function.
    // x_init: initial noise [B, T_action, action_dim] — caller provides this.
    // condition: pre-computed condition (image/language tokens) [B, S, D].
    // Returns the denoised action tensor.
    virtual Tensor sample(const Tensor& x_init,
                          const Tensor& condition,
                          const DenoiseFn& denoise_fn,
                          const Schedule& schedule,
                          BackendPtr backend,
                          StreamHandle stream,
                          SamplerBuffers* buffers = nullptr) = 0;
};

// ─────────────────────────────────────────────────────────────────────────────
// FlowMatchingSampler: Euler ODE solver for Rectified Flow / Flow Matching.
//
// Used by RDT-1B (and π0). The ODE is:
//   dx/dt = v_θ(x_t, t)
// Euler update: x_{t-Δt} = x_t + dt * v_θ(x_t, t)
//
// CUDA Graph requirement: all working buffers (x_t, velocity) must have
// stable addresses across calls. The caller (InferenceEngine) must pre-
// allocate these buffers and pass them via SamplerBuffers. The sampler
// never allocates inside sample() when buffers are provided.
// ─────────────────────────────────────────────────────────────────────────────

// Pre-allocated working buffers owned by InferenceEngine.
// Passed into sample() to guarantee stable addresses for CUDA Graph capture.
struct SamplerBuffers {
    Tensor x_t;       // [B, T_action, action_dim] — noisy action (in/out)
    Tensor velocity;  // [B, T_action, action_dim] — predicted velocity (scratch)
};

class FlowMatchingSampler : public ISampler {
public:
    // use_cuda_graph: capture the denoising loop on first call, replay thereafter.
    explicit FlowMatchingSampler(bool use_cuda_graph = true);

    // buffers: pre-allocated by InferenceEngine; must remain valid for the
    // lifetime of the sampler. If nullptr, buffers are allocated internally
    // (disables CUDA Graph support).
    Tensor sample(const Tensor& x_init,
                  const Tensor& condition,
                  const DenoiseFn& denoise_fn,
                  const Schedule& schedule,
                  BackendPtr backend,
                  StreamHandle stream,
                  SamplerBuffers* buffers = nullptr) override;

private:
    bool use_cuda_graph_ = true;
    bool graph_captured_ = false;
};

// ─────────────────────────────────────────────────────────────────────────────
// DDIMSampler: deterministic DDIM for DDPM-trained models.
// ─────────────────────────────────────────────────────────────────────────────
class DDIMSampler : public ISampler {
public:
    Tensor sample(const Tensor& x_init,
                  const Tensor& condition,
                  const DenoiseFn& denoise_fn,
                  const Schedule& schedule,
                  BackendPtr backend,
                  StreamHandle stream,
                  SamplerBuffers* buffers = nullptr) override;
};

}  // namespace sf
