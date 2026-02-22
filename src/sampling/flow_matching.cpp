// SPDX-License-Identifier: Apache-2.0
#include "servoflow/sampling/sampler.h"

#include <stdexcept>
#include <vector>
#include <cmath>

namespace sf {

std::vector<float> Schedule::linspace() const {
    std::vector<float> ts(num_steps + 1);
    for (int i = 0; i <= num_steps; ++i) {
        float alpha = static_cast<float>(i) / num_steps;
        ts[i] = t_start + alpha * (t_end - t_start);
    }
    return ts;
}

// ─────────────────────────────────────────────────────────────────────────────
// FlowMatchingSampler
// ─────────────────────────────────────────────────────────────────────────────
FlowMatchingSampler::FlowMatchingSampler(bool use_cuda_graph)
    : use_cuda_graph_(use_cuda_graph) {}

Tensor FlowMatchingSampler::sample(const Tensor& x_init,
                                    const Tensor& condition,
                                    const DenoiseFn& denoise_fn,
                                    const Schedule& schedule,
                                    BackendPtr backend,
                                    StreamHandle stream,
                                    SamplerBuffers* buffers) {
    // ── Buffer selection ─────────────────────────────────────────────────
    // When buffers are provided by InferenceEngine, their addresses are stable
    // across calls, which is the prerequisite for CUDA Graph capture.
    // When buffers are nullptr (standalone use), we fall back to local alloc
    // and disable CUDA Graph.
    bool use_graph = use_cuda_graph_ && (buffers != nullptr);

    Tensor x_t, vel;
    if (buffers) {
        // Use pre-allocated stable buffers (addresses guaranteed constant).
        x_t = buffers->x_t;
        vel = buffers->velocity;
    } else {
        x_t = backend->alloc(x_init.shape(), x_init.dtype(), stream);
        vel = backend->alloc(x_init.shape(), x_init.dtype(), stream);
    }

    // Copy initial noise into x_t. This copy is intentionally OUTSIDE the
    // CUDA Graph — the graph only captures the denoising loop body.
    backend->copy(x_t, x_init, stream);

    auto timesteps = schedule.linspace();  // length = num_steps + 1

    // ── CUDA Graph path ───────────────────────────────────────────────────
    // First call: begin capture, run loop live (CUDA records all kernels).
    // Subsequent calls: replay the graph (zero CPU launch overhead per step).
    //
    // Correctness requirements (must be maintained by the caller):
    //   1. x_t, vel addresses are identical across calls (provided by buffers).
    //   2. denoise_fn issues only device-side work; no stream syncs inside.
    //   3. schedule.num_steps is constant (changing it requires re-capture).
    if (use_graph && !graph_captured_) {
        backend->graph_begin_capture(stream);
    }

    if (!use_graph || !graph_captured_) {
        // Live execution (first call or no-graph path).
        for (int step = 0; step < schedule.num_steps; ++step) {
            float t      = timesteps[step];
            float t_next = timesteps[step + 1];
            float dt     = t_next - t;  // negative: noise → data

            denoise_fn(x_t, t, condition, vel, stream);

            // x_t = x_t + dt * vel  (in-place, no extra alloc)
            backend->scale(vel, dt, vel, stream);
            backend->add(x_t, vel, x_t, stream);
        }

        if (use_graph) {
            backend->graph_end_capture(stream);
            graph_captured_ = true;
        }
    } else {
        // Graph replay: all kernel launches happen in a single driver call.
        backend->graph_launch(stream);
    }

    return x_t;
}

// ─────────────────────────────────────────────────────────────────────────────
// DDIMSampler
// ─────────────────────────────────────────────────────────────────────────────
Tensor DDIMSampler::sample(const Tensor& x_init,
                            const Tensor& condition,
                            const DenoiseFn& denoise_fn,
                            const Schedule& schedule,
                            BackendPtr backend,
                            StreamHandle stream,
                            SamplerBuffers* /*buffers*/) {
    Tensor x_t  = backend->alloc(x_init.shape(), x_init.dtype(), stream);
    Tensor eps  = backend->alloc(x_init.shape(), x_init.dtype(), stream);
    Tensor tmp  = backend->alloc(x_init.shape(), x_init.dtype(), stream);

    backend->copy(x_t, x_init, stream);
    auto timesteps = schedule.linspace();

    for (int step = 0; step < schedule.num_steps; ++step) {
        float t      = timesteps[step];
        float t_prev = timesteps[step + 1];

        float alpha_t      = 1.f - t;
        float alpha_t_prev = 1.f - t_prev;

        // Predict noise (model output is treated as noise prediction here).
        denoise_fn(x_t, t, condition, eps, stream);

        // DDIM deterministic update:
        //   x_0_pred = (x_t - sqrt(1-alpha_t) * eps) / sqrt(alpha_t)
        //   x_prev   = sqrt(alpha_t_prev) * x_0_pred + sqrt(1-alpha_t_prev) * eps
        float sqrt_alpha_t      = sqrtf(alpha_t);
        float sqrt_1m_alpha_t   = sqrtf(1.f - alpha_t);
        float sqrt_alpha_t_prev = sqrtf(alpha_t_prev);
        float sqrt_1m_alpha_tp  = sqrtf(1.f - alpha_t_prev);

        // x_0_pred = (x_t - sqrt_1m_alpha_t * eps) / sqrt_alpha_t
        backend->scale(eps, -sqrt_1m_alpha_t, tmp, stream);
        backend->add(x_t, tmp, tmp, stream);           // tmp = x_t - sqrt(1-a)*eps
        backend->scale(tmp, 1.f / sqrt_alpha_t, tmp, stream);  // tmp = x_0_pred

        // x_prev = sqrt_alpha_t_prev * x_0_pred + sqrt_1m_alpha_tp * eps
        backend->scale(tmp, sqrt_alpha_t_prev, x_t, stream);
        backend->scale(eps, sqrt_1m_alpha_tp, tmp, stream);
        backend->add(x_t, tmp, x_t, stream);
    }

    return x_t;
}

}  // namespace sf
