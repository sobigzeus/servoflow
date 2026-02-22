// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "servoflow/backend/backend.h"
#include "servoflow/core/tensor.h"
#include "servoflow/sampling/sampler.h"
#include <memory>
#include <string>
#include <vector>

namespace sf {

// ─────────────────────────────────────────────────────────────────────────────
// VLAInput: all observation inputs for one inference call.
// ─────────────────────────────────────────────────────────────────────────────
struct VLAInput {
    // Images from one or more cameras: [num_cameras, C, H, W] in fp32/fp16,
    // values in [0, 1].
    std::vector<Tensor> images;

    // Tokenised language instruction: [1, seq_len] int32.
    Tensor language_tokens;

    // Current robot state (joint positions/velocities): [1, state_dim] fp32.
    Tensor robot_state;
};

// ─────────────────────────────────────────────────────────────────────────────
// VLAOutput: predicted action chunk from one inference call.
// ─────────────────────────────────────────────────────────────────────────────
struct VLAOutput {
    // Action sequence: [1, T_action, action_dim] fp32 on CPU (ready to use).
    Tensor actions;
    // Wall-clock latency of this inference call (milliseconds).
    double latency_ms = 0.0;
};

// ─────────────────────────────────────────────────────────────────────────────
// EngineConfig: tuning knobs exposed to the user.
// ─────────────────────────────────────────────────────────────────────────────
struct EngineConfig {
    Device device = kCUDA0;

    // Inference precision for the DiT backbone.
    DType  compute_dtype = DType::Float16;

    // Number of diffusion denoising steps (lower = faster, less accurate).
    int    num_denoise_steps = 10;

    // Cache the condition embedding (image + language) across multiple
    // calls when the scene is static. Invalidated when images change.
    // This is the single biggest latency win for VLA inference loops.
    bool   cache_condition = true;

    // Capture the denoising loop into a CUDA Graph on first call.
    // Eliminates CPU kernel-launch overhead for the inner loop.
    bool   use_cuda_graph  = true;

    // Use pinned host memory for action output (faster D2H transfer).
    bool   pinned_output   = true;
};

// ─────────────────────────────────────────────────────────────────────────────
// IVLAModel: abstract interface that model implementations must satisfy.
// Separates the InferenceEngine (orchestration) from model weights (math).
// ─────────────────────────────────────────────────────────────────────────────
class IVLAModel {
public:
    virtual ~IVLAModel() = default;

    // Encode images and language into a joint condition embedding.
    // Called once per scene; result is cached by InferenceEngine.
    // Returns: condition [1, S, D]
    virtual Tensor encode_condition(const VLAInput& input,
                                    BackendPtr backend,
                                    StreamHandle stream) = 0;

    // Single denoising step: predict velocity given noisy action + condition.
    // x_t:       [1, T_action, action_dim]
    // t:         scalar timestep in [0, 1]
    // condition: [1, S, D]
    // velocity:  [1, T_action, action_dim]  (written in-place)
    virtual void denoise_step(const Tensor& x_t, float t,
                              const Tensor& condition,
                              Tensor& velocity,
                              BackendPtr backend,
                              StreamHandle stream) = 0;

    // Decode raw network output to action space (e.g. un-normalise).
    virtual Tensor decode_action(const Tensor& raw,
                                 BackendPtr backend,
                                 StreamHandle stream) = 0;

    // Metadata.
    virtual int64_t action_dim()   const = 0;
    virtual int64_t action_horizon() const = 0;
    virtual DType   dtype()         const = 0;
};

// ─────────────────────────────────────────────────────────────────────────────
// InferenceEngine: orchestrates memory, backend, sampler, and model.
//
// Key optimisations:
//   1. Condition cache: vision + language encoding only re-runs when
//      images change, saving ~60-80% of total compute for slow-changing scenes.
//   2. Static buffer pool: all intermediate tensors are pre-allocated once
//      at init; zero runtime allocation in the hot path.
//   3. CUDA Graph: the entire denoising loop (N denoise_step calls) is
//      captured as a CUDA Graph on the first inference call and replayed
//      subsequently, eliminating CPU–GPU synchronisation overhead.
//   4. Async D2H: action result is copied host-side on a separate stream
//      while the next encode can begin on the main stream.
//   5. Multi-stream: encode_condition runs on stream_encode,
//      denoising loop runs on stream_denoise; they are serialised via
//      a CUDA event only at the condition hand-off point.
// ─────────────────────────────────────────────────────────────────────────────
class InferenceEngine {
public:
    InferenceEngine(std::shared_ptr<IVLAModel> model,
                    std::shared_ptr<ISampler>  sampler,
                    EngineConfig               config = {});
    ~InferenceEngine();

    // Load model weights from a ServoFlow checkpoint directory or file.
    void load_weights(const std::string& path);

    // Run one full inference cycle. Thread-safe via internal mutex.
    // input.frame_id: if non-zero, used to detect new camera frames for
    // condition cache invalidation. Set to 0 to rely on mark_new_frame().
    VLAOutput infer(const VLAInput& input);

    // Invalidate the condition cache (force re-encode on next call).
    void invalidate_condition_cache();

    // Call this each time the camera produces a new frame before infer().
    // The engine will re-encode condition on the next infer() call.
    // frame_id must be monotonically increasing; wrapping is fine.
    void mark_new_frame(uint64_t frame_id);

    // Release cached GPU memory.
    void empty_cache();

    const EngineConfig& config() const { return config_; }
    BackendPtr          backend() const { return backend_; }

private:
    // Pre-allocate all working tensors. Called once at construction.
    void preallocate_buffers();

    std::shared_ptr<IVLAModel> model_;
    std::shared_ptr<ISampler>  sampler_;
    EngineConfig               config_;
    BackendPtr                 backend_;

    // Persistent GPU buffers (pre-allocated, addresses stable for CUDA Graph).
    Tensor buf_condition_;    // [1, S, D]
    Tensor buf_x_t_;          // [1, T_action, action_dim]  — noisy action
    Tensor buf_velocity_;     // [1, T_action, action_dim]  — predicted velocity
    Tensor buf_action_out_;   // [1, T_action, action_dim]  — final action (pinned)

    // Sampler working buffers (passed to sampler to guarantee stable addresses).
    SamplerBuffers sampler_bufs_;

    // Condition cache state.
    // condition_valid_: false forces re-encode on next call.
    // condition_frame_id_: monotonically increasing counter set by the caller
    //   via mark_new_frame(). The engine re-encodes when frame_id changes.
    //   This is more reliable than pointer-based hashing because camera drivers
    //   typically reuse the same pinned buffer for every frame.
    bool     condition_valid_   = false;
    uint64_t condition_frame_id_ = UINT64_MAX;

    // CUDA streams and event for async overlap.
    StreamHandle stream_encode_  = nullptr;
    StreamHandle stream_denoise_ = nullptr;
    void*        encode_done_event_ = nullptr;   // cudaEvent_t

    std::mutex infer_mu_;

    // current frame id seen by the engine (updated by mark_new_frame)
    uint64_t current_frame_id_ = 0;
};

}  // namespace sf
