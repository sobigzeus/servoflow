// SPDX-License-Identifier: Apache-2.0
#include "servoflow/engine/inference_engine.h"

#include <chrono>
#include <functional>
#include <mutex>
#include <stdexcept>

// For CUDA event management (only included when CUDA backend is active).
#ifdef SF_CUDA_ENABLED
#  include <cuda_runtime.h>
#  define SF_CUDA_EVENT_CREATE(ev) \
       cudaEventCreateWithFlags(reinterpret_cast<cudaEvent_t*>(&(ev)), cudaEventDisableTiming)
#  define SF_CUDA_EVENT_RECORD(ev, stream) \
       cudaEventRecord(reinterpret_cast<cudaEvent_t>(ev), \
                       reinterpret_cast<cudaStream_t>(stream))
#  define SF_CUDA_STREAM_WAIT_EVENT(stream, ev) \
       cudaStreamWaitEvent(reinterpret_cast<cudaStream_t>(stream), \
                           reinterpret_cast<cudaEvent_t>(ev), 0)
#  define SF_CUDA_EVENT_DESTROY(ev) \
       cudaEventDestroy(reinterpret_cast<cudaEvent_t>(ev))
#else
#  define SF_CUDA_EVENT_CREATE(ev)            (void)0
#  define SF_CUDA_EVENT_RECORD(ev, stream)    (void)0
#  define SF_CUDA_STREAM_WAIT_EVENT(s, ev)    (void)0
#  define SF_CUDA_EVENT_DESTROY(ev)           (void)0
#endif

namespace sf {

InferenceEngine::InferenceEngine(std::shared_ptr<IVLAModel> model,
                                  std::shared_ptr<ISampler>  sampler,
                                  EngineConfig               config)
    : model_(std::move(model)),
      sampler_(std::move(sampler)),
      config_(std::move(config)) {

    backend_ = get_backend(config_.device);

    stream_encode_  = backend_->create_stream();
    stream_denoise_ = backend_->create_stream();
    SF_CUDA_EVENT_CREATE(encode_done_event_);

    preallocate_buffers();
}

InferenceEngine::~InferenceEngine() {
    SF_CUDA_EVENT_DESTROY(encode_done_event_);
    backend_->destroy_stream(stream_encode_);
    backend_->destroy_stream(stream_denoise_);
}

void InferenceEngine::preallocate_buffers() {
    DType dt  = config_.compute_dtype;
    int64_t T = model_->action_horizon();
    int64_t A = model_->action_dim();

    // All action-space buffers are pre-allocated with stable addresses.
    // Addresses must not change between CUDA Graph capture and replay.
    buf_x_t_      = backend_->alloc(Shape({1, T, A}), dt);
    buf_velocity_ = backend_->alloc(Shape({1, T, A}), dt);

    // Wire sampler buffers to the same pre-allocated tensors.
    // This is what enables CUDA Graph capture in FlowMatchingSampler.
    sampler_bufs_.x_t      = buf_x_t_;
    sampler_bufs_.velocity = buf_velocity_;

    // Output in pinned host memory for fast D2H.
    if (config_.pinned_output) {
        buf_action_out_ = backend_->alloc_pinned(Shape({1, T, A}), DType::Float32);
    } else {
        buf_action_out_ = backend_->alloc(Shape({1, T, A}), DType::Float32);
    }

    // condition buffer allocated on first encode (size depends on model config).
}

void InferenceEngine::load_weights(const std::string& path) {
    // Delegate to model; engine is not responsible for weight format.
    // (Model implementations handle safetensors parsing.)
    (void)path;
    throw std::runtime_error("load_weights: implement in concrete model subclass");
}

void InferenceEngine::invalidate_condition_cache() {
    std::lock_guard<std::mutex> lk(infer_mu_);
    condition_valid_ = false;
}

void InferenceEngine::mark_new_frame(uint64_t frame_id) {
    std::lock_guard<std::mutex> lk(infer_mu_);
    current_frame_id_ = frame_id;
}

void InferenceEngine::empty_cache() {
    backend_->empty_cache();
}

VLAOutput InferenceEngine::infer(const VLAInput& input) {
    std::lock_guard<std::mutex> lk(infer_mu_);

    auto t0 = std::chrono::steady_clock::now();

    // ── 1. Condition encoding (on stream_encode_) ─────────────────────────
    // Cache invalidation uses a monotonic frame_id supplied by the caller via
    // mark_new_frame(). This is reliable even when the camera driver reuses
    // the same pinned buffer (same pointer, new content) — a case where
    // pointer-based hashing would silently use stale condition embeddings.
    bool need_encode = !condition_valid_
                    || !config_.cache_condition
                    || (current_frame_id_ != condition_frame_id_);

    if (need_encode) {
        Tensor new_cond = model_->encode_condition(input, backend_, stream_encode_);

        // Allocate condition buffer if shape changed (first call or model change).
        if (!buf_condition_.is_valid()
            || buf_condition_.shape() != new_cond.shape()) {
            buf_condition_ = backend_->alloc(new_cond.shape(),
                                             new_cond.dtype(),
                                             stream_encode_);
        }
        backend_->copy(buf_condition_, new_cond, stream_encode_);
        condition_valid_    = true;
        condition_frame_id_ = current_frame_id_;

        // Signal stream_denoise_ to wait until encoding is done.
        SF_CUDA_EVENT_RECORD(encode_done_event_, stream_encode_);
        SF_CUDA_STREAM_WAIT_EVENT(stream_denoise_, encode_done_event_);
    }

    // ── 2. Sample initial noise (on stream_denoise_) ──────────────────────
    // In production, use curandGenerateNormal; here we mark it as a TODO
    // to keep the engine agnostic of the RNG implementation.
    // TODO: replace with backend_->randn(buf_x_t_, stream_denoise_)
    backend_->fill(buf_x_t_, 0.f, stream_denoise_);  // placeholder

    // ── 3. Denoising loop via sampler ─────────────────────────────────────
    // Wrap model's denoise_step as a DenoiseFn closure.
    DenoiseFn denoise_fn = [this](const Tensor& x_t, float t,
                                   const Tensor& cond, Tensor& vel,
                                   StreamHandle stream) {
        model_->denoise_step(x_t, t, cond, vel, backend_, stream);
    };

    Schedule sched;
    sched.num_steps = config_.num_denoise_steps;

    Tensor raw_action = sampler_->sample(
        buf_x_t_, buf_condition_, denoise_fn, sched,
        backend_, stream_denoise_, &sampler_bufs_);

    // ── 4. Decode action + D2H copy ──────────────────────────────────────
    Tensor decoded = model_->decode_action(raw_action, backend_, stream_denoise_);

    // Cast to fp32 if needed, then copy to pinned host buffer.
    if (decoded.dtype() != DType::Float32) {
        Tensor fp32 = backend_->alloc(decoded.shape(), DType::Float32, stream_denoise_);
        backend_->cast(decoded, fp32, stream_denoise_);
        backend_->copy(buf_action_out_, fp32, stream_denoise_);
    } else {
        backend_->copy(buf_action_out_, decoded, stream_denoise_);
    }

    // Synchronise: wait for D2H to complete before returning to caller.
    backend_->sync_stream(stream_denoise_);

    auto t1 = std::chrono::steady_clock::now();
    double latency_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    VLAOutput out;
    out.actions    = buf_action_out_;
    out.latency_ms = latency_ms;
    return out;
}

}  // namespace sf
