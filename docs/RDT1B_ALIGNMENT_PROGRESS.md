# ServoFlow RDT‑1B 对齐与性能优化总结

> 记录截至目前为止：RDT‑1B 在 ServoFlow 上的端到端数值对齐过程、已完成工作、当前性能评测结果，以及后续性能目标与优化方向。

---

## 1. 目标与整体阶段

- **阶段一：数值对齐**
  - 目标：在相同输入下，ServoFlow C++/CUDA 实现与 PyTorch 参考实现输出一致（或在极小容差内）。
  - 范围：完整 RDT‑1B 前向（语言条件 + 图像条件 + state/action 轨迹 → 未来 64 步 action）。

- **阶段二：性能评测与瓶颈定位**
  - 目标：在完成数值对齐基础上，对比 ServoFlow vs PyTorch 的推理耗时，区分：
    - 端到端（含 Docker/进程/IO）的整体耗时。
    - 单次 forward 的“kernel‑only” GPU 时间。

- **阶段三：性能优化（目标：≥ PyTorch 性能的 2×）**
  - 在保持数值精度前提下，利用 FlashAttention、半精度、kernel 融合等手段，显著降低 kernel‑only latency。

目前 **阶段一已经完成**，阶段二的评测与分析也已经完成，阶段三刚刚启动（已打通 FlashAttention 编译路径，等待 bf16 checkpoint 做进一步评测）。

---

## 2. 数值对齐：从最初错误到最终 PASS

### 2.1 PyTorch 参考实现与对齐脚本

新建了一个纯 PyTorch 的参考实现与测试脚本：

- `tests/alignment/test_rdt1b_alignment.py`
  - 实现了完整 RDT‑1B 前向，包括：
    - `RDT1BRef`：lang/img/state adaptor、RDT blocks（SelfAttn + CrossAttn + FFN）、final layer。
    - `sincos_emb`：与 C++ `TimestepEmbedding::build_sincos_table` 完全一致的 \[cos, sin] 布局。
    - `SelfAttn` / `CrossAttn`：
      - 使用 `scaled_dot_product_attention`，并强制 `enable_flash=False, enable_math=True`，以便与 C++ naive attention 更公平对齐。
  - 测试流程：
    1. 加载 ServoFlow 转换后的 `model.safetensors` 到 PyTorch 参考模型。
    2. 生成固定随机输入（语言、图像、state、noisy_action、t、freq，种子固定）。
    3. 前向得到 `ref_out`。
    4. 将输入以 float32 形式保存到 `inputs.safetensors`。
    5. 调用 C++ 对齐 binary（经 `docker_align_rdt1b.sh`），得到 ServoFlow 输出，反序列化为 `cpp_out`。
    6. 比较 `ref_out` 与 `cpp_out`：
       - `max|err|`、`mean|err|`、`rel_err`、`cosine_similarity`。
       - 通过条件：`max|err| < 0.05` 且 `cos_sim > 0.9999`。

为方便用户手动运行，对齐脚本又封装了一层：

- `tests/alignment/run_rdt1b_align_manual.py`
  - 默认参数：
    - `--dtype fp32`
    - `--device cuda:3`
    - `--sf-ckpt /tmp/sf_checkpoint_fp32`
    - `--cpp-bin tests/alignment/docker_align_rdt1b.sh`
  - 可以快速跑：
    - 只 PyTorch：`--no-cpp`
    - 完整 C++ 对齐：默认 `cpp-bin` 即 Docker wrapper。

### 2.2 C++ 对齐入口与 Docker 封装

#### 对齐 binary：`tests/alignment/align_rdt1b.cpp`

- 功能：从 ServoFlow checkpoint 和输入 safetensors 中加载数据，调用：
  ```cpp
  Tensor out = model.forward_raw(lang_tok, img_tok, state_tok,
                                 noisy_act, t_step, freq);
  ```
  并将输出写成简单的 float32 二进制格式（带 magic 和 shape）。
- 内部逻辑：
  - 读取 `config.json` → `RDT1BConfig`。
  - 构建 `RDT1BModel` 并加载权重（来自 safetensors）。
  - 使用 `host_f32_to_gpu` 将 CPU float32 输入上传到 GPU，并 cast 到 `cfg.compute_dtype`。
  - 调用 `forward_raw`，得到 `[1, 64, 128]` 输出。
  - 使用 `gpu_to_host_f32` 将输出拷回 CPU，写 `output.bin`。

#### Docker 封装：`tests/alignment/docker_align_rdt1b.sh`

- 作用：在 Docker 容器内运行 `align_rdt1b`，并固定使用 `cuda:3`（避免与其它服务冲突）。
- 主要实现细节：
  - 设备绑定：`docker run --rm --gpus "device=3" ...`
  - 路径映射：
    - `/home/ydwang/servoflow` → `/sf_data`
    - `/tmp` → `/tmp`
  - checkpoint 路径规则：
    - 若 `CKPT_HOST` 在 `/home/ydwang/servoflow/*` 下，则映射为 `/sf_data/...`；
    - 否则（例如 `/tmp/sf_checkpoint_fp32`），直接原样传入（容器中通过 `/tmp` 共享）。
  - 额外支持调试环境变量（用于 forward_raw 内部调试）：
    - `SF_DEBUG_NUM_BLOCKS`：限制前向只跑前 N 个 transformer blocks。
    - `SF_DEBUG_SAVE_X`：在进 block 之前，保存 `x_before_blocks` 到指定路径的二进制文件（float32 + shape）。

### 2.3 主要 bug 与修复过程

对齐过程暴露出若干关键 bug，按时间顺序大致如下。

#### Bug 1：`elementwise_binary` 不支持广播，触发非法内存访问

- 现象：
  - 早期运行 C++ 对齐 binary，经常在 RDT blocks 内部报 `CUDA illegal memory access`。
  - 通过加 `sync_device()` 和日志，发现问题出在某些 `add` 操作（矩阵 + bias）上。
- 根因：
  - `backend->add` 使用的 `elementwise_binary` 假设 `a.numel() == b.numel()`。
  - 在 `matrix[N, D] + bias[D]` 场景下，将 `bias[D]` 当作了 `bias[N*D]`，造成 OOB 访问。
- 修复：
  - 在 `src/backend/cuda/ops/elementwise.cuh` 中新增：
    - `binary_kernel_bcast`：按 `b[idx % b_n]` 广播 `b`。
    - `elementwise_binary` 增强逻辑：
      - 若 `b.numel() == a.numel()` → 原始逐元素 kernel。
      - 否则，要求 `a.numel() % b.numel() == 0`，走 `binary_kernel_bcast`。

#### Bug 2：Attention 输出布局错误，head‑major/seq‑major 混用

- SelfAttention/CrossAttention 的 output 在 C++ 中为 `[B, H, S, D]`（head‑major），而后续 projection 期望 `[B, S, H*D]`（seq‑major）。
- 初稿实现中直接使用 `view` 把 `[B, H, S, D]` 视作 `[B*S, D]`，等价于错序重排。
- 修复：
  - 在 `src/backend/cuda/ops/split_ops.cuh` 中新增：
    - `seq_to_head`：`[B, S, H*D] -> [B, H, S, D]`。
    - `head_to_seq`：`[B, H, S, D] -> [B, S, H*D]`。
  - SelfAttention:
    - Q/K/V 按 head‑major 布局做 attention；
    - 注意力输出 `attn_out[B,H,S,D]` 通过 `head_to_seq` 转成 `attn_seq[B,S,H*D]`，再做投影。
  - CrossAttention:
    - Q 从 seq‑major `[B,S,H*D]` 经 `seq_to_head` 变为 `[B,H,S,D]`；
    - K/V 用 `split_kv_kernel` 得到 `[B,H,Sk,D]`；
    - 输出同样走 `head_to_seq` 再投影。

#### Bug 3：Sinusoidal embedding 顺序不一致

- C++ 的 `TimestepEmbedding::build_sincos_table` 采用的布局为 \[cos, sin]。
- 早期 Python 参考的 `sincos_emb` 为 \[sin, cos]，导致 timestep/freq 嵌入错位。
- 修复：
  - Python 侧 `sincos_emb` 改为与 C++ 一致的 \[cos, sin]。

#### Bug 4：`cat_kernel` 只支持 dim=0，导致 state_adaptor 输入错误

- 现象：
  - 即便所有 block 忽略（`SF_DEBUG_NUM_BLOCKS=0`），C++ 输出与 PyTorch 仍有较大偏差。
  - 进一步比较 `x_before_blocks`（即加完 positional embedding，但尚未进入任何 block）发现：
    - `t_emb`（tok0）与 `freq_emb`（tok1）高度一致。
    - 从 tok2（state/action 序列）开始差异巨大。
- 根因排查：
  - Python 侧 `traj_in` 构造：
    ```python
    state_in = torch.cat([state, noisy], dim=1)   # [1, 65, A]
    traj_in  = torch.cat([state_in, mask], dim=2) # [1, 65, 2A]
    ```
  - C++ 侧：
    ```cpp
    Tensor st_2d = st.view({1, A});          // [1,128]
    Tensor na_2d = na.view({T, A});          // [64,128]
    cat({st_2d, na_2d}, state_action_2d, 0); // [65,128]

    mask_full = ones [65,128];
    cat({state_action_2d, mask_full}, traj_2d, 1); // dim=1
    ```
  - 但 CUDA 实现 `cat_kernel` 是：
    ```cpp
    void cat_kernel(const std::vector<Tensor>& inputs, Tensor& out,
                    int64_t /*dim*/, cudaStream_t stream) {
        // 简单按内存顺序 memcpy，忽略 dim，只等价于 dim=0。
    }
    ```
  - 即：对 `dim=1` 的调用完全错误，导致 state_adaptor 的输入特征严重错乱。
- 修复：
  - 重写 `src/backend/cuda/ops/elementwise.cu` 中的 `cat_kernel`：
    - 支持 `dim=0`：
      - 保持原行为：按输入顺序整体拼接，单次 memcpy 即可。
    - 支持 `dim=1`：
      - 假定所有输入 shape 为 `[rows, cols_i]`，输出为 `[rows, sum(cols_i)]`。
      - 对每一行 `r`，依次从每个输入的第 `r` 行 memcpy 到输出的第 `r` 行对应偏移位置。
    - 其它 dim 直接抛错：`cat_kernel: only dim=0 and dim=1 are supported`。
  - 修复后，再次比较：
    - `x_before_blocks`：
      - tok0/tok1 完全一致；
      - tok2..（state/action）在去掉 positional embedding 后，与 Python `state_adaptor` 输出完全对齐。

### 2.4 最终对齐结果

在 fp32、`cuda:3` 下，以 `/tmp/sf_checkpoint_fp32` 为权重，重新运行完整对齐测试：

- 命令：
  ```bash
  cd /home/ydwang/servoflow
  python tests/alignment/run_rdt1b_align_manual.py \
    --dtype fp32 \
    --device cuda:3 \
    --sf-ckpt /tmp/sf_checkpoint_fp32
  ```
- 输出（核心指标）：
  ```text
  [PyTorch ref vs ServoFlow C++]  shape=[1, 64, 128]
    max|err|  = 2.7978e-04
    mean|err| = 8.6616e-06
    rel_err   = 2.2512e-03
    cos_sim   = 1.00000119

  ALIGNMENT TEST RESULT
  ✓ PASS   max_err=2.80e-04 < 5.00e-02  cos=1.000001
  ```

**结论：RDT‑1B 在 ServoFlow C++/CUDA 实现与 PyTorch 参考实现之间的端到端数值对齐已经完成。**

---

## 3. 性能评测与对比

### 3.1 端到端耗时（含 Docker / 进程 / IO）

> 注意：这一节只是说明“当前部署方式”的整体开销，并不能直接反映内核快慢。

同一输入、同一 GPU（`cuda:3`），用 Python 侧驱动进行 20 次测试：

- **PyTorch 参考**：
  - 一次加载权重，多次 forward。
  - 平均单次 forward（含 GPU 计算，`cuda.synchronize`）：
    - **约 96 ms / 次**。

- **ServoFlow C++（经 `docker_align_rdt1b.sh` 调用 `align_rdt1b`）**：
  - 每次都：
    - `docker run --rm ... servoflow:cpp-latest ...`
    - 重新加载 1B 级别权重到 GPU。
    - 从磁盘读写输入/输出文件。
    - 跑一次 `forward_raw`。
  - 平均单次整体耗时：
    - **约 11.95 s / 次**。

原因一目了然：ServoFlow 这一路把所有启动 & IO 成本都算进去了，而 PyTorch 只有一次模型加载，多次 forward。后续部署时应改为 **常驻进程 + 复用模型**，或者通过 RPC/Service 形式提供接口。

### 3.2 Kernel‑only 耗时（仅算一次 forward 的 GPU 时间）

这一节才是“公平”的内核性能对比：单进程、复用模型、用 CUDA events 对一次 forward 计时。

#### PyTorch 参考 (`RDT1BRef`)

- 配置：fp32、batch=1、S=67、`cuda:3`。
- 测试方式：CUDA events（5 次 warmup + 50 次计时）。
- 实测结果：
  - **平均：约 69 ms / 次**。

#### ServoFlow C++ (`bench_rdt1b_kernel.cpp`)

- 文件：`tests/alignment/bench_rdt1b_kernel.cpp`。
- 功能：
  - 在容器中启动一次可执行：
    - `bench_rdt1b_kernel <ckpt_dir> <inputs.safetensors> [warmup] [iters]`
  - 行为：
    - 构建 `RDT1BModel`，加载权重一次。
    - 上传输入一次。
    - 若干次 warmup，不计时。
    - 用 CUDA events 记录 `iters` 次 `forward_raw` 的耗时。
    - 输出平均/最小/最大 latency。
- 配置：fp32、batch=1、S=67、`cuda:3`。
- 实测结果：
  - **平均：约 323 ms / 次**。

#### 小结

在相同任务 & 相同精度下，目前 **ServoFlow C++ 的 kernel‑only 延迟约是 PyTorch 的 4.5× 左右**。

这已经排除了 Docker/起进程/IO 影响，说明在当前 fp32 + naive attention 路径上，ServoFlow 性能还有较大优化空间。

---

## 4. 性能差距的主要原因分析

结合代码实现和结构特性，粗略分析当前差距的主要来源：

### 4.1 Attention：naive kernel vs PyTorch SDPA

- 当前 C++ 路径在 fp32 下使用的是 `fallback_attention`：
  - 每个 query token 一个线程，显式三重循环（Sk × D），在线 softmax。
  - 算法上正确，但**没有 head/block 级别的高度优化**，属于“参考实现”。
- PyTorch 的 `scaled_dot_product_attention` 即便关闭 Flash，也会走高度优化的 math SDPA 内核（多维并行、warp 利用、shared memory 等）。
- RDT‑1B 每个 block 有 self‑attn + cross‑attn，共 28 blocks，attention FLOPs 本身就占大头：
  - naive kernel 在这里拖慢整个前向。

### 4.2 kernel 过碎、缺乏融合

ServoFlow 当前实现以“清晰易读”优先：

- 每一层/步骤的典型模式：
  - `alloc` 一个输出 Tensor（`backend_->alloc`）。
  - 调一次 `gemm` / `add` / `rms_norm` / `gelu` / `attention`。
  - 再 `alloc` 下一步的中间 tensor。
- 对一个 RDTBlock 来说，roughly 包含：
  - 3× RMSNorm
  - 2× Attention（self + cross）
  - 1× FFN（2× GEMM + GELU）
  - 多个 layout 变换 / cat / add。
- 在 batch=1、S=67 的场景中：
  - 纯算子 FLOPs 不算巨大，
  - 但 **kernel launch 开销**、allocator 开销、以及小 tensor 上的 memcpy/变换成本被放大。

相比之下，PyTorch 在很多路径上使用了：

- 融合算子（例如 GEMM + bias + activation）、更复杂的 kernel 调度策略。

### 4.3 内存操作与 layout 变换还不够精简

- `cat_kernel`：
  - 虽然逻辑正确，但 dim=1 上实现为多次逐行 `cudaMemcpyAsync`。
  - 对于 `[65,128] + [65,128] -> [65,256]` 的简单场景，其实完全可以写成 1–2 个 kernel 完成，而不是大量小 memcpy。
- `seq_to_head` / `head_to_seq` / `split_qkv` / `split_kv`：
  - 目前是“每元素一个线程、直观 but naive 的重排”，还没做更进一步的 vectorization/coalescing。
- 同时配合一些全局 `sync_device`（为 debug/对齐而保留），整体 pipeline 的 overlap 机会有限。

---

## 5. 已经落地的改动与提交情况

### 5.1 核心代码改动

主要涉及以下文件（已通过一次 commit 提交到 `main`，不包含模型权重）：

- `include/servoflow/models/rdt1b/config.h`
- `include/servoflow/models/rdt1b/dit_block.h`
- `include/servoflow/models/rdt1b/rdt1b.h`
- `src/models/rdt1b/config.cpp`
- `src/models/rdt1b/dit_block.cu`
- `src/models/rdt1b/rdt1b.cpp`
- `src/backend/cuda/ops/elementwise.cuh`
- `src/backend/cuda/ops/elementwise.cu`
- `src/backend/cuda/ops/split_ops.cuh`
- `tests/CMakeLists.txt`
- `tests/alignment/*`（对齐脚本、Docker wrapper、kernel benchmark）
- `Dockerfile` / `Dockerfile.bench`（确保编译并打包 `align_rdt1b` & `bench_rdt1b_kernel`）
- `tools/convert/hf_to_servoflow.py`（支持 fp32/bf16 配置、输出 ServoFlow 格式 config）。

### 5.2 提交信息

- commit 标题：
  - `Add RDT-1B alignment tests and CUDA kernel fixes`
- 内容摘要：
  - 新增完整的 RDT‑1B PyTorch 参考实现和对齐测试脚本。
  - 修复 `elementwise_binary` 广播、attention 布局、`cat_kernel` 等 C++/CUDA bug。
  - 新增 `bench_rdt1b_kernel`，用于测量 `forward_raw` 的 kernel‑only latency。

模型文件（HF 原始 checkpoint、ServoFlow 转换后的 `model.safetensors`、`/tmp/sf_checkpoint_*` 等）都未纳入 Git。

---

## 6. 下一步计划与性能目标

整体目标：**在数值对齐的前提下，使 ServoFlow RDT‑1B 的 kernel‑only 延迟达到或优于 PyTorch 的 2× 性能。**

以当前 fp32 baseline 为例（约 69 ms/次）：

- 目标区间：**≈ 30–40 ms/次**。

### 6.1 启用半精度主干 + FlashAttention（最大收益）

1. 使用 `tools/convert/hf_to_servoflow.py` 生成 bf16 ServoFlow checkpoint：
   ```bash
   python tools/convert/hf_to_servoflow.py \
     --input robotics-diffusion-transformer/rdt-1b \
     --output /tmp/sf_checkpoint_bf16 \
     --dtype bf16
   ```
   > 当前在本机尝试时，受到 HuggingFace cache 锁文件权限限制，需要在本地清理/调整权限后由用户再次运行。

2. 使用同样的 `bench_rdt1b_kernel` 在 `/tmp/sf_checkpoint_bf16` 上测：
   - 在 Docker 中构建时已 `-DSF_USE_FLASH_ATTN=ON`，bf16/fp16 dtype 下 attention 会走 FlashAttention v2。
   - 通过输入 bf16 + FlashAttention 内核，目标是首先将 kernel‑only 延迟从 ~323 ms 降到接近 100 ms 以下。

3. 同步 PyTorch 参考：
   - 在 PyTorch 侧使用 amp/bf16 配置 + `scaled_dot_product_attention` 对齐，确保数值误差仍在可接受范围（重新跑对齐测试）。

### 6.2 Kernel 融合与内存复用

在 FlashAttention/bf16 路径上稳定之后，进一步优化：

- 在 `RDTBlock` 上：
  - 为自注意/cross 注意/FFN 引入简单的 per‑stream workspace，复用中间 tensor，减少 `backend->alloc` 次数。
  - 对常见 op 序列（`gemm + add_bias + gelu`、`rms_norm + add_residual`）加入融合实现或专用 kernel。

- 在 adaptor（`lang_adaptor` / `img_adaptor` / `state_adaptor`）上：
  - 将多层 MLP 链简化为少量更大的 kernel，降低 kernel launch 与访存开销。

- 在 layout 变换与 `cat_kernel` 上：
  - 为常见形状（如 `[65,128]+[65,128]`）提供特化的高效 kernel，而非多次 memcpy。

### 6.3 持续 benchmark 与分模块剖析

- 使用已经存在的：
  - `bench_rdt1b_kernel`（C++ kernel‑only）
  - PyTorch CUDA events benchmark（Python kernel‑only）
  对比不同 dtype（fp32 / bf16 / fp16）与不同配置（FlashAttention on/off）。

- 若有需要，可在 `forward_raw` 内部增加更细粒度的计时（例如：
  - adaptor（lang/img/state）
  - 28 blocks 自注意 / cross 注意 / FFN
  - final layer
 ），逐步定位哪个模块仍是瓶颈。

---

## 7. 小结

- RDT‑1B 在 ServoFlow C++/CUDA 路径上已经实现了**严格的端到端数值对齐**，fp32 下与 PyTorch 参考的误差约 `2.8e-4`，cosine similarity 约 `1.000001`。
- 当前 fp32 kernel‑only 性能约为 PyTorch 的 4.5× 延迟，主要原因包括：attention 仍走 naive kernel、kernel 过碎、内存和 layout 操作未高度收敛。
- 已经搭建了对齐与 benchmark 的完整基础设施，并开启了 FlashAttention 编译；下一阶段重点是：
  - 准备 bf16 ServoFlow checkpoint。
  - 基于 bf16 + FlashAttention 做内核测量。
  - 通过 kernel 融合和内存复用，将 ServoFlow 的 kernel‑only latency 压到 PyTorch fp32 的 2× 甚至更优。

