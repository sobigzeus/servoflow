#!/bin/bash
set -e

# Default args
STEPS=10
ITERS=20
MODEL_HOST_PATH="$(pwd)/tests/alignment/sf_checkpoint"
SF_IMAGE="servoflow:bench"

# Parse args
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --steps) STEPS="$2"; shift ;;
        --iters) ITERS="$2"; shift ;;
        --model) MODEL_HOST_PATH="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

echo "Running Comparison with Real Model Checkpoint"
echo "  Model Path: $MODEL_HOST_PATH"
echo "  Steps: $STEPS, Iters: $ITERS"

# Check model path
if [ ! -d "$MODEL_HOST_PATH" ]; then
    echo "Error: Model directory not found at $MODEL_HOST_PATH"
    exit 1
fi

# Check Base Image
if [[ "$(docker images -q servoflow:latest 2> /dev/null)" == "" ]]; then
    echo "Base image servoflow:latest not found. Building..."
    bash docker-build.sh
fi

# Check Bench Image
if [[ "$(docker images -q $SF_IMAGE 2> /dev/null)" == "" ]]; then
    echo "Image $SF_IMAGE not found. Building from Dockerfile.bench..."
    docker build -t "$SF_IMAGE" -f Dockerfile.bench .
fi

# Prepare Temp Dir for Results
TMP_DIR="/tmp/sf_bench_$$"
mkdir -p "$TMP_DIR"
chmod 777 "$TMP_DIR"
HF_JSON="$TMP_DIR/hf_results.json"
SF_JSON="$TMP_DIR/sf_results.json"

# 1. Run Python (PyTorch Manual) Benchmark
echo "---------------------------------------------------"
echo "[1/2] Running PyTorch (Manual) Benchmark..."
docker run --rm --gpus all \
    -v "$(pwd):/workspace/servoflow" \
    -v "$MODEL_HOST_PATH:/model" \
    -v "$TMP_DIR:/tmp_bench" \
    -w /workspace/servoflow \
    "$SF_IMAGE" \
    python3 benchmarks/bench_manual_rdt1b.py \
        --sf-ckpt /model \
        --steps "$STEPS" --iters "$ITERS" \
        --output-json /tmp_bench/hf_results.json

# 2. Run ServoFlow C++ Benchmark
echo "---------------------------------------------------"
echo "[2/2] Running ServoFlow C++ Benchmark..."
# We compile the benchmark inside the container to ensure binary compatibility
docker run --rm --gpus all \
        -v "$(pwd):/workspace/servoflow" \
        -v "$MODEL_HOST_PATH:/model" \
        -v "$TMP_DIR:/tmp_bench" \
        -w /workspace/servoflow \
        "$SF_IMAGE" \
        bash -c "
            rm -rf build && mkdir -p build && cd build && \
            cmake .. -DSF_BUILD_BENCHMARKS=ON -DCMAKE_BUILD_TYPE=Release -DSF_ENABLE_CUDA=ON > /dev/null && \
            make -j\$(nproc) bench_pipeline > /dev/null && \
            export LD_LIBRARY_PATH=\$(pwd):\$LD_LIBRARY_PATH && \
            ./benchmarks/bench_pipeline /model $STEPS $ITERS | tee /tmp_bench/sf_raw.txt
        "

# Extract Mean Latency from C++ Output
# Expected output: "Mean latency: 12.34 ms"
SF_MEAN=$(grep "Mean latency:" "$TMP_DIR/sf_raw.txt" | awk '{print $3}')
if [ -z "$SF_MEAN" ]; then SF_MEAN=0; fi

# 3. Generate Summary
echo "---------------------------------------------------"
python3 -c "
import json
import sys

steps = $STEPS
sf_mean = float('$SF_MEAN')

try:
    with open('$HF_JSON') as f:
        hf = json.load(f)
        hf_mean = hf['denoise_loop']['mean_ms']
except:
    hf_mean = 0

hf_per_step = hf_mean / steps if steps > 0 else 0
sf_per_step = sf_mean / steps if steps > 0 else 0

hf_hz = 1000.0 / hf_mean if hf_mean > 0 else 0
sf_hz = 1000.0 / sf_mean if sf_mean > 0 else 0

speedup_loop = hf_mean / sf_mean if sf_mean > 0 else 0
speedup_step = hf_per_step / sf_per_step if sf_per_step > 0 else 0
speedup_hz = sf_hz / hf_hz if hf_hz > 0 else 0

print(f'\nComparison Results ({steps} steps, Real RDT-1B Model):')
print(f'Metric                  | PyTorch (Manual) | ServoFlow (C++) | Speedup')
print(f'------------------------|------------------|-----------------|--------')
print(f'Loop Latency (ms)       | {hf_mean:16.2f} | {sf_mean:15.2f} | {speedup_loop:.2f}x')
print(f'Per-step Latency (ms)   | {hf_per_step:16.2f} | {sf_per_step:15.2f} | {speedup_step:.2f}x')
print(f'Control Freq (Hz)       | {hf_hz:16.2f} | {sf_hz:15.2f} | {speedup_hz:.2f}x')
"

# Cleanup
rm -rf "$TMP_DIR"
