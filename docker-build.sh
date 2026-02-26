#!/usr/bin/env bash
set -euo pipefail

IMAGE_NAME="${IMAGE_NAME:-servoflow}"
IMAGE_TAG="${IMAGE_TAG:-latest}"
CUDA_ARCHS="${CUDA_ARCHS:-86}"          # sm_86 = RTX 3090
BUILD_TYPE="${BUILD_TYPE:-Release}"

echo "==> Building Docker image: ${IMAGE_NAME}:${IMAGE_TAG}"
echo "    CUDA archs : ${CUDA_ARCHS}"
echo "    Build type : ${BUILD_TYPE}"
echo "    Limits     : 28GB RAM, CPUs 0-6"
echo ""

docker build \
    --memory=28g \
    --cpuset-cpus=0-6 \
    --network=host \
    --build-arg SF_CUDA_ARCHS="${CUDA_ARCHS}" \
    --build-arg CMAKE_BUILD_TYPE="${BUILD_TYPE}" \
    --target builder \
    -t "${IMAGE_NAME}:${IMAGE_TAG}" \
    -f "$(dirname "$0")/Dockerfile" \
    "$(dirname "$0")"

echo ""
echo "==> Build complete: ${IMAGE_NAME}:${IMAGE_TAG}"
echo ""
echo "Run tests inside the container:"
echo "  docker run --rm --gpus all --memory=28g --cpus=7 ${IMAGE_NAME}:${IMAGE_TAG} \\"
echo "    ctest --test-dir /workspace/servoflow/build --output-on-failure"
echo ""
echo "Interactive shell:"
echo "  docker run --rm -it --gpus all --memory=28g --cpus=7 ${IMAGE_NAME}:${IMAGE_TAG} bash"
