#!/bin/bash
# Wrapper: runs align_rdt1b (with its shared libs) inside Docker on GPU 3.
# Args: <ckpt_dir> <input.sf> <output.bin> <t> <freq>

CKPT_HOST="$1"
INPUT_HOST="$2"
OUTPUT_HOST="$3"
T_VAL="$4"
FREQ_VAL="$5"

# Map host paths: /home/ydwang/servoflow/* → /sf_data/*, /tmp/* stays as /tmp/*
if [[ "$CKPT_HOST" == /home/ydwang/servoflow/* ]]; then
    CKPT_DOCKER="/sf_data/${CKPT_HOST#/home/ydwang/servoflow/}"
else
    CKPT_DOCKER="$CKPT_HOST"
fi

# Pass through debug env vars if set.
DEBUG_ARGS=()
[[ -n "$SF_DEBUG_NUM_BLOCKS" ]] && DEBUG_ARGS+=(-e "SF_DEBUG_NUM_BLOCKS=$SF_DEBUG_NUM_BLOCKS")
[[ -n "$SF_DEBUG_SAVE_X" ]]     && DEBUG_ARGS+=(-e "SF_DEBUG_SAVE_X=$SF_DEBUG_SAVE_X")

exec docker run --rm --gpus '"device=3"' \
  -v /home/ydwang/servoflow:/sf_data \
  -v /tmp:/tmp \
  "${DEBUG_ARGS[@]}" \
  servoflow:cpp-latest \
  /workspace/servoflow/build/tests/align_rdt1b \
  "$CKPT_DOCKER" "$INPUT_HOST" "$OUTPUT_HOST" "$T_VAL" "$FREQ_VAL"
