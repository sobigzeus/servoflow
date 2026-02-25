#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_rdt1b_align_manual.py

手动跑 RDT-1B 端到端数值对齐测试的小脚本。

功能：
- 调用现有的 test_rdt1b_alignment.py
- 默认使用：
    dtype  = fp32
    device = cuda:3
    ckpt   = /tmp/sf_checkpoint_fp32
    cpp    = tests/alignment/docker_align_rdt1b.sh (Docker 包一层 C++ binary)
- 打印 PyTorch 与 C++ 的对齐结果（max_err / mean_err / cos_sim）

示例：
    python tests/alignment/run_rdt1b_align_manual.py
    python tests/alignment/run_rdt1b_align_manual.py --dtype bf16
    python tests/alignment/run_rdt1b_align_manual.py --no-cpp
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
TEST_SCRIPT = REPO_ROOT / "tests" / "alignment" / "test_rdt1b_alignment.py"
DOCKER_WRAPPER = REPO_ROOT / "tests" / "alignment" / "docker_align_rdt1b.sh"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--dtype",
        default="fp32",
        choices=["fp32", "bf16", "fp16"],
        help="对齐测试使用的数据类型（传给 test_rdt1b_alignment.py）",
    )
    p.add_argument(
        "--device",
        default="cuda:3",
        help="PyTorch 侧使用的 device，例如 cuda:3 / cuda / cpu",
    )
    p.add_argument(
        "--sf-ckpt",
        default="/tmp/sf_checkpoint_fp32",
        help="ServoFlow 转换后的 checkpoint 路径（目录或 model.safetensors 文件）",
    )
    p.add_argument(
        "--no-cpp",
        action="store_true",
        help="只跑 PyTorch 参考（跳过 C++ / Docker 部分）",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if not TEST_SCRIPT.is_file():
        print(f"[ERROR] 找不到测试脚本: {TEST_SCRIPT}", file=sys.stderr)
        sys.exit(1)

    cmd = [
        sys.executable,
        str(TEST_SCRIPT),
        "--dtype",
        args.dtype,
        "--device",
        args.device,
        "--sf-ckpt",
        args.sf_ckpt,
    ]

    if args.no_cpp:
        cmd.append("--no-cpp")
    else:
        # 使用我们已经配置好的 Docker wrapper 作为 C++ 入口
        cmd.extend(["--cpp-bin", str(DOCKER_WRAPPER)])

    print("=" * 66)
    print("  ServoFlow RDT-1B 手动对齐测试")
    print("=" * 66)
    print(f"  repo      : {REPO_ROOT}")
    print(f"  ckpt      : {args.sf_ckpt}")
    print(f"  dtype     : {args.dtype}")
    print(f"  device    : {args.device}")
    print(f"  cpp-bin   : {'SKIP (--no-cpp)' if args.no_cpp else DOCKER_WRAPPER}")
    print("-" * 66)
    print("  实际执行命令：")
    print(" ", " ".join(str(x) for x in cmd))
    print("=" * 66)
    print()

    env = os.environ.copy()
    # 这里不去改 CUDA_VISIBLE_DEVICES，由你外部环境控制。
    proc = subprocess.run(cmd, env=env)
    sys.exit(proc.returncode)


if __name__ == "__main__":
    main()

