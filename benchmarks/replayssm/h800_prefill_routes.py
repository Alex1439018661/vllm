# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Capture Qwen3.5 prefill expert routes for offline overlap analysis."""

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np

DEFAULT_MODEL = "/mnt/disk1/senber/Qwen3.5-35B-A3B"
DEFAULT_POOL = (
    "/home/senber/replaySSM/workspace/profile/decode_benchmark/"
    "results/qwen35_large_batch_001/request_pool.json"
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--request-pool", type=Path, default=Path(DEFAULT_POOL))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-prompts", type=int, default=24)
    args = parser.parse_args()

    if args.num_prompts < 1:
        parser.error("--num-prompts must be positive")
    if args.output.exists():
        parser.error(f"output already exists: {args.output}")

    requests = json.loads(args.request_pool.read_text())["requests"]
    if len(requests) < args.num_prompts:
        parser.error("request pool contains fewer prompts than requested")
    requests = requests[: args.num_prompts]
    prompt_lengths = {len(req["prompt_token_ids"]) for req in requests}
    if prompt_lengths != {2048}:
        parser.error(f"expected 2048-token prompts, got lengths {prompt_lengths}")

    # Directly invoking .venv/bin/python does not put .venv/bin on PATH.
    # vLLM's worker invokes the ninja executable while compiling kernels.
    venv_bin = Path(sys.executable).parent
    os.environ["PATH"] = f"{venv_bin}{os.pathsep}{os.environ.get('PATH', '')}"
    if shutil.which("ninja") is None:
        parser.error(f"ninja is unavailable; expected it in {venv_bin}")

    from vllm import LLM, SamplingParams

    start = time.monotonic()
    llm = LLM(
        model=args.model,
        dtype="bfloat16",
        moe_backend="triton",
        enable_return_routed_experts=True,
        enable_prefix_caching=False,
        enforce_eager=True,
        max_model_len=4096,
        max_num_batched_tokens=2048,
        max_num_seqs=1,
        gpu_memory_utilization=0.85,
    )
    params = SamplingParams(max_tokens=1, ignore_eos=True, temperature=0)
    routes = []
    for index, req in enumerate(requests, start=1):
        output = llm.generate(
            [{"prompt_token_ids": req["prompt_token_ids"]}],
            params,
            use_tqdm=False,
        )[0]
        route = output.outputs[0].routed_experts
        if route is None or route.shape != (2048, 40, 8):
            shape = None if route is None else route.shape
            raise RuntimeError(f"{req['request_id']}: unexpected route shape {shape}")
        if not np.all((route >= 0) & (route < 256)):
            raise RuntimeError(f"{req['request_id']}: expert ID outside [0, 255]")
        routes.append(route.astype(np.uint8, copy=False))
        print(
            f"{index}/{len(requests)} {req['request_id']} "
            f"shape={route.shape} elapsed={time.monotonic() - start:.1f}s",
            flush=True,
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        routes=np.stack(routes),
        request_ids=np.array([req["request_id"] for req in requests]),
    )
    print(f"Saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
