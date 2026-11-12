# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Isolate the static history width BC of the GDN ReplaySSM verify kernel.

Actual committed history h is held fixed while buffer (hence BC) varies, so any
time change comes from BC and not from how much history is really replayed.
Only the verify launch is timed (is_flush all zero, launch_mode='verify').
30 distinct state sets are launched inside one CUDA graph (30 GDN layers) so the
state is cold, as in the model.
"""

import argparse
import csv
import json
import statistics
from pathlib import Path

import torch

from vllm.model_executor.layers.fla.ops.gdn_replayssm_spec_decode import (
    gdn_replayssm_spec_decode,
)
from vllm.model_executor.layers.mamba.ops.replayssm_config import (
    get_replayssm_config,
)

H, HV, K, V, LAYERS = 16, 32, 128, 128, 30


def pow2(n):
    return 1 << (n - 1).bit_length()


def measure(B, d, buffer, h, reps=21):
    t = d + 1
    logical = buffer + t
    P = pow2(logical)
    dev = "cuda"
    torch.manual_seed(0)
    layers = [
        (
            torch.randn(B + 1, HV, V, K, device=dev) * 0.01,
            (torch.randn(B + 1, HV, P, V, device=dev) * 0.1).half(),
            (torch.randn(B + 1, H, P, K, device=dev) * 0.1).half(),
            -torch.rand(B + 1, HV, P, device=dev) * 0.1,
        )
        for _ in range(LAYERS)
    ]
    qkv = torch.randn(B * t, 2 * H * K + HV * V, device=dev, dtype=torch.bfloat16) * 0.2
    a = torch.randn(B * t, HV, device=dev, dtype=torch.bfloat16)
    b = torch.randn_like(a)
    a_log = torch.full((HV,), -2.0, device=dev)
    bias = torch.zeros(HV, device=dev)
    out = torch.empty(B * t, HV, V, device=dev, dtype=torch.bfloat16)
    qsl = torch.arange(B + 1, device=dev, dtype=torch.int32) * t
    idx = torch.arange(1, B + 1, device=dev, dtype=torch.int32)
    wp = torch.full((B + 1,), h, device=dev, dtype=torch.int32)
    base = torch.zeros(B + 1, device=dev, dtype=torch.int32)
    fl = torch.zeros(B + 1, device=dev, dtype=torch.int8)

    def run():
        for s0, dc, kc, gc in layers:
            gdn_replayssm_spec_decode(
                qkv,
                a,
                b,
                a_log,
                bias,
                s0,
                dc,
                kc,
                gc,
                out,
                qsl,
                idx,
                wp,
                base,
                fl,
                logical,
                t,
                launch_mode="verify",
            )

    for _ in range(3):
        run()
    torch.accelerator.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    for _ in range(5):
        graph.replay()
    torch.accelerator.synchronize()
    times = []
    for _ in range(reps):
        e0, e1 = torch.Event(enable_timing=True), torch.Event(enable_timing=True)
        e0.record()
        graph.replay()
        e1.record()
        e1.synchronize()
        times.append(e0.elapsed_time(e1) * 1000 / LAYERS)
    cfg = get_replayssm_config("gdn_spec_verify", max_spec_len=t, head_k_dim=K)
    med = statistics.median(times)
    return dict(
        B=B,
        d=d,
        T=t,
        buffer=buffer,
        BC=max(16, pow2(max(1, logical - t))),
        ring_P=P,
        h=h,
        cfg_bv_warps_nk_stages=list(cfg),
        us_per_layer=med,
        us_per_req_layer=med / B,
        min_us=min(times),
        max_us=max(times),
        n=reps,
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", required=True)
    ap.add_argument("--batch", type=int, default=48)
    ap.add_argument("--drafts", default="4,8")
    ap.add_argument("--buffers", default="16,32,64,128")
    args = ap.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    rows = []
    for d in [int(x) for x in args.drafts.split(",")]:
        for buffer in [int(x) for x in args.buffers.split(",")]:
            avail = buffer - d - 1
            hs = sorted({h for h in (0, 7, avail // 2, avail) if 0 <= h <= avail})
            for h in hs:
                r = measure(args.batch, d, buffer, h)
                rows.append(r)
                print(
                    f"d={d} buf={buffer} BC={r['BC']} h={h} "
                    f"cfg={r['cfg_bv_warps_nk_stages']} "
                    f"{r['us_per_layer']:.2f} us/layer "
                    f"{r['us_per_req_layer']:.3f} us/req/layer",
                    flush=True,
                )
                torch.accelerator.empty_cache()
    (out / "raw.json").write_text(json.dumps(rows, indent=1))
    with (out / "summary.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    (out / "environment.json").write_text(
        json.dumps(
            dict(
                torch=torch.__version__,
                device=torch.accelerator.get_device_name(),
                layers=LAYERS,
                note="verify launch only, cold state via 30 distinct layers",
            ),
            indent=1,
        )
    )
    print("wrote", out)


if __name__ == "__main__":
    main()
