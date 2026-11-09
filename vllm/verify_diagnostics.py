# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

import json
import os
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import regex as re
import torch

from vllm.triton_utils import tl, triton

_ACTIVE = None
_SCHEDULER_FILE = None
_SCHEDULER_PATH = None

STAT_FIELDS = [
    "token_rows",
    "top_k",
    "gemm1_n",
    "gemm1_k",
    "gemm2_n",
    "gemm2_k",
    "block_m",
    "block_n",
    "block_k",
    "naive_assignment",
    "num_tokens_post_padded",
    "gemm1_em",
    "gemm2_em",
    "gemm1_grid",
    "gemm2_grid",
    "step_id",
    "visits",
]


def enabled():
    return bool(os.environ.get("VLLM_VERIFY_DIAG_DIR"))


def active():
    return _ACTIVE


def scheduler_log(event, **fields):
    """CPU-only log. Never reads a GPU tensor."""
    global _SCHEDULER_FILE, _SCHEDULER_PATH

    directory = os.environ.get("VLLM_VERIFY_DIAG_DIR")
    if not directory:
        return

    root = Path(directory)
    path = root / f"scheduler_{os.getpid()}.jsonl"
    if _SCHEDULER_FILE is None or path != _SCHEDULER_PATH:
        close_scheduler_log()
        root.mkdir(parents=True, exist_ok=True)
        _SCHEDULER_FILE = path.open("a", buffering=1024 * 1024, encoding="utf-8")
        _SCHEDULER_PATH = path

    record = {
        "event": event,
        "wall_ns": time.time_ns(),
        "monotonic_ns": time.perf_counter_ns(),
        **fields,
    }
    _SCHEDULER_FILE.write(json.dumps(record) + "\n")


def flush_scheduler_log():
    if _SCHEDULER_FILE is not None:
        _SCHEDULER_FILE.flush()


def close_scheduler_log():
    global _SCHEDULER_FILE, _SCHEDULER_PATH

    if _SCHEDULER_FILE is not None:
        _SCHEDULER_FILE.close()
    _SCHEDULER_FILE = None
    _SCHEDULER_PATH = None


@contextmanager
def region(diag, name):
    """GPU events are recorded only outside captured model execution."""
    if diag is None or diag.current is None:
        yield
        return

    if hasattr(diag, "before_phase"):
        diag.before_phase(name)

    current = diag.current
    record = {
        "name": name,
        "cpu_start_ns": time.perf_counter_ns(),
    }
    start = torch.Event(enable_timing=True)
    end = torch.Event(enable_timing=True)

    step_id = current["step_id"]
    torch.cuda.nvtx.range_push(f"verify_diag/{step_id}/{name}")
    start.record()
    try:
        yield
    finally:
        end.record()
        torch.cuda.nvtx.range_pop()
        record["cpu_end_ns"] = time.perf_counter_ns()
        record["_events"] = (start, end)
        current["phases"].append(record)


@triton.jit
def _capture_moe(
    Ids,
    PostPadded,
    Step,
    Routes,
    Stats,
    STRIDE_0: tl.constexpr,
    STRIDE_1: tl.constexpr,
    R: tl.constexpr,
    TOP_K: tl.constexpr,
    N1: tl.constexpr,
    K1: tl.constexpr,
    N2: tl.constexpr,
    K2: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    NAIVE: tl.constexpr,
    EM1: tl.constexpr,
    EM2: tl.constexpr,
    GRID1: tl.constexpr,
    GRID2: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.arange(0, BLOCK)
    mask = offsets < R * TOP_K
    ids = tl.load(
        Ids + (offsets // TOP_K) * STRIDE_0 + (offsets % TOP_K) * STRIDE_1,
        mask=mask,
        other=-1,
    )
    tl.store(Routes + offsets, ids, mask=mask)

    tl.store(Stats + 0, R)
    tl.store(Stats + 1, TOP_K)
    tl.store(Stats + 2, N1)
    tl.store(Stats + 3, K1)
    tl.store(Stats + 4, N2)
    tl.store(Stats + 5, K2)
    tl.store(Stats + 6, BM)
    tl.store(Stats + 7, BN)
    tl.store(Stats + 8, BK)
    tl.store(Stats + 9, NAIVE)
    tl.store(Stats + 10, tl.load(PostPadded))
    tl.store(Stats + 11, EM1)
    tl.store(Stats + 12, EM2)
    tl.store(Stats + 13, GRID1)
    tl.store(Stats + 14, GRID2)
    tl.store(Stats + 15, tl.load(Step))
    tl.atomic_add(Stats + 16, 1)


@triton.jit
def _capture_sample_counts(
    Tokens,
    Counts,
    STRIDE_0: tl.constexpr,
    STRIDE_1: tl.constexpr,
    WIDTH: tl.constexpr,
    BLOCK: tl.constexpr,
):
    request_index = tl.program_id(0)
    offsets = tl.arange(0, BLOCK)
    tokens = tl.load(
        Tokens + request_index * STRIDE_0 + offsets * STRIDE_1,
        mask=offsets < WIDTH,
        other=-1,
    )
    count = tl.sum((tokens >= 0).to(tl.int32), axis=0)
    tl.store(Counts + request_index, count)


@triton.jit
def _capture_gdn(
    StateIndices,
    WritePos,
    CacheBase,
    IsFlush,
    Accepted,
    QueryStarts,
    Output,
    N: tl.constexpr,
    INDEX_STRIDE: tl.constexpr,
    ACCEPTED_STRIDE: tl.constexpr,
    QUERY_STRIDE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.arange(0, BLOCK)
    row_mask = row < N
    block_id = tl.load(
        StateIndices + row * INDEX_STRIDE,
        mask=row_mask,
        other=0,
    ).to(tl.int64)
    valid = row_mask & (block_id > 0)

    write_pos = tl.load(WritePos + block_id, mask=valid, other=0)
    cache_base = tl.load(CacheBase + block_id, mask=valid, other=0)
    flush = tl.load(IsFlush + block_id, mask=valid, other=0)
    accepted = tl.load(
        Accepted + row * ACCEPTED_STRIDE,
        mask=row_mask,
        other=0,
    )
    query_start = tl.load(
        QueryStarts + row * QUERY_STRIDE,
        mask=row_mask,
        other=0,
    )
    query_end = tl.load(
        QueryStarts + (row + 1) * QUERY_STRIDE,
        mask=row_mask,
        other=0,
    )

    tl.store(Output + row * 6 + 0, block_id, mask=row_mask)
    tl.store(Output + row * 6 + 1, write_pos, mask=row_mask)
    tl.store(Output + row * 6 + 2, cache_base, mask=row_mask)
    tl.store(Output + row * 6 + 3, flush, mask=row_mask)
    tl.store(Output + row * 6 + 4, accepted, mask=row_mask)
    tl.store(Output + row * 6 + 5, query_end - query_start, mask=row_mask)


class VerifyDiagnostics:
    def __init__(self, runner):
        global _ACTIVE

        parallel = runner.parallel_config
        if (
            parallel.tensor_parallel_size != 1
            or parallel.pipeline_parallel_size != 1
            or parallel.data_parallel_size != 1
            or parallel.enable_expert_parallel
            or parallel.enable_eplb
            or parallel.use_ubatching
            or runner.lora_config is not None
        ):
            raise RuntimeError(
                "Verify diagnostics currently requires single-GPU execution "
                "without LoRA, EPLB, EP, or ubatching."
            )

        from vllm.model_executor.layers.fused_moe.runner.moe_runner import (
            MoERunner,
        )

        self.runner = runner
        self.max_b = runner.scheduler_config.max_num_seqs
        self.d = runner.num_spec_tokens
        self.max_steps = int(os.environ.get("VLLM_VERIFY_DIAG_MAX_STEPS", "1024"))
        self.token_cap = int(
            os.environ.get(
                "VLLM_VERIFY_DIAG_TOKEN_CAP",
                str(2 * self.max_b * (self.d + 1)),
            )
        )
        if self.max_steps < 1 or self.token_cap < self.max_b * (self.d + 1):
            raise ValueError(
                "Diagnostic step capacity must be positive and token capacity "
                "must cover B * (d + 1)."
            )

        config = runner.model_config.hf_text_config
        self.top_k = int(config.num_experts_per_tok)
        self.num_experts = int(
            getattr(config, "num_experts", None) or config.n_routed_experts
        )

        target_ids = {id(module) for module in runner.get_model().modules()}
        entries = []
        for name, module in runner.compilation_config.static_forward_context.items():
            if isinstance(module, MoERunner) and id(module) in target_ids:
                method = module._quant_method
                experts = getattr(
                    getattr(method, "moe_kernel", None),
                    "fused_experts",
                    None,
                )
                if (
                    experts is None
                    or type(experts).__name__ != "TritonExperts"
                    or type(method).__name__ != "UnquantizedFusedMoEMethod"
                    or module.routed_experts.w13_weight.dtype != torch.bfloat16
                    or module.routed_experts.w2_weight.dtype != torch.bfloat16
                ):
                    raise RuntimeError(f"Unsupported MoE implementation: {name}")
                entries.append((int(module.layer_id), name, experts))

        entries.sort()
        if not entries:
            raise RuntimeError("No target Triton MoE layers were found.")
        if len({id(item[2]) for item in entries}) != len(entries):
            raise RuntimeError("Shared expert implementation needs explicit handling.")

        self.layers = [
            {"index": i, "layer_id": layer_id, "name": name}
            for i, (layer_id, name, _) in enumerate(entries)
        ]
        self.num_layers = len(entries)
        self.max_gdn_groups = int(config.num_hidden_layers)

        device = runner.device
        s, layers, t, k = (
            self.max_steps,
            self.num_layers,
            self.token_cap,
            self.top_k,
        )

        # Fixed addresses: captured graphs always write these scratch buffers.
        self.step_gpu = torch.full((1,), -1, dtype=torch.int64, device=device)
        self.route_scratch = torch.full(
            (layers, t, k), -1, dtype=torch.int32, device=device
        )
        self.stat_scratch = torch.zeros(
            (layers, len(STAT_FIELDS)), dtype=torch.int64, device=device
        )

        # Each recorded step owns a different destination.
        self.routes = torch.full(
            (s, layers, t, k), -1, dtype=torch.int32, device=device
        )
        self.stats = torch.zeros(
            (s, layers, len(STAT_FIELDS)), dtype=torch.int64, device=device
        )
        self.contexts = torch.full(
            (s, self.max_b), -1, dtype=torch.int64, device=device
        )
        self.sample_counts = torch.full(
            (s, self.max_b), -1, dtype=torch.int32, device=device
        )
        self.gdn = torch.full(
            (s, self.max_gdn_groups, self.max_b, 6),
            -1,
            dtype=torch.int64,
            device=device,
        )
        self.gdn_masks = torch.zeros(
            (s, self.max_gdn_groups, self.max_b),
            dtype=torch.bool,
            device=device,
        )

        for index, (_, _, experts) in enumerate(entries):
            experts._verify_diag = self
            experts._verify_diag_index = index

        self.armed = False
        self.current = None
        self.records = []
        self.anchor = None
        self.run_name = None
        self.anchor_wall_bounds = None
        _ACTIVE = self

    def _warm_capture_kernels(self):
        """Compile non-model diagnostic kernels before the timed interval."""
        device = self.runner.device
        tokens = torch.zeros((self.max_b, self.d + 1), device=device, dtype=torch.int32)
        output = torch.empty((self.max_b,), device=device, dtype=torch.int32)
        _capture_sample_counts[(self.max_b,)](
            tokens,
            output,
            STRIDE_0=tokens.stride(0),
            STRIDE_1=tokens.stride(1),
            WIDTH=tokens.shape[1],
            BLOCK=triton.next_power_of_2(tokens.shape[1]),
        )
        # GDN's FULL-graph metadata uses d+1-strided state-index rows.
        if self.runner.cache_config.use_replayssm_spec:
            indices = torch.ones(
                (self.max_b, self.d + 1), device=device, dtype=torch.int32
            )
            cursors = torch.zeros(2, device=device, dtype=torch.int32)
            flush = torch.zeros(2, device=device, dtype=torch.int8)
            accepted = torch.ones(self.max_b, device=device, dtype=torch.int32)
            query_starts = torch.arange(
                self.max_b + 1, device=device, dtype=torch.int32
            )
            for n in range(1, self.max_b + 1):
                _capture_gdn[(1,)](
                    indices[:, 0],
                    cursors,
                    cursors,
                    flush,
                    accepted,
                    query_starts,
                    self.gdn[0, 0],
                    N=n,
                    INDEX_STRIDE=indices.stride(0),
                    ACCEPTED_STRIDE=1,
                    QUERY_STRIDE=1,
                    BLOCK=triton.next_power_of_2(n),
                )

    def arm(self, run_name):
        if self.current is not None or self.armed:
            raise RuntimeError("Cannot arm while a step is pending.")
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", run_name):
            raise ValueError("Diagnostic run name must be a plain filename.")
        self.output_dir = Path(os.environ["VLLM_VERIFY_DIAG_DIR"]) / run_name
        self.output_dir.mkdir(parents=True, exist_ok=False)
        self._warm_capture_kernels()
        self.gdn.fill_(-1)
        self.gdn_masks.zero_()
        self.records = []
        self.skipped_steps = []

        # One synchronization before measurement, never once per layer/step.
        torch.accelerator.synchronize()
        self.anchor = torch.Event(enable_timing=True)
        lower = time.time_ns()
        self.anchor.record()
        self.anchor.synchronize()
        upper = time.time_ns()

        self.anchor_wall_bounds = [lower, upper]
        self.run_name = run_name
        self.armed = True

    def begin(self, scheduler_output):
        if self.current is not None:
            raise RuntimeError("Previous diagnostic step was not finished.")

        if not self.armed or not scheduler_output.total_num_scheduled_tokens:
            return

        requests = scheduler_output.verify_diag_requests
        if not requests:
            raise RuntimeError("Scheduler did not provide diagnostic request metadata.")
        if any(value["is_decode"] == 0 for value in requests.values()):
            # This version records pure decode/verify steps.
            self.skipped_steps.append(scheduler_output.verify_diag_id)
            return

        slot = len(self.records)
        if slot >= self.max_steps:
            raise RuntimeError("Increase VLLM_VERIFY_DIAG_MAX_STEPS.")

        self.current = {
            "slot": slot,
            "step_id": scheduler_output.verify_diag_id,
            "requests_at_schedule": requests,
            "phases": [],
        }
        self.step_gpu.fill_(scheduler_output.verify_diag_id)
        self.stat_scratch.zero_()

    def inputs(
        self,
        scheduler_output,
        num_tokens_padded,
        cudagraph_mode,
        spec_decode_metadata,
        attn_metadata,
    ):
        if self.current is None:
            return

        if num_tokens_padded > self.token_cap:
            raise RuntimeError("Increase VLLM_VERIFY_DIAG_TOKEN_CAP.")

        record = self.current
        slot = record["slot"]
        req_ids = list(self.runner.input_batch.req_ids)
        b = len(req_ids)

        record.update(
            req_ids=req_ids,
            active_requests=b,
            target_tokens=[
                int(scheduler_output.num_scheduled_tokens[rid]) for rid in req_ids
            ],
            scheduled_drafts=[
                len(scheduler_output.scheduled_spec_decode_tokens.get(rid, []))
                for rid in req_ids
            ],
            verify_drafts=(
                list(map(int, spec_decode_metadata.num_draft_tokens))
                if spec_decode_metadata is not None
                else [0] * b
            ),
            target_tokens_padded=int(num_tokens_padded),
            cudagraph_mode=str(cudagraph_mode),
            gdn_groups=[],
            request_row_offsets=np.cumsum(
                [0] + [scheduler_output.num_scheduled_tokens[rid] for rid in req_ids]
            ).tolist(),
        )
        if b > self.max_b or len(record["verify_drafts"]) != b:
            raise RuntimeError("Diagnostic request ordering/size mismatch.")

        # This GPU buffer contains async-speculation corrections.
        self.contexts[slot, :b].copy_(self.runner.num_computed_tokens[:b])

        if not isinstance(attn_metadata, dict):
            raise RuntimeError("Ubatches need a separate diagnostic implementation.")

        groups = {}
        for name, metadata in attn_metadata.items():
            if getattr(metadata, "spec_write_pos_d", None) is not None:
                key = id(metadata)
                if key not in groups:
                    groups[key] = [metadata, []]
                groups[key][1].append(name)

        if len(groups) > self.max_gdn_groups:
            raise RuntimeError("Too many GDN metadata groups.")

        for group_index, (metadata, names) in enumerate(groups.values()):
            n = int(metadata.num_spec_decodes)
            if not n:
                continue
            if n > self.max_b:
                raise RuntimeError("Unexpected GDN batch size.")

            indices = metadata.spec_state_indices_tensor[:, 0]
            accepted = metadata.num_accepted_tokens
            query_starts = metadata.spec_query_start_loc

            _capture_gdn[(1,)](
                indices,
                metadata.spec_write_pos_d,
                metadata.spec_cache_base_d,
                metadata.spec_is_flush_d,
                accepted,
                query_starts,
                self.gdn[slot, group_index],
                N=n,
                INDEX_STRIDE=indices.stride(0),
                ACCEPTED_STRIDE=accepted.stride(0),
                QUERY_STRIDE=query_starts.stride(0),
                BLOCK=triton.next_power_of_2(max(n, 1)),
            )
            self.gdn_masks[slot, group_index, :b].copy_(
                metadata.spec_sequence_masks[:b]
            )
            record["gdn_groups"].append(
                {
                    "index": group_index,
                    "layer_names": names,
                    "num_spec_rows": n,
                }
            )

    def moe(
        self,
        layer_index,
        topk_ids,
        sorted_token_ids,
        num_tokens_post_padded,
        w1,
        w2,
        config,
    ):
        # This method is executed during capture. Its GPU operation then
        # executes again on every replay, regardless of self.armed.
        rows, top_k = topk_ids.shape
        if rows > self.token_cap:
            # Prefill uses larger shapes; pure decode is checked in inputs().
            return
        if top_k != self.top_k:
            raise RuntimeError("Unexpected router top-k.")

        bm = int(config["BLOCK_SIZE_M"])
        bn = int(config["BLOCK_SIZE_N"])
        bk = int(config["BLOCK_SIZE_K"])
        assignments = rows * top_k
        naive = sorted_token_ids is None

        if naive:
            em1 = em2 = assignments * bm
        else:
            allocation = sorted_token_ids.numel()
            em1 = min(allocation, assignments * bm) if rows < bm else allocation
            em2 = min(allocation, assignments * bm) if assignments < bm else allocation

        grid1 = triton.cdiv(em1, bm) * triton.cdiv(w1.shape[1], bn)
        grid2 = triton.cdiv(em2, bm) * triton.cdiv(w2.shape[1], bn)

        _capture_moe[(1,)](
            topk_ids,
            num_tokens_post_padded,
            self.step_gpu,
            self.route_scratch[layer_index],
            self.stat_scratch[layer_index],
            STRIDE_0=topk_ids.stride(0),
            STRIDE_1=topk_ids.stride(1),
            R=rows,
            TOP_K=top_k,
            N1=w1.shape[1],
            K1=w1.shape[2],
            N2=w2.shape[1],
            K2=w2.shape[2],
            BM=bm,
            BN=bn,
            BK=bk,
            NAIVE=naive,
            EM1=em1,
            EM2=em2,
            GRID1=grid1,
            GRID2=grid2,
            BLOCK=triton.next_power_of_2(max(assignments, 1)),
        )

    def sampled(self, sampled_token_ids):
        if self.current is None:
            return
        b = self.current["active_requests"]
        if sampled_token_ids.shape[0] != b:
            raise RuntimeError("Sampler batch differs from the recorded target batch.")
        width = sampled_token_ids.shape[1]
        _capture_sample_counts[(b,)](
            sampled_token_ids,
            self.sample_counts[self.current["slot"]],
            STRIDE_0=sampled_token_ids.stride(0),
            STRIDE_1=sampled_token_ids.stride(1),
            WIDTH=width,
            BLOCK=triton.next_power_of_2(max(width, 1)),
        )

    def finish(self):
        if self.current is None:
            return

        slot = self.current["slot"]
        # Same-stream snapshots finish before the next step clears scratch.
        # Draft modules are not bound, so they do not overwrite target routes.
        self.routes[slot].copy_(self.route_scratch)
        self.stats[slot].copy_(self.stat_scratch)

        self.records.append(self.current)
        self.current = None

    def dump(self):
        if self.current is not None:
            raise RuntimeError("A diagnostic step is still pending.")
        if self.anchor is None:
            raise RuntimeError("Diagnostics must be armed before dumping.")

        self.armed = False
        torch.accelerator.synchronize()

        root = self.output_dir
        n = len(self.records)

        arrays = {
            "routes": self.routes[:n].cpu().numpy(),
            "stats": self.stats[:n].cpu().numpy(),
            "contexts": self.contexts[:n].cpu().numpy(),
            "sample_counts": self.sample_counts[:n].cpu().numpy(),
            "gdn": self.gdn[:n].cpu().numpy(),
            "gdn_masks": self.gdn_masks[:n].cpu().numpy(),
        }
        np.savez(root / "buffers.npz", **arrays)

        errors = []
        for record in self.records:
            slot = record["slot"]
            b = record["active_requests"]
            real_rows = sum(record["target_tokens"])

            record["actual_context_before"] = arrays["contexts"][slot, :b].tolist()
            record["sampled_output_counts_before_stop"] = arrays["sample_counts"][
                slot, :b
            ].tolist()
            record["accepted_drafts_before_stop"] = [
                max(int(count) - 1, 0) for count in arrays["sample_counts"][slot, :b]
            ]
            if any(
                count < 1 or count > draft + 1
                for count, draft in zip(
                    record["sampled_output_counts_before_stop"], record["verify_drafts"]
                )
            ):
                errors.append(
                    {"step_id": record["step_id"], "reason": "invalid acceptance count"}
                )

            for phase in record["phases"]:
                start, end = phase.pop("_events")
                phase["gpu_start_ms"] = self.anchor.elapsed_time(start)
                phase["gpu_end_ms"] = self.anchor.elapsed_time(end)
                phase["gpu_elapsed_ms"] = start.elapsed_time(end)

            phases = {phase["name"]: phase for phase in record["phases"]}
            required = {
                "prepare",
                "forward",
                "logits",
                "acceptance",
                "state_postprocess",
            }
            if not required.issubset(phases):
                errors.append(
                    {"step_id": record["step_id"], "reason": "missing phase boundaries"}
                )
            else:
                record["forward_through_acceptance_ms"] = (
                    phases["acceptance"]["gpu_end_ms"]
                    - phases["forward"]["gpu_start_ms"]
                )
                record["forward_through_state_postprocess_ms"] = (
                    phases["state_postprocess"]["gpu_end_ms"]
                    - phases["forward"]["gpu_start_ms"]
                )

            for group in record["gdn_groups"]:
                group_index = group["index"]
                n_spec = group["num_spec_rows"]
                mask = arrays["gdn_masks"][slot, group_index, :b]
                group["req_ids"] = [
                    rid for rid, selected in zip(record["req_ids"], mask) if selected
                ]
                group["rows"] = arrays["gdn"][slot, group_index, :n_spec].tolist()
                if len(group["req_ids"]) != n_spec:
                    errors.append(
                        {
                            "step_id": record["step_id"],
                            "reason": "GDN row mapping mismatch",
                        }
                    )

            record["moe"] = []
            for layer_index, layer in enumerate(self.layers):
                values = arrays["stats"][slot, layer_index]
                stats = dict(zip(STAT_FIELDS, map(int, values)))
                rows = stats["token_rows"]

                if (
                    stats["step_id"] != record["step_id"]
                    or stats["visits"] != 1
                    or rows not in (real_rows, record["target_tokens_padded"])
                    or rows > self.token_cap
                ):
                    errors.append(
                        {
                            "step_id": record["step_id"],
                            "layer_id": layer["layer_id"],
                            "reason": "missing, stale, or repeated MoE capture",
                            "stats": stats,
                        }
                    )
                    continue

                routes = arrays["routes"][slot, layer_index, :rows]
                if (routes < 0).any() or (routes >= self.num_experts).any():
                    errors.append(
                        {
                            "step_id": record["step_id"],
                            "layer_id": layer["layer_id"],
                            "reason": "expert ID out of range",
                        }
                    )
                    continue

                real_counts = np.bincount(
                    routes[:real_rows].reshape(-1),
                    minlength=self.num_experts,
                )
                executed_counts = np.bincount(
                    routes.reshape(-1),
                    minlength=self.num_experts,
                )

                bm = stats["block_m"]
                bn = stats["block_n"]
                if min(bm, bn, stats["block_k"]) <= 0:
                    errors.append(
                        {"step_id": record["step_id"], "reason": "invalid GEMM config"}
                    )
                    continue
                post = stats["num_tokens_post_padded"]
                expected_post = (
                    rows * self.top_k * bm
                    if stats["naive_assignment"]
                    else int(((executed_counts + bm - 1) // bm).sum()) * bm
                )
                if post != expected_post:
                    errors.append(
                        {
                            "step_id": record["step_id"],
                            "layer_id": layer["layer_id"],
                            "reason": "assignment work does not match padding",
                            "actual": post,
                            "expected": expected_post,
                        }
                    )

                active_m = post // bm
                record["moe"].append(
                    {
                        **layer,
                        **stats,
                        "expert_counts_real": real_counts.tolist(),
                        "expert_counts_executed": executed_counts.tolist(),
                        "active_experts_real": int((real_counts > 0).sum()),
                        "graph_padding_assignments": (rows - real_rows) * self.top_k,
                        "expert_padding_slots": post - rows * self.top_k,
                        "active_m_tiles": active_m,
                        "gemm1_active_tiles": (
                            active_m * triton.cdiv(stats["gemm1_n"], bn)
                        ),
                        "gemm2_active_tiles": (
                            active_m * triton.cdiv(stats["gemm2_n"], bn)
                        ),
                        "gemm1_k_iterations": triton.cdiv(
                            stats["gemm1_k"], stats["block_k"]
                        ),
                        "gemm2_k_iterations": triton.cdiv(
                            stats["gemm2_k"], stats["block_k"]
                        ),
                        "gemm1_early_exit_programs": stats["gemm1_grid"]
                        - active_m * triton.cdiv(stats["gemm1_n"], bn),
                        "gemm2_early_exit_programs": stats["gemm2_grid"]
                        - active_m * triton.cdiv(stats["gemm2_n"], bn),
                        "useful_row_fraction": (
                            real_rows * self.top_k / post if post else 0.0
                        ),
                    }
                )

        metadata = {
            "schema_version": 1,
            "model": self.runner.model_config.model,
            "B": self.max_b,
            "d": self.d,
            "token_cap": self.token_cap,
            "layers": self.layers,
            "stat_fields": STAT_FIELDS,
            "gdn_fields": [
                "block_id",
                "history_write_pos",
                "cache_base",
                "is_flush",
                "previous_accepted_including_bonus",
                "query_length",
            ],
            "anchor_wall_ns_bounds": self.anchor_wall_bounds,
            "scope": "pure_decode_verify_excludes_prefill_and_mixed_steps",
            "timing_scope": (
                "CUDA events on the calling stream; CPU times are monotonic ns; "
                "GPU absolute times come from NSys"
            ),
            "gdn_commit_semantics": (
                "commit-at-start uses previous-step acceptance; "
                "first-decode cursors can be reset"
            ),
            "skipped_scheduler_steps": self.skipped_steps,
            "buffers_bytes": sum(array.nbytes for array in arrays.values()),
            "validation_errors": errors,
            "steps": self.records,
        }
        (root / "steps.json").write_text(
            json.dumps(metadata, indent=2),
            encoding="utf-8",
        )
        return {
            "directory": str(root),
            "steps": n,
            "valid": n > 0 and not errors,
            "validation_errors": len(errors),
        }
