# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import itertools
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from h800_gdn_bc_isolation import measure

from vllm.model_executor.layers.mamba.ops.replayssm_config import (
    override_replayssm_config,
)

out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=False)
rows = []
for d in (7, 8):
    for bv, w, nk, st in itertools.product(
        (32, 64, 128), (1, 2, 4, 8), (2, 4, 8), (1, 2, 3)
    ):
        try:
            with override_replayssm_config("gdn_spec_verify", (bv, w, nk, st)):
                r = measure(48, d, 64, 27, reps=21)
            r["cfg"] = [bv, w, nk, st]
            rows.append(r)
            print(d, [bv, w, nk, st], round(r["us_per_req_layer"], 3), flush=True)
        except Exception as e:
            print(d, [bv, w, nk, st], "FAIL", type(e).__name__, flush=True)
        torch.accelerator.empty_cache()
(out / "raw.json").write_text(json.dumps(rows, indent=1))
