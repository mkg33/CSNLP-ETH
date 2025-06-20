#!/usr/bin/env python3
"""
Evaluator for the multi-scale OTStyleModel variant.

Required dir layout:

      pan25_data/
        ├─ easy/   ├─ validation/problem-*.txt
        │          └─ truth-problem-*.json (yes, I moved those files because it was easier; adjust, if needed)
        ├─ medium/ ...
        └─ hard/   ...

  Predictions written to <output-dir>/<difficulty>/solution-problem-*.json
  -> use the official evaluator directly.
"""

from __future__ import annotations
import argparse, json, os, random, sys
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



def sliced_w1(x: torch.Tensor, y: torch.Tensor, n_proj: int) -> torch.Tensor:
    d   = x.size(1)
    v   = torch.randn(n_proj, d, device=x.device)
    v   = v / v.norm(dim=1, keepdim=True)
    xp  = (x @ v.T).sort(dim=0)[0]
    yp  = (y @ v.T).sort(dim=0)[0]
    m   = min(xp.size(0), yp.size(0))
    return (xp[:m] - yp[:m]).abs().mean()



class OTStyleModel(nn.Module):
    PROJ_LIST = (8, 32, 128)

    def __init__(self,
                 encoder: str              = "microsoft/deberta-v3-base",
                 content_dim: int          = 192,
                 style_dim: int            = 128,
                 device: str | torch.device = "cpu"):
        super().__init__()
        self.device = torch.device(device)

        self.tokenizer = AutoTokenizer.from_pretrained(
            encoder, add_prefix_space=True)
        self.encoder   = AutoModel.from_pretrained(encoder)

        hidden2 = self.encoder.config.hidden_size * 2
        self.content_proj = nn.Linear(hidden2, content_dim)
        self.style_proj   = nn.Linear(hidden2, style_dim)
        self.gate         = nn.Parameter(torch.tensor([0.5]))

        self.classifier = nn.Sequential(
            nn.LayerNorm(content_dim + style_dim + 3),
            nn.Linear(content_dim + style_dim + 3, 2),
        )


    def _tokenize(self, s1: List[str], s2: List[str],
                  max_len: int = 128):
        enc = self.tokenizer(
            s1, s2,
            padding=True,
            truncation=True,
            max_length=max_len,
            return_tensors="pt",
            return_offsets_mapping=True,
            return_special_tokens_mask=True,
            return_token_type_ids=False,
        )
        tt = torch.zeros_like(enc["input_ids"])
        for i, spec in enumerate(enc["special_tokens_mask"]):
            sep = (enc["input_ids"][i] == self.tokenizer.sep_token_id
                   ).nonzero(as_tuple=False)
            if sep.numel():
                tt[i, sep[0, 0] + 1:] = 1
        enc["token_type_ids"] = tt
        enc.pop("offset_mapping")
        enc.pop("special_tokens_mask")
        return {k: v.to(self.device) for k, v in enc.items()}


    def forward(self, s1: List[str], s2: List[str]) -> torch.Tensor:
        inp = self._tokenize(s1, s2)
        H   = self.encoder(**inp).last_hidden_state
        tt, m = inp["token_type_ids"], inp["attention_mask"]
        cls = H[:, 0]

        def mmean(h, mask):
            return (h * mask.unsqueeze(-1)).sum(1) / mask.sum(
                1, keepdim=True).clamp(min=1)

        mean0 = mmean(H, (tt == 0) & (m == 1))
        mean1 = mmean(H, (tt == 1) & (m == 1))
        rep   = torch.cat([cls, mean0 - mean1], 1)

        C = self.content_proj(rep)
        S = self.style_proj(rep)
        g = torch.sigmoid(self.gate)
        fused = torch.cat([C * (1 - g), S * g], 1)


        dists = []
        for n_proj in self.PROJ_LIST:
            batch_d = []
            for b in range(H.size(0)):
                a_idx = ((tt[b] == 0) & (m[b] == 1)).nonzero(as_tuple=True)[0]
                b_idx = ((tt[b] == 1) & (m[b] == 1)).nonzero(as_tuple=True)[0]
                batch_d.append(sliced_w1(H[b, a_idx], H[b, b_idx], n_proj))
            dists.append(torch.stack(batch_d))
        d_ot = torch.stack(dists, dim=1)

        logits = self.classifier(torch.cat([fused, d_ot], 1))
        return logits



def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        "Predict changes with the multi-scale OT model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model-path", required=True, help="model (.pt)")
    p.add_argument("--data-root", required=True, help="pan25_data directory")
    p.add_argument("--output-dir", required=True, help="output dir for solutions")
    p.add_argument("--device", choices=["cuda", "cpu"], default="cpu")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()



def load_model(path: str, device: torch.device) -> OTStyleModel:
    state = torch.load(path, map_location=device)

    c_dim = state["content_proj.weight"].shape[0]
    s_dim = state["style_proj.weight"].shape[0]

    model = OTStyleModel(content_dim=c_dim,
                         style_dim=s_dim,
                         device=device)
    model.load_state_dict(state, strict=False)
    model.to(device).eval()
    return model



@torch.no_grad()
def predict_diff(model: OTStyleModel, diff: str,
                 data_root: str, out_root: str) -> None:
    inp_dir = Path(data_root) / diff / "validation"
    out_dir = Path(out_root) / diff
    out_dir.mkdir(parents=True, exist_ok=True)

    for txt in tqdm(sorted(inp_dir.glob("problem-*.txt")),
                    desc=diff, unit="file"):
        pid = txt.stem[8:]
        with txt.open(encoding="utf-8") as fh:
            sents = [ln.strip() for ln in fh if ln.strip()]

        changes: List[int] = []
        for s1, s2 in zip(sents, sents[1:]):
            logits = model([s1], [s2])
            changes.append(int(logits.argmax(1).item()))

        (out_dir / f"solution-problem-{pid}.json").write_text(
            json.dumps({"changes": changes}, indent=2),
            encoding="utf-8")



def main() -> None:
    args   = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print("[INFO] model loading ")
    model = load_model(args.model_path, device)

    for diff in ("easy", "medium", "hard"):
        predict_diff(model, diff, args.data_root, args.output_dir)

    print("[INFO] wrote to", args.output_dir)


if __name__ == "__main__":
    main()
