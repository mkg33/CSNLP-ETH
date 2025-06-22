#!/usr/bin/env python3
"""
Evaluator for the sliced OT variant.
"""

from __future__ import annotations
import argparse, json, os, subprocess, sys
from typing import List

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm


def sliced_w1(x: torch.Tensor, y: torch.Tensor, n_proj: int) -> torch.Tensor:
    d = x.size(1)
    v = torch.randn(n_proj, d, device=x.device)
    v = v / v.norm(dim=1, keepdim=True)
    xp = (x @ v.t()).sort(dim=0)[0]
    yp = (y @ v.t()).sort(dim=0)[0]
    m = min(xp.size(0), yp.size(0))
    return torch.abs(xp[:m] - yp[:m]).mean()


class OTStyleModel(nn.Module):
    def __init__(self,
                 encoder: str = "microsoft/deberta-v3-base",
                 content_dim: int = 192,
                 style_dim: int = 128,
                 n_proj: int = 64,
                 device: str | torch.device = "cpu"):
        super().__init__()
        self.device = torch.device(device)
        self.n_proj = n_proj

        self.tokenizer = AutoTokenizer.from_pretrained(encoder, add_prefix_space=True)
        self.encoder   = AutoModel.from_pretrained(encoder)

        hidden2 = self.encoder.config.hidden_size * 2
        self.content_proj = nn.Linear(hidden2, content_dim)
        self.style_proj   = nn.Linear(hidden2, style_dim)
        self.gate         = nn.Parameter(torch.tensor([0.5]))

        self.classifier = nn.Sequential(
            nn.LayerNorm(content_dim + style_dim + 1),
            nn.Linear(content_dim + style_dim + 1, 2),
        )


    def _tokenize(self, s1: List[str], s2: List[str], max_len: int = 128):
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
        for i, mask in enumerate(enc["special_tokens_mask"]):
            sep = (enc["input_ids"][i] == self.tokenizer.sep_token_id).nonzero(as_tuple=False)
            if sep.numel():
                tt[i, sep[0, 0] + 1 :] = 1
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
            return (h * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp(min=1)

        mean0 = mmean(H, (tt == 0) & (m == 1))
        mean1 = mmean(H, (tt == 1) & (m == 1))
        rep   = torch.cat([cls, mean0 - mean1], 1)

        C = self.content_proj(rep)
        S = self.style_proj(rep)
        g = torch.sigmoid(self.gate)
        fused = torch.cat([C * (1 - g), S * g], 1)

        dists = []
        for b in range(H.size(0)):
            a_idx = ((tt[b] == 0) & (m[b] == 1)).nonzero(as_tuple=True)[0]
            b_idx = ((tt[b] == 1) & (m[b] == 1)).nonzero(as_tuple=True)[0]
            dists.append(sliced_w1(H[b, a_idx], H[b, b_idx], self.n_proj))
        dists = torch.stack(dists).unsqueeze(1)

        logits = self.classifier(torch.cat([fused, dists], 1))
        return logits


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        "Evaluator of the sliced OT variant.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model-path", required=True, help="Model (.pt)")
    p.add_argument("--data-root", required=True, help="pan25_data dir")
    p.add_argument("--output-dir", required=True, help="predictions dir")
    p.add_argument("--device", choices=["cuda", "cpu"], default="cuda", help="cuda or cpu")
    p.add_argument("--n-proj", type=int, default=64,
                   help="Number of random directions (must match training!!!)")
    return p.parse_args()


def load_model(path: str, device: torch.device, n_proj: int) -> OTStyleModel:
    state = torch.load(path, map_location=device)
    c_dim = state["content_proj.weight"].shape[0]
    s_dim = state["style_proj.weight"].shape[0]
    model = OTStyleModel(content_dim=c_dim, style_dim=s_dim, n_proj=n_proj, device=device)
    model.load_state_dict(state, strict=False)
    model.to(device).eval()
    return model


@torch.no_grad()
def predict_diff(model: OTStyleModel, diff: str, data_root: str, out_root: str):
    inp_dir = os.path.join(data_root, diff, "validation")
    out_dir = os.path.join(out_root, diff)
    os.makedirs(out_dir, exist_ok=True)

    for fname in tqdm(sorted(os.listdir(inp_dir)), desc=diff, unit="file"):
        if not (fname.startswith("problem-") and fname.endswith(".txt")):
            continue
        pid = fname[8:-4]

        with open(os.path.join(inp_dir, fname), encoding="utf-8") as f:
            sents = [ln.strip() for ln in f if ln.strip()]

        changes: List[int] = []
        for s1, s2 in zip(sents, sents[1:]):
            logits = model([s1], [s2])
            changes.append(int(logits.argmax(1).item()))

        with open(os.path.join(out_dir, f"solution-problem-{pid}.json"), "w") as fp:
            json.dump({"changes": changes}, fp, indent=2)


def main() -> None:
    args   = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print("Loading model ...")
    model = load_model(args.model_path, device, args.n_proj)

    for diff in ("easy", "medium", "hard"):
        predict_diff(model, diff, args.data_root, args.output_dir)


if __name__ == "__main__":
    main()
