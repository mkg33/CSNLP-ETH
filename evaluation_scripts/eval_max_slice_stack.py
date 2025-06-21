#!/usr/bin/env python3
"""
Evaluator for max projection (StackExchange dataset).
"""

from __future__ import annotations
import argparse, json, random
from pathlib import Path
from typing import List, Sequence, Tuple

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


def max_sliced_w1_proj(x: torch.Tensor, y: torch.Tensor,
                       proj: torch.Tensor) -> torch.Tensor:
    """Max-sliced Wasserstein-1 distance."""
    xp, _ = (x @ proj.T).sort(dim=0)
    yp, _ = (y @ proj.T).sort(dim=0)
    m = min(xp.size(0), yp.size(0))
    return (xp[:m] - yp[:m]).abs().mean(0).max()


class OTStyleModel(nn.Module):

    def __init__(self,
                 encoder: str              = "microsoft/deberta-v3-base",
                 content_dim: int          = 192,
                 style_dim: int            = 128,
                 n_proj: int               = 128,
                 resample_proj: bool       = True,
                 proj_seed: int            = 42,
                 device: str | torch.device = "cpu"):
        super().__init__()
        self.device = torch.device(device)
        self.n_proj = n_proj
        self.resample_proj = resample_proj

        self.tokenizer = AutoTokenizer.from_pretrained(
            encoder, add_prefix_space=True)
        self.encoder = AutoModel.from_pretrained(encoder)
        d_model = self.encoder.config.hidden_size
        hidden2 = 2 * d_model

        self.content_proj = nn.Linear(hidden2, content_dim)
        self.style_proj   = nn.Linear(hidden2, style_dim)
        self.gate         = nn.Parameter(torch.tensor([0.5]))

        self.classifier = nn.Sequential(
            nn.LayerNorm(content_dim + style_dim + 1),
            nn.Linear(content_dim + style_dim + 1, 2),
        )


        if not resample_proj:
            g = torch.Generator(device).manual_seed(proj_seed)
            dirs = torch.randn(n_proj, d_model, generator=g, device=self.device)
            dirs = dirs / dirs.norm(dim=1, keepdim=True)
            self.register_buffer("proj_dirs", dirs, persistent=False)
        else:
            self.register_buffer("proj_dirs", torch.empty(0), persistent=False)

    def _sample_dirs(self) -> torch.Tensor:
        """Draw random unit dirs."""
        d_model = self.encoder.config.hidden_size
        dirs = torch.randn(self.n_proj, d_model, device=self.device)
        return dirs / dirs.norm(dim=1, keepdim=True)

    def _tokenize(self, s1: Sequence[str], s2: Sequence[str],
                  max_len: int = 128) -> dict[str, torch.Tensor]:

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
                tt[i, sep[0, 0] + 1:] = 1
        enc["token_type_ids"] = tt
        enc.pop("offset_mapping")
        enc.pop("special_tokens_mask")
        return {k: v.to(self.device) for k, v in enc.items()}

    def forward(self, s1: Sequence[str], s2: Sequence[str]) -> torch.Tensor:
        inp = self._tokenize(s1, s2)
        H   = self.encoder(**inp).last_hidden_state
        tt, m = inp["token_type_ids"], inp["attention_mask"]
        cls = H[:, 0]

        def masked_mean(h, mask):
            return (h * mask.unsqueeze(-1)).sum(1) / mask.sum(
                1, keepdim=True).clamp(min=1)

        mean0 = masked_mean(H, (tt == 0) & (m == 1))
        mean1 = masked_mean(H, (tt == 1) & (m == 1))
        rep   = torch.cat([cls, mean0 - mean1], dim=1)

        C = self.content_proj(rep)
        S = self.style_proj(rep)
        g = torch.sigmoid(self.gate)
        fused = torch.cat([C * (1 - g), S * g], dim=1)


        dirs = self._sample_dirs() if self.resample_proj else self.proj_dirs

        dists = []
        for b in range(H.size(0)):
            a_idx = ((tt[b] == 0) & (m[b] == 1)).nonzero(as_tuple=True)[0]
            b_idx = ((tt[b] == 1) & (m[b] == 1)).nonzero(as_tuple=True)[0]

            if a_idx.numel() == 0 or b_idx.numel() == 0:
                dists.append(torch.tensor(0.0, device=self.device))
                continue
            dists.append(max_sliced_w1_proj(H[b, a_idx], H[b, b_idx], dirs))
        d_ot = torch.stack(dists).unsqueeze(1)

        logits = self.classifier(torch.cat([fused, d_ot], dim=1))
        return logits


def load_model(ckpt: Path, device: torch.device,
               n_proj: int, resample_proj: bool, seed: int) -> OTStyleModel:
    sd = torch.load(ckpt, map_location=device)
    if "model_state_dict" in sd:
        sd = sd["model_state_dict"]

    c_dim = sd["content_proj.weight"].shape[0]
    s_dim = sd["style_proj.weight"].shape[0]

    model = OTStyleModel(content_dim=c_dim,
                         style_dim=s_dim,
                         n_proj=n_proj,
                         resample_proj=resample_proj,
                         proj_seed=seed,
                         device=device)

    model.load_state_dict(sd, strict=False)
    model.to(device).eval()
    return model


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        "max projection OT model evaluator (StackExchange)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model-path", required=True, help="Model (.pt)")
    p.add_argument("--input-dir", required=True,
                   help="path to problem-*.txt files")
    p.add_argument("--output-dir", required=True,
                   help="Directory for predictions")
    p.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    p.add_argument("--n-proj", type=int, default=128,
                   help="Number of random projections (must match training!!!)")
    p.add_argument("--batch-size", type=int, default=32, help="Batch size for inference")
    p.add_argument("--no-resample-proj", action="store_true",
                   help="fixed projections (actually incorrect, but deterministic)")
    p.add_argument("--seed", type=int, default=42,
                   help="Seed (projection sampling)")
    return p.parse_args()


@torch.inference_mode()
def predict_batched(model: OTStyleModel, inp_dir: Path, out_dir: Path, batch_size: int) -> None:

    out_dir.mkdir(parents=True, exist_ok=True)

    for txt_file in tqdm(sorted(inp_dir.glob("problem-*.txt")), unit="file"):
        pid = txt_file.stem[8:]
        with txt_file.open(encoding="utf-8") as fh:
            sents = [ln.strip() for ln in fh if ln.strip()]

        pairs: List[Tuple[str, str]] = list(zip(sents, sents[1:]))
        predictions: List[int] = []

        for i in range(0, len(pairs), batch_size):
            batch = pairs[i:i + batch_size]
            if not batch:
                continue
            s1_batch, s2_batch = zip(*batch)
            logits = model(list(s1_batch), list(s2_batch))
            predictions.extend(logits.argmax(dim=1).tolist())

        (out_dir / f"solution-problem-{pid}.json").write_text(
            json.dumps({"changes": predictions}, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print("Loading model..")
    model = load_model(Path(args.model_path), device,
                       n_proj=args.n_proj,
                       resample_proj=not args.no_resample_proj,
                       seed=args.seed)

    print(f"Starting prediction (batch size: {args.batch_size})...")
    predict_batched(model, Path(args.input_dir), Path(args.output_dir), args.batch_size)

    print(f"predictions written to: {args.output_dir}")


if __name__ == "__main__":
    main()
