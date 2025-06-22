#!/usr/bin/env python3
"""
Evaluator for the multi-scale OTStyleModel.
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


def sliced_w1(x: torch.Tensor, y: torch.Tensor, n_proj: int) -> torch.Tensor:
    """sliced Wasserstein-1 distance."""
    d = x.size(1)
    v = torch.randn(n_proj, d, device=x.device)
    v = v / v.norm(dim=1, keepdim=True)
    xp = (x @ v.T).sort(dim=0)[0]
    yp = (y @ v.T).sort(dim=0)[0]
    m  = min(xp.size(0), yp.size(0))
    return (xp[:m] - yp[:m]).abs().mean()


class OTStyleModel(nn.Module):

    PROJ_LIST = (8, 32, 128)

    def __init__(self, content_dim: int, style_dim: int,
                 encoder: str="microsoft/deberta-v3-base",
                 device: str|torch.device="cpu"):
        super().__init__()
        self.device = torch.device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(
            encoder, add_prefix_space=True)
        self.encoder   = AutoModel.from_pretrained(encoder)
        hidden2 = self.encoder.config.hidden_size * 2
        self.content_proj = nn.Linear(hidden2, content_dim)
        self.style_proj   = nn.Linear(hidden2, style_dim)
        self.gate         = nn.Parameter(torch.tensor([0.5]))
        self.classifier   = nn.Sequential(
            nn.LayerNorm(content_dim + style_dim + 3),
            nn.Linear(content_dim + style_dim + 3, 2)
        )

    def _tokenize(self, s1: Sequence[str], s2: Sequence[str], max_len: int = 128) -> dict[str, torch.Tensor]:

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
        H = self.encoder(**inp).last_hidden_state
        cls = H[:, 0]
        token_type_ids, attention_mask = inp["token_type_ids"], inp["attention_mask"]

        def mmean(h, mask):
            return (h * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp(min=1)

        mean0 = mmean(H, (token_type_ids == 0) & (attention_mask == 1))
        mean1 = mmean(H, (token_type_ids == 1) & (attention_mask == 1))
        rep = torch.cat([cls, mean0 - mean1], dim=1)

        C = self.content_proj(rep)
        S = self.style_proj(rep)
        g = torch.sigmoid(self.gate)
        fused = torch.cat([C * (1 - g), S * g], dim=1)

        multi_scale_dists = []
        for n_proj in self.PROJ_LIST:
            batch_dists = []
            for b in range(H.size(0)):
                a_mask = (token_type_ids[b] == 0) & (attention_mask[b] == 1)
                b_mask = (token_type_ids[b] == 1) & (attention_mask[b] == 1)

                if not a_mask.any() or not b_mask.any():
                    batch_dists.append(torch.tensor(0.0, device=self.device))
                    continue
                batch_dists.append(sliced_w1(H[b, a_mask], H[b, b_mask], n_proj))
            multi_scale_dists.append(torch.stack(batch_dists))

        d_ot = torch.stack(multi_scale_dists, dim=1)
        logits = self.classifier(torch.cat([fused, d_ot], dim=1))
        return logits


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        "evaluator for multi-scale OT model (StackExchange dataset)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    p.add_argument("--model-path", required=True, help="Model (.pt)")
    p.add_argument("--input-dir", required=True, help="Directory with problem-*.txt files")
    p.add_argument("--output-dir", required=True, help="Directory for preds")
    p.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    p.add_argument("--batch-size", type=int, default=32, help="Better inference")
    p.add_argument("--seed", type=int, default=42, help="Always 42.")
    return p.parse_args()


def load_model(ckpt: Path, device: torch.device) -> OTStyleModel:
    sd = torch.load(ckpt, map_location=device)
    c_dim = sd["content_proj.weight"].shape[0]
    s_dim = sd["style_proj.weight"].shape[0]
    model = OTStyleModel(c_dim, s_dim, device=device)
    model.load_state_dict(sd, strict=False)
    model.to(device).eval()
    return model


@torch.inference_mode()
def predict_batched(model: OTStyleModel, inp_dir: Path, out_dir: Path, batch_size: int) -> None:

    out_dir.mkdir(parents=True, exist_ok=True)
    for txt_file in tqdm(sorted(inp_dir.glob("problem-*.txt")), unit="file"):
        pid = txt_file.stem
        with txt_file.open(encoding="utf-8") as fh:
            sents = [line.strip() for line in fh if line.strip()]

        pairs: List[Tuple[str, str]] = list(zip(sents, sents[1:]))
        predictions: List[int] = []

        for i in range(0, len(pairs), batch_size):
            batch = pairs[i:i + batch_size]
            if not batch:
                continue
            s1_batch, s2_batch = zip(*batch)
            logits = model(list(s1_batch), list(s2_batch))
            predictions.extend(logits.argmax(dim=1).tolist())

        (out_dir / f"solution-{pid}.json").write_text(
            json.dumps({"changes": predictions}, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print("Loading model..")
    model = load_model(Path(args.model_path), device)

    print(f"Started predicting (batch size: {args.batch_size})...")
    predict_batched(model, Path(args.input_dir), Path(args.output_dir), args.batch_size)

    print(f"predictions saved to {args.output_dir}")


if __name__ == "__main__":
    main()
