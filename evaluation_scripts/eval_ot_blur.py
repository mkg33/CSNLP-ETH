#!/usr/bin/env python3
"""
Evaluator for the balanced OT model with no features.
Blur hyperparameter sweep.
"""

from __future__ import annotations
import argparse, re, json, random, sys
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from geomloss import SamplesLoss
from tqdm import tqdm


def set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class OTStyleModel(nn.Module):
    def __init__(self,
                 blur: float,
                 content_dim: int,
                 style_dim: int,
                 encoder_name: str = "microsoft/deberta-v3-base",
                 device: str | torch.device = "cpu"):
        super().__init__()
        self.device = torch.device(device)
        self.blur   = blur

        self.tokenizer = AutoTokenizer.from_pretrained(
            encoder_name, add_prefix_space=True)
        self.encoder   = AutoModel.from_pretrained(encoder_name)

        hidden2 = self.encoder.config.hidden_size * 2
        self.content_proj = nn.Linear(hidden2, content_dim)
        self.style_proj   = nn.Linear(hidden2, style_dim)
        self.gate         = nn.Parameter(torch.tensor([0.5]))

        self.classifier = nn.Sequential(
            nn.LayerNorm(content_dim + style_dim + 1),
            nn.Linear(content_dim + style_dim + 1, 2),
        )

        # OT
        self.ot = SamplesLoss("sinkhorn", p=1, blur=blur,
                              debias=True, potentials=False).to(self.device)


    def _tokenize(self, s1: List[str], s2: List[str],
                  max_len: int = 128) -> Dict[str, torch.Tensor]:
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
        tt, am = inp["token_type_ids"], inp["attention_mask"]
        cls = H[:, 0]

        def mmean(h, mask):
            return (h * mask.unsqueeze(-1)).sum(1) / mask.sum(
                1, keepdim=True).clamp(min=1)

        m0 = mmean(H, (tt == 0) & (am == 1))
        m1 = mmean(H, (tt == 1) & (am == 1))
        rep = torch.cat([cls, m0 - m1], 1)

        C = self.content_proj(rep)
        S = self.style_proj(rep)
        g = torch.sigmoid(self.gate)
        fused = torch.cat([C * (1 - g), S * g], 1)

        dists = []
        for b in range(H.size(0)):
            a_idx = ((tt[b] == 0) & (am[b] == 1)).nonzero(as_tuple=True)[0]
            b_idx = ((tt[b] == 1) & (am[b] == 1)).nonzero(as_tuple=True)[0]
            dists.append(self.ot(H[b, a_idx], H[b, b_idx]))
        d_ot = torch.stack(dists).unsqueeze(1)

        logits = self.classifier(torch.cat([fused, d_ot], 1))
        return logits


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        "Evaluator of the balanced OT model (with no features).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model-path", required=True, help="Model (.pt)")
    p.add_argument("--blur", type=float, default=None,
                   help="Blur value; must match training!!!")
    p.add_argument("--data-root", required=True, help="pan25_data dir")
    p.add_argument("--output-dir", required=True, help="predictions dir")
    p.add_argument("--device", choices=["cuda", "cpu"], default="cuda",
                   help="cpu or cuda")
    p.add_argument("--seed", type=int, default=42,
                   help="Always 42.")
    return p.parse_args()


def load_model(ckpt_path: Path, blur: float, device: torch.device) -> OTStyleModel:
    state = torch.load(ckpt_path, map_location=device)


    if "model_state" in state:
        state_dict = state["model_state"]
    else:
        state_dict = state

    c_dim = state_dict["content_proj.weight"].shape[0]
    s_dim = state_dict["style_proj.weight"].shape[0]

    model = OTStyleModel(blur=blur,
                         content_dim=c_dim,
                         style_dim=s_dim,
                         device=device)
    model.load_state_dict(state_dict, strict=False)
    model.to(device).eval()
    return model


@torch.no_grad()
def predict_diff(model: OTStyleModel, diff: str,
                 data_root: Path, out_root: Path) -> None:
    inp_dir = data_root / diff / "validation"
    out_dir = out_root / diff
    out_dir.mkdir(parents=True, exist_ok=True)

    for txt in tqdm(sorted(inp_dir.glob("problem-*.txt")),
                    desc=diff, unit="file"):
        pid = txt.stem[8:]
        with txt.open(encoding="utf-8") as fh:
            sents = [ln.strip() for ln in fh if ln.strip()]

        changes = []
        for s1, s2 in zip(sents, sents[1:]):
            logits = model([s1], [s2])
            changes.append(int(logits.argmax(1).item()))

        (out_dir / f"solution-problem-{pid}.json").write_text(
            json.dumps({"changes": changes}, indent=2),
            encoding="utf-8")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    blur = args.blur
    if blur is None:
        m = re.search(r"blur([0-9.]+)", Path(args.model_path).stem)
        if not m:
            sys.exit("Blur not in filename and flag not specified. Aborting...")
        blur = float(m.group(1))

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print(f"Load model (blur={blur}) ...")
    model = load_model(Path(args.model_path), blur, device)

    data_root = Path(args.data_root)
    out_root  = Path(args.output_dir)

    for diff in ("easy", "medium", "hard"):
        predict_diff(model, diff, data_root, out_root)

    print("preds written to", out_root)


if __name__ == "__main__":
    main()
