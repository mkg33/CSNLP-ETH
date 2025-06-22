#!/usr/bin/env python3
"""
Evaluator for the CL + OT model on the StackExchange dataset.
"""

from __future__ import annotations
import argparse, json, warnings
from pathlib import Path
from functools import lru_cache
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


def set_seed(seed: int = 42) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


try:
    from geomloss import SamplesLoss
except ImportError:
    SamplesLoss = None


class ContrastiveOTModel(nn.Module):

    def __init__(self,
                 model_name: str = "microsoft/deberta-v3-base",
                 hidden_dim: int = 768,
                 style_dim:  int = 128,
                 content_dim:int = 128,
                 feature_dim:int = 7,
                 ot_blur:    float = 0.05):
        super().__init__()

        self.feature_dim = feature_dim
        self.encoder     = AutoModel.from_pretrained(model_name)

        self.content_proj = nn.Linear(hidden_dim, content_dim)
        self.style_proj   = nn.Linear(hidden_dim, style_dim)
        self.feature_proj = nn.Linear(feature_dim * 2, style_dim)


        self.cl_proj = nn.Sequential(
            nn.Linear(style_dim, style_dim, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(style_dim, style_dim, bias=False)
        )

        # OT
        if SamplesLoss:
            self.sinkhorn = SamplesLoss("sinkhorn", p=1,
                                        blur=ot_blur, debias=True)
        else:
            self.sinkhorn = None
        self.ot_proj = nn.Linear(1, style_dim)
        self.ot_norm = nn.LayerNorm(style_dim)

        self.classifier = nn.Linear(style_dim * 3, 1)


    @staticmethod
    def _clouds(H, tt, am):
        c0, c1, w0, w1 = [], [], [], []
        for h, t, m in zip(H, tt, am):
            mask0 = (t == 0) & (m == 1)
            mask1 = (t == 1) & (m == 1)
            A = h[mask0] if mask0.any() else h[:1] * 0
            B = h[mask1] if mask1.any() else h[:1] * 0
            c0.append(A);  c1.append(B)
            w0.append(torch.full((A.size(0),), 1 / A.size(0), device=h.device))
            w1.append(torch.full((B.size(0),), 1 / B.size(0), device=h.device))

        def pad(lst, one_d=False):
            N = max(x.size(0) for x in lst)
            if one_d:
                return torch.stack([F.pad(x, (0, N - x.size(0))) for x in lst])
            return torch.stack([
                F.pad(x, (0, 0, 0, N - x.size(0))) for x in lst
            ])

        return pad(c0), pad(w0, True), pad(c1), pad(w1, True)


    def forward(self,
                ids:  torch.Tensor,
                mask: torch.Tensor,
                tt:   torch.Tensor,
                f1:   torch.Tensor,
                f2:   torch.Tensor) -> torch.Tensor:

        H = self.encoder(input_ids=ids,
                         attention_mask=mask,
                         return_dict=True).last_hidden_state
        H = H * mask.unsqueeze(-1)
        cls = H[:, 0]

        content_vec = self.content_proj(cls)
        style_vec   = self.style_proj(cls)

        feats       = torch.cat([f1, f2], dim=-1)
        feat_emb    = self.feature_proj(feats)

        # OT
        if self.sinkhorn is not None:
            t0, w0, t1, w1 = self._clouds(H, tt, mask)
            ot_scalar = self.sinkhorn(w0, t0, w1, t1)
            ot_emb    = self.ot_norm(self.ot_proj(torch.log1p(ot_scalar)
                                                  .unsqueeze(-1)))
        else:
            ot_emb = torch.zeros_like(feat_emb)


        gate = torch.sigmoid(torch.norm(feats, dim=-1, keepdim=True))
        style_vec   = style_vec   * gate
        content_vec = content_vec * (1 - gate)

        rep   = torch.cat([style_vec, feat_emb, ot_emb], dim=-1)
        logits = self.classifier(rep).squeeze(-1)
        return logits


FUNCTION_WORDS = {
    "the", "is", "and", "but", "or", "because", "as", "that"
}

try:
    import spacy, textstat
    _NLP = spacy.load("en_core_web_sm", disable=["ner", "textcat"])
except Exception as e:
    _NLP = None
    warnings.warn(f"no spaCy, so features set to zeros ({e})")


@lru_cache(maxsize=50_000)
def extract(text: str) -> List[float]:
    if _NLP is None:
        return [0.0] * 7

    doc  = _NLP(text)
    toks = [t for t in doc if t.is_alpha]

    awl   = np.mean([len(t) for t in toks]) if toks else 0.0
    func  = sum(t.text.lower() in FUNCTION_WORDS for t in toks)

    pos   = doc.count_by(spacy.attrs.POS)
    nn    = pos.get(_NLP.vocab.strings["NOUN"], 0)
    nv    = pos.get(_NLP.vocab.strings["VERB"], 0)

    asl   = np.mean([len(s.text.split()) for s in doc.sents]) if doc.sents else 0.0
    punct = sum(c in ".,!?;:-" for c in text)
    read  = textstat.flesch_reading_ease(text) if textstat else 0.0

    return [awl, func, nn, nv, asl, punct, read]


def load_model(path: Path,
               device: torch.device,
               ot_blur: float = 0.05) -> ContrastiveOTModel:
    ckpt = torch.load(path, map_location=device)
    sd   = ckpt.get("model_state_dict", ckpt)

    style_dim   = sd["style_proj.weight"].shape[0]
    content_dim = sd["content_proj.weight"].shape[0]
    feature_dim = sd["feature_proj.weight"].shape[1] // 2

    model = ContrastiveOTModel(style_dim   = style_dim,
                               content_dim = content_dim,
                               feature_dim = feature_dim,
                               ot_blur     = ot_blur).to(device)
    model.load_state_dict(sd, strict=True)
    model.eval()
    return model



@torch.no_grad()
def predict_dir(model:   ContrastiveOTModel,
                tok:     AutoTokenizer,
                inp_dir: Path,
                out_dir: Path,
                threshold:  float,
                device:  torch.device) -> None:

    out_dir.mkdir(parents=True, exist_ok=True)

    for txt in tqdm(sorted(inp_dir.glob("problem-*.txt")),
                    unit="file", leave=False):
        pid    = txt.stem[8:]
        sents  = [l.strip() for l in txt.open(encoding="utf-8") if l.strip()]

        changes: List[int] = []
        for a, b in zip(sents, sents[1:]):
            enc = tok(a, b,
                      padding="max_length",
                      truncation=True,
                      max_length=128,
                      return_token_type_ids=True,
                      return_tensors="pt").to(device)

            f1 = torch.tensor(extract(a), dtype=torch.float32,
                              device=device).unsqueeze(0)
            f2 = torch.tensor(extract(b), dtype=torch.float32,
                              device=device).unsqueeze(0)

            logit = model(enc["input_ids"],
                          enc["attention_mask"],
                          enc["token_type_ids"],
                          f1, f2)

            pred = int(torch.sigmoid(logit).item() > threshold)
            changes.append(pred)

        (out_dir / f"solution-problem-{pid}.json").write_text(
            json.dumps({"changes": changes}, indent=2), encoding="utf-8")


def cli() -> argparse.Namespace:
    p = argparse.ArgumentParser("Evaluate the CL+OT model (StackExchange dataset).")
    p.add_argument("--model-path",  required=True)
    p.add_argument("--input-dir",   required=True)
    p.add_argument("--output-dir",  required=True)
    p.add_argument("--threshold",   type=float, default=0.5,
                   help="threshold after sigmoid")
    p.add_argument("--device",      choices=["cuda", "cpu"], default="cuda")
    p.add_argument("--seed",        type=int, default=42)
    return p.parse_args()


def main() -> None:
    args   = cli()
    set_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model  = load_model(Path(args.model_path), device)

    tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")

    predict_dir(model,
                tokenizer,
                Path(args.input_dir),
                Path(args.output_dir),
                args.threshold,
                device)

    print("Preds written to", args.output_dir)


if __name__ == "__main__":
    main()
