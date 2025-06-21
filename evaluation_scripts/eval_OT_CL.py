#!/usr/bin/env python3
"""
Evaluator for the CL + OT model.

Folder layout I used for evaluation:

    <DATA_ROOT>/
        easy/   validation/problem-*.txt
        medium/ ditto
        hard/   ditto
"""

from __future__ import annotations
import argparse, json, random, pickle, warnings
from pathlib import Path
from functools import lru_cache
from typing import Dict, List, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from geomloss import SamplesLoss
from tqdm import tqdm

DIFFS = ("easy", "medium", "hard")


def set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


FUNCTION_WORDS = {"the","is","and","but","or","because","as","that"}

try:
    import spacy, textstat
    _NLP = spacy.load("en_core_web_sm", disable=["ner","textcat"])
except Exception as e:
    _NLP = None
    warnings.warn(f"No spaCy found, so features set to zeros ({e})")

@lru_cache(maxsize=50_000)
def extract(txt: str) -> List[float]:
    if _NLP is None:
        return [0.0]*7
    doc = _NLP(txt)
    toks = [t for t in doc if t.is_alpha]
    awl  = np.mean([len(t) for t in toks]) if toks else 0.0
    func = sum(t.text.lower() in FUNCTION_WORDS for t in toks)
    pos  = doc.count_by(spacy.attrs.POS)
    nn   = pos.get(_NLP.vocab.strings["NOUN"], 0)
    nv   = pos.get(_NLP.vocab.strings["VERB"], 0)
    asl  = np.mean([len(s.text.split()) for s in doc.sents]) if doc.sents else 0.0
    punct= sum(c in ".,!?;:-" for c in txt)
    read = textstat.flesch_reading_ease(txt) if textstat else 0.0
    return [awl, func, nn, nv, asl, punct, read]


class ContrastiveOTModel(nn.Module):
    def __init__(self,
                 content_dim: int,
                 style_dim: int,
                 feature_dim: int,
                 ot_blur: float,
                 encoder_name: str = "microsoft/deberta-v3-base",
                 device: str | torch.device = "cpu"):
        super().__init__()
        self.device = torch.device(device)
        self.feature_dim = feature_dim

        self.tokenizer = AutoTokenizer.from_pretrained(
            encoder_name, add_prefix_space=True)
        self.encoder = AutoModel.from_pretrained(encoder_name)

        hidden = self.encoder.config.hidden_size
        self.content_proj = nn.Linear(hidden, content_dim)
        self.style_proj   = nn.Linear(hidden, style_dim)
        self.feature_proj = nn.Linear(feature_dim*2, style_dim)

        self.cl_proj = nn.Sequential(
            nn.Linear(style_dim, style_dim, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(style_dim, style_dim, bias=False)
        )

        self.sinkhorn = SamplesLoss("sinkhorn", p=1,
                                    blur=ot_blur, debias=True).to(self.device)
        self.ot_proj  = nn.Linear(1, style_dim)
        self.ot_norm  = nn.LayerNorm(style_dim)

        self.classifier = nn.Linear(style_dim*3, 1)


    @staticmethod
    def _clouds(H, tt, am):
        c0,c1,w0,w1 = [],[],[],[]
        for h,t,m in zip(H,tt,am):
            a = (t==0)&(m==1); b = (t==1)&(m==1)
            A = h[a] if a.any() else h[:1]*0
            B = h[b] if b.any() else h[:1]*0
            c0.append(A); c1.append(B)
            w0.append(torch.full((A.size(0),),1/A.size(0),device=h.device))
            w1.append(torch.full((B.size(0),),1/B.size(0),device=h.device))
        def pad(lst,left=False):
            n = max(x.size(0) for x in lst)
            if left:   # 1-D
                return torch.stack([F.pad(x,(0,n-x.size(0))) for x in lst])
            return torch.stack([F.pad(x,(0,0,0,n-x.size(0))) for x in lst])
        return pad(c0),pad(w0,True),pad(c1),pad(w1,True)

    def _tok(self, a: List[str], b: List[str], max_len=128):
        enc = self.tokenizer(
            a, b, padding=True, truncation=True, max_length=max_len,
            return_tensors="pt", return_token_type_ids=True)
        return {k:v.to(self.device) for k,v in enc.items()}


    @torch.no_grad()
    def predict(self, a: str, b: str,
                f1: torch.Tensor, f2: torch.Tensor) -> float:
        inp = self._tok([a], [b])
        H   = self.encoder(**inp).last_hidden_state
        H   = H * inp["attention_mask"].unsqueeze(-1)
        cls = H[:,0]

        content = self.content_proj(cls)
        style   = self.style_proj(cls)

        feats   = torch.cat([f1, f2], dim=-1)
        feat_emb= self.feature_proj(feats)

        t0,w0,t1,w1 = self._clouds(H, inp["token_type_ids"], inp["attention_mask"])
        ot_val = torch.log1p(self.sinkhorn(w0,t0,w1,t1))
        ot_emb = self.ot_norm(self.ot_proj(ot_val.unsqueeze(-1)))

        gate = torch.sigmoid(torch.norm(feats, dim=-1, keepdim=True))
        style_vec = style * gate
        content_vec = content * (1 - gate)

        rep = torch.cat([style_vec, feat_emb, ot_emb], dim=-1)
        logit = self.classifier(rep).squeeze(1).item()
        return logit



def load_model(ckpt: Path, blur: float, device: torch.device) -> ContrastiveOTModel:
    raw = torch.load(ckpt, map_location=device)
    sd  = raw["model_state_dict"] if "model_state_dict" in raw else raw

    style_dim   = sd["style_proj.weight"].shape[0]
    content_dim = sd["content_proj.weight"].shape[0]
    feature_dim = sd["feature_proj.weight"].shape[1] // 2

    model = ContrastiveOTModel(content_dim=content_dim,
                               style_dim=style_dim,
                               feature_dim=feature_dim,
                               ot_blur=blur,
                               device=device)
    model.load_state_dict(sd, strict=False)
    model.to(device).eval()
    return model


def cli() -> argparse.Namespace:
    p = argparse.ArgumentParser("Evaluate the CL + OT model.")
    p.add_argument("--model-path", required=True)
    p.add_argument("--data-root", required=True,
                   help="Root with pan25_data.")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--blur", type=float, default=0.05,
                   help="Blur for Sinkhorn.")
    p.add_argument("--feat-cache", type=str, default=None,
                   help="Pickle file to cache sentence -> 7-dim np.ndarray")
    p.add_argument("--device", choices=["cuda","cpu"], default="cuda")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


@torch.no_grad()
def predict_diff(model: ContrastiveOTModel, diff: str,
                 root: Path, out_root: Path,
                 cache: Dict[str, np.ndarray]) -> None:
    inp_dir = root / diff / "validation"
    out_dir = out_root / diff
    out_dir.mkdir(parents=True, exist_ok=True)

    for txt in tqdm(sorted(inp_dir.glob("problem-*.txt")),
                    desc=diff, unit="file"):
        pid = txt.stem[8:]
        sents = [l.strip() for l in txt.open(encoding="utf-8") if l.strip()]

        changes: List[int] = []
        for a, b in zip(sents, sents[1:]):
            v1 = cache.get(a)
            if v1 is None:
                v1 = np.asarray(extract(a), dtype=np.float32); cache[a] = v1
            v2 = cache.get(b)
            if v2 is None:
                v2 = np.asarray(extract(b), dtype=np.float32); cache[b] = v2
            f1 = torch.from_numpy(v1).unsqueeze(0).to(model.device)
            f2 = torch.from_numpy(v2).unsqueeze(0).to(model.device)

            logit = model.predict(a, b, f1, f2)
            changes.append(int(torch.sigmoid(torch.tensor(logit)) > 0.5))

        (out_dir / f"solution-problem-{pid}.json").write_text(
            json.dumps({"changes": changes}, indent=2), encoding="utf-8")


def main() -> None:
    args = cli()
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # cache for features
    cache: Dict[str, np.ndarray] = {}
    if args.feat_cache:
        p = Path(args.feat_cache)
        if p.exists():
            cache = pickle.load(p.open("rb"))
            print(f"Loaded cached features ({len(cache)} sentences)")

    model = load_model(Path(args.model_path), args.blur, device)
    root  = Path(args.data_root); out_root = Path(args.output_dir)

    for diff in DIFFS:
        predict_diff(model, diff, root, out_root, cache)

    if args.feat_cache:
        pickle.dump(cache, open(args.feat_cache, "wb"))
        print(f"saved cache to {args.feat_cache} ({len(cache)} sentences)")

    print(f"predictions in: {out_root}")

if __name__ == "__main__":
    main()
