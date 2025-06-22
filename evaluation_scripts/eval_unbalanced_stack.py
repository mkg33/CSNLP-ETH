#!/usr/bin/env python3
"""
Evaluator of unbalanced OT (StackExchange dataset).
"""

from __future__ import annotations
import argparse, json, random, pickle, inspect
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from geomloss import SamplesLoss
from tqdm import tqdm


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
    print(f"no spaCy/textstat, so features will be zeros ({e})")

def extract_features(txt: str) -> List[float]:
    if _NLP is None:
        return [0.0]*7
    doc = _NLP(txt)
    toks = [t for t in doc if t.is_alpha]
    avg_len = np.mean([len(t) for t in toks]) if toks else 0.0
    func_wc = sum(t.text.lower() in FUNCTION_WORDS for t in toks)
    pos = doc.count_by(spacy.attrs.POS)
    n_noun = pos.get(_NLP.vocab.strings["NOUN"], 0)
    n_verb = pos.get(_NLP.vocab.strings["VERB"], 0)
    sent_l = np.mean([len(s.text.split()) for s in doc.sents]) if doc.sents else 0.0
    punct  = sum(ch in ".,!?;:-" for ch in txt)
    read   = textstat.flesch_reading_ease(txt) if textstat else 0.0
    return [avg_len, func_wc, n_noun, n_verb, sent_l, punct, read]


class OTModel(nn.Module):
    def __init__(self,
                 content_dim:int, style_dim:int, feature_dim:int,
                 blur:float, scaling:float|None,
                 unbalanced:bool, tau:float,
                 encoder:str="microsoft/deberta-v3-base",
                 device:str|torch.device="cpu"):
        super().__init__()
        self.device = torch.device(device)
        self.feature_dim = feature_dim

        self.tokenizer = AutoTokenizer.from_pretrained(
            encoder, add_prefix_space=True)
        self.encoder   = AutoModel.from_pretrained(encoder)

        hid = self.encoder.config.hidden_size
        self.content_proj = nn.Linear(hid, content_dim)
        self.style_proj   = nn.Linear(hid, style_dim)
        self.feature_proj = nn.Linear(feature_dim*2, style_dim)

        kw: Dict[str,Any] = dict(loss="sinkhorn", p=1, blur=blur, debias=True)
        if scaling is not None:
            kw["scaling"] = scaling
        if unbalanced:
            kw["tau"] = tau

        sig = inspect.signature(SamplesLoss)
        if "tau" not in sig.parameters and "reach" in sig.parameters and "tau" in kw:
            kw["reach"] = kw.pop("tau")

        self.sinkhorn = SamplesLoss(**kw).to(self.device)
        self.ot_proj  = nn.Linear(1, style_dim)
        self.ot_norm  = nn.LayerNorm(style_dim)

        gate_in = style_dim*3 + content_dim
        self.gate_net = nn.Sequential(
            nn.Linear(gate_in, 64),
            nn.ReLU(),
            nn.Linear(64, style_dim)
        )
        self.classifier = nn.Linear(style_dim*3, 1)

    @staticmethod
    def _clouds(H, tt, am):
        c0,c1,w0,w1 = [],[],[],[]
        for h,t,m in zip(H,tt,am):
            seg0 = (t==0)&(m==1); seg1 = (t==1)&(m==1)
            a = h[seg0] if seg0.any() else h[:1]*0
            b = h[seg1] if seg1.any() else h[:1]*0
            c0.append(a); c1.append(b)
            w0.append(torch.full((a.size(0),),1/a.size(0),device=h.device))
            w1.append(torch.full((b.size(0),),1/b.size(0),device=h.device))
        def pad(lst,left=False):
            n=max(t.size(0) for t in lst)
            return torch.stack([F.pad(t,(0,0,0,n-t.size(0))) if not left
                                else F.pad(t,(0,n-t.size(0)))
                                for t in lst])
        return pad(c0),pad(w0,left=True),pad(c1),pad(w1,left=True)

    def _tok(self,s1:List[str],s2:List[str],max_len=128):
        enc = self.tokenizer(s1,s2,padding=True,truncation=True,
                             max_length=max_len,return_tensors="pt",
                             return_token_type_ids=True)
        return {k:v.to(self.device) for k,v in enc.items()}

    def forward(self,s1:List[str],s2:List[str],
                f1:torch.Tensor,f2:torch.Tensor)->torch.Tensor:
        enc = self._tok(s1,s2); H = self.encoder(**enc).last_hidden_state
        cls = H[:,0]; tt,am = enc["token_type_ids"], enc["attention_mask"]

        C = self.content_proj(cls)
        S = self.style_proj(cls)
        feat_emb = self.feature_proj(torch.cat([f1,f2],1))

        t0,w0,t1,w1 = self._clouds(H,tt,am)
        ot_val = torch.log1p(self.sinkhorn(w0,t0,w1,t1))
        ot_emb = self.ot_norm(self.ot_proj(ot_val.unsqueeze(-1)))

        gate = torch.sigmoid(self.gate_net(torch.cat([feat_emb,C,S,ot_emb],1)))
        mix  = (1-gate)*C + gate*S
        rep  = torch.cat([mix,feat_emb,ot_emb],1)

        return self.classifier(rep).squeeze(1)


def parse_args()->argparse.Namespace:
    p = argparse.ArgumentParser("unbalanced OT evaluator (StackExchange dataset)")
    p.add_argument("--model-path",required=True)
    p.add_argument("--input-dir",required=True,
                   help="Dir with dataset3/validation files")
    p.add_argument("--output-dir",required=True)
    p.add_argument("--blur",type=float,default=0.05)
    p.add_argument("--scaling",type=float,default=None)
    p.add_argument("--unbalanced",action="store_true",default=False)
    p.add_argument("--tau",type=float,default=0.8)
    p.add_argument("--feat-cache",type=str,default=None)
    p.add_argument("--batch-size", type=int, default=32, help="Batch size (better inference)")
    p.add_argument("--device",choices=["cuda","cpu"],default="cuda")
    p.add_argument("--seed",type=int,default=42)
    return p.parse_args()


def load_model(path:Path,args,device)->OTModel:
    raw=torch.load(path,map_location=device)
    sd = raw["model_state_dict"] if "model_state_dict" in raw else raw
    c_dim = sd["content_proj.weight"].shape[0]
    s_dim = sd["style_proj.weight"].shape[0]
    f_dim = sd["feature_proj.weight"].shape[1]//2
    model = OTModel(
        c_dim,s_dim,f_dim,
        blur=args.blur,scaling=args.scaling,
        unbalanced=args.unbalanced,tau=args.tau,
        device=device)
    model.load_state_dict(sd,strict=False)
    model.to(device).eval()
    return model



@torch.inference_mode()
def predict_batched(model: OTModel,
                    inp_dir: Path,
                    out_dir: Path,
                    cache: Dict[str, np.ndarray],
                    batch_size: int):

    out_dir.mkdir(parents=True, exist_ok=True)

    for txt_file in tqdm(sorted(inp_dir.glob("problem-*.txt")), unit="file"):
        pid = txt_file.stem[8:]
        with txt_file.open(encoding="utf-8") as fh:
            sents = [line.strip() for line in fh if line.strip()]

        pairs: List[Tuple[str, str]] = list(zip(sents, sents[1:]))
        predictions: List[int] = []

        if not pairs:
            (out_dir / f"solution-problem-{pid}.json").write_text(
                json.dumps({"changes": []}), encoding="utf-8")
            continue

        for i in range(0, len(pairs), batch_size):
            batch_pairs = pairs[i : i + batch_size]
            s1_batch, s2_batch = zip(*batch_pairs)

            f1_list, f2_list = [], []
            for s1, s2 in batch_pairs:
                v1 = cache.get(s1)
                if v1 is None: v1 = np.asarray(extract_features(s1), dtype=np.float32); cache[s1] = v1
                f1_list.append(v1)

                v2 = cache.get(s2)
                if v2 is None: v2 = np.asarray(extract_features(s2), dtype=np.float32); cache[s2] = v2
                f2_list.append(v2)

            f1_tensor = torch.from_numpy(np.stack(f1_list)).to(model.device)
            f2_tensor = torch.from_numpy(np.stack(f2_list)).to(model.device)

            logits = model(list(s1_batch), list(s2_batch), f1_tensor, f2_tensor)

            preds = (torch.sigmoid(logits) > 0.5).int().tolist()
            predictions.extend(preds)

        (out_dir / f"solution-problem-{pid}.json").write_text(
            json.dumps({"changes": predictions}, indent=2), encoding="utf-8")


def main() -> None:
    args=parse_args(); set_seed(args.seed)
    device=torch.device(args.device if torch.cuda.is_available() else "cpu")

    cache:Dict[str,np.ndarray]={}
    if args.feat_cache:
        p=Path(args.feat_cache)
        if p.exists():
            obj=pickle.load(open(p,"rb"))
            if isinstance(obj,dict):
                cache=obj
                print(f"loaded cache: {len(cache)} sentences")
            else:
                import pandas as pd
                df:"pd.DataFrame"=obj
                for _,row in df.iterrows():
                    cache[row["sentence1"]]=np.asarray(row["features1"],dtype=np.float32)
                    cache[row["sentence2"]]=np.asarray(row["features2"],dtype=np.float32)
                print(f"converted to cache: ({len(cache)})")

    model = load_model(Path(args.model_path),args,device)
    print(f"Predicting with batch size: {args.batch_size}...")
    predict_batched(model, Path(args.input_dir), Path(args.output_dir), cache, args.batch_size)

    if args.feat_cache:
        pickle.dump(cache,open(args.feat_cache,"wb"))
        print(f"cache saved: ({len(cache)} sentences)")

    print("Preds written to", args.output_dir)

if __name__ == "__main__":
    main()
