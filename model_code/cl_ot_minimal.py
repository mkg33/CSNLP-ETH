# CL + OT
import torch, torch.nn as nn, torch.nn.functional as F
from transformers import AutoModel
from geomloss import SamplesLoss


def info_nce(z: torch.Tensor,
             labels: torch.Tensor,
             tau: float = 0.07) -> torch.Tensor:
    """
    Supervised contrastive loss.
    """
    b = z.size(0)
    sim = torch.matmul(z, z.T) / tau
    mask_self = torch.eye(b, dtype=torch.bool, device=z.device)
    sim = sim.masked_fill(mask_self, -1e9)

    pos_mask = labels.unsqueeze(0).eq(labels.unsqueeze(1)) & ~mask_self
    log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)
    loss = -(log_prob[pos_mask]).mean()
    return loss


class ContrastiveOTModel(nn.Module):

    def __init__(self,
                 model_name="microsoft/deberta-v3-base",
                 hidden_dim=768,
                 style_dim=128,
                 content_dim=128,
                 feature_dim=7,
                 ot_blur=0.05):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)


        self.content_proj = nn.Linear(hidden_dim, content_dim)
        self.style_proj   = nn.Linear(hidden_dim, style_dim)
        self.feature_proj = nn.Linear(feature_dim * 2, style_dim)


        self.cl_proj = nn.Sequential(
            nn.Linear(style_dim, style_dim, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(style_dim, style_dim, bias=False)
        )

        # OT branch
        self.sinkhorn = SamplesLoss("sinkhorn", p=1, blur=ot_blur, debias=True)
        self.ot_proj  = nn.Linear(1, style_dim)
        self.ot_norm  = nn.LayerNorm(style_dim)

        # final
        self.classifier = nn.Linear(style_dim * 3, 1)


    @staticmethod
    def _build_clouds(seq_emb, token_type_ids, attention_mask):
        """
        Build two token-clouds (segment-0 / segment-1) per sample and
        return tensors (with padding) plus uniform weights.

        Returns:

        t0, w0, t1, w1 :  (B, N_max, D) / (B, N_max)
        """
        B, L, D = seq_emb.size()
        clouds0, clouds1, w0s, w1s = [], [], [], []

        for i in range(B):
            real = attention_mask[i].bool()
            seg0 = real & (token_type_ids[i] == 0)
            seg1 = real & (token_type_ids[i] == 1)

            c0 = seq_emb[i][seg0]
            c1 = seq_emb[i][seg1]


            if c0.size(0) == 0:
                c0 = seq_emb[i, :1].detach() * 0.0
            if c1.size(0) == 0:
                c1 = seq_emb[i, :1].detach() * 0.0

            n0, n1 = c0.size(0), c1.size(0)
            w0 = torch.full((n0,), 1.0 / n0, device=seq_emb.device)
            w1 = torch.full((n1,), 1.0 / n1, device=seq_emb.device)

            clouds0.append(c0); w0s.append(w0)
            clouds1.append(c1); w1s.append(w1)


        def _pad(tensors, pad_weights=False):
            N = max(t.size(0) for t in tensors)
            if pad_weights:                     # 1-D
                return torch.stack([F.pad(t, (0, N - t.size(0))) for t in tensors])
            else:                               # 2-D
                return torch.stack([
                    F.pad(t, (0, 0, 0, N - t.size(0))) for t in tensors
                ])

        return (_pad(clouds0),
                _pad(w0s,  pad_weights=True),
                _pad(clouds1),
                _pad(w1s,  pad_weights=True))


    def forward(self,
                input_ids,
                attention_mask,
                token_type_ids,
                features1,
                features2):
        hidden = self.encoder(input_ids,
                              attention_mask,
                              return_dict=True).last_hidden_state
        hidden = hidden * attention_mask.unsqueeze(-1)
        cls = hidden[:, 0]

        content_vec = self.content_proj(cls)
        style_vec   = self.style_proj(cls)


        z = F.normalize(self.cl_proj(style_vec), dim=-1)


        feats = torch.cat([features1, features2], dim=-1)
        feat_emb = self.feature_proj(feats)


        t0, w0, t1, w1 = self._build_clouds(hidden, token_type_ids, attention_mask)
        ot_scalar = self.sinkhorn(w0, t0, w1, t1)
        ot_emb = self.ot_norm(self.ot_proj(torch.log1p(ot_scalar).unsqueeze(-1)))


        gate = torch.sigmoid(torch.norm(feats, dim=-1, keepdim=True))
        content_vec = content_vec * (1 - gate);  style_vec = style_vec * gate

        rep = torch.cat([style_vec, feat_emb, ot_emb], dim=-1)
        logits = self.classifier(rep).squeeze(-1)
        return logits, z


import torch, torch.nn.functional as F
from tqdm.auto import tqdm


def train_model(model,
                dataloader,
                epochs       = 3,
                λ            = 0.1,
                τ            = 0.07,
                optimizer    = None,
                device       = "cuda" if torch.cuda.is_available() else "cpu"):
    if optimizer is None:
        from torch.optim import AdamW
        optimizer = AdamW(model.parameters(), lr=1e-5, weight_decay=1e-4)

    for ep in range(1, epochs + 1):
        model.train()
        running = 0.0
        for batch in tqdm(dataloader, desc=f"Epoch {ep}", leave=False):
            ids   = batch["input_ids"].to(device)
            attn  = batch["attention_mask"].to(device)
            ttype = batch["token_type_ids"].to(device)
            f1    = batch["features1"].to(device)
            f2    = batch["features2"].to(device)
            lbl   = batch["label"].to(device).float()

            logits, z = model(ids, attn, ttype, f1, f2)

            loss_bce = F.binary_cross_entropy_with_logits(logits, lbl)
            loss_con = info_nce(z, lbl.long(), tau=τ)       # use 0/1 as class id
            loss     = loss_bce + λ * loss_con

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running += loss.item()

        print(f"Epoch {ep}/{epochs} – BCE+SupCon: {running/len(dataloader):.4f}")
    return model, optimizer

from torch.optim import AdamW

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ContrastiveOTModel().to(device)
optimizer = AdamW([
    {"params": model.encoder.parameters(), "lr": 1e-5},
    {"params": model.style_proj.parameters(), "lr": 1e-4},
    {"params": model.cl_proj.parameters(),    "lr": 1e-4},
    {"params": model.feature_proj.parameters(),"lr": 1e-4},
    {"params": model.ot_proj.parameters(),    "lr": 1e-4},
    {"params": model.classifier.parameters(), "lr": 1e-4},
])

model, optimizer = train_model(model, train_loader,
                               epochs=3, λ=0.1, τ=0.07,
                               optimizer=optimizer,
                               device=device)

torch.save({'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()},
           "/content/drive/My Drive/ot_model_contrastive.pt")
