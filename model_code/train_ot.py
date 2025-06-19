import os
import random
import pickle
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from geomloss import SamplesLoss
from torch.optim import AdamW


SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class OTModel(nn.Module):
    def __init__(
        self,
        model_name="microsoft/deberta-v3-base",
        hidden_dim=768,
        style_dim=128,
        content_dim=128,
        feature_dim=7,
        gate_hidden=64,
        ot_blur=0.05,
    ):
        super().__init__()

        self.encoder = AutoModel.from_pretrained(model_name)

        self.content_proj = nn.Linear(hidden_dim, content_dim)
        self.style_proj   = nn.Linear(hidden_dim, style_dim)
        self.feature_proj = nn.Linear(feature_dim * 2, style_dim)
        # OT
        self.sinkhorn  = SamplesLoss("sinkhorn", p=1, blur=ot_blur, debias=True)
        self.ot_proj   = nn.Linear(1, style_dim)
        self.ot_norm   = nn.LayerNorm(style_dim)
        # dynamic gates
        gate_in_dim = style_dim + content_dim + style_dim + style_dim
        self.gate_net = nn.Sequential(
            nn.Linear(gate_in_dim, gate_hidden),
            nn.ReLU(),
            nn.Linear(gate_hidden, style_dim)
        )

        self.classifier = nn.Linear(style_dim * 3, 1)

    @staticmethod
    def _build_clouds(seq_emb, token_type_ids, attention_mask):
        B, L, D = seq_emb.size()
        clouds0, clouds1, w0s, w1s = [], [], [], []
        for i in range(B):
            real = attention_mask[i] == 1
            seg0 = real & (token_type_ids[i] == 0)
            seg1 = real & (token_type_ids[i] == 1)
            c0 = seq_emb[i][seg0]
            c1 = seq_emb[i][seg1]
            # handle emp.
            if c0.size(0) == 0:
                c0 = seq_emb[i, :1].detach() * 0.0
            if c1.size(0) == 0:
                c1 = seq_emb[i, :1].detach() * 0.0
            n0, n1 = c0.size(0), c1.size(0)
            w0 = torch.full((n0,), 1.0/n0, device=seq_emb.device)
            w1 = torch.full((n1,), 1.0/n1, device=seq_emb.device)
            clouds0.append(c0); clouds1.append(c1)
            w0s.append(w0);     w1s.append(w1)
        def _pad(tlist, pad_left=False):
            nmax = max(t.size(0) for t in tlist)
            return torch.stack([
                F.pad(t, (0, 0, 0, nmax - t.size(0))) if not pad_left
                else F.pad(t, (0, nmax - t.size(0)))
                for t in tlist
            ])
        t0 = _pad(clouds0)
        t1 = _pad(clouds1)
        w0 = _pad(w0s, pad_left=True)
        w1 = _pad(w1s, pad_left=True)
        return t0, w0, t1, w1

    def forward(self, input_ids, attention_mask, token_type_ids, features1, features2):

        hid = self.encoder(
            input_ids, attention_mask, return_dict=True
        ).last_hidden_state
        hid = hid * attention_mask.unsqueeze(-1)
        cls = hid[:, 0, :]

        content_vec = self.content_proj(cls)
        style_vec   = self.style_proj(cls)

        feats = torch.cat([features1, features2], dim=-1)
        feat_emb = self.feature_proj(feats)
        # embed OT
        t0, w0, t1, w1 = self._build_clouds(hid, token_type_ids, attention_mask)
        ot_scalar = torch.log1p(self.sinkhorn(w0, t0, w1, t1))
        ot_emb    = self.ot_norm(self.ot_proj(ot_scalar.unsqueeze(-1)))
        # vector gate
        gate_inp = torch.cat([feat_emb, content_vec, style_vec, ot_emb], dim=-1)
        gate     = torch.sigmoid(self.gate_net(gate_inp))
        mix      = (1 - gate) * content_vec + gate * style_vec

        rep    = torch.cat([mix, feat_emb, ot_emb], dim=-1)
        logits = self.classifier(rep).squeeze(-1)
        return logits


class StyleChangeDataset(Dataset):
    def __init__(self, df, tokenizer, max_length=128):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        enc = self.tokenizer(
            row['sentence1'], row['sentence2'],
            padding='max_length', truncation=True,
            max_length=self.max_length,
            return_tensors='pt', return_token_type_ids=True
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item['features1'] = torch.tensor(row['features1'], dtype=torch.float32)
        item['features2'] = torch.tensor(row['features2'], dtype=torch.float32)
        item['label']     = torch.tensor(row['label'], dtype=torch.float32)
        return item


def train_model(model, dataloader, optimizer, device, epochs=3):
    criterion = nn.BCEWithLogitsLoss()
    model.train()
    for ep in range(1, epochs+1):
        total_loss = 0.0
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(
                batch['input_ids'], batch['attention_mask'], batch['token_type_ids'],
                batch['features1'], batch['features2']
            )
            loss = criterion(logits, batch['label'])
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {ep}/{epochs} - Loss: {total_loss/len(dataloader):.4f}")


if __name__ == '__main__':

    train_features_path = '/home/mgwozdz/valid/train_features.pkl'
    valid_features_path = '/home/mgwozdz/valid/valid_features.pkl'
    save_dir = '/home/mgwozdz/OTmodel'
    os.makedirs(save_dir, exist_ok=True)


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    with open(train_features_path, 'rb') as f:
        train_df = pickle.load(f)
    with open(valid_features_path, 'rb') as f:
        valid_df = pickle.load(f)


    model_name = 'microsoft/deberta-v3-base'
    tokenizer = AutoTokenizer.from_pretrained(model_name)


    train_dataset = StyleChangeDataset(train_df, tokenizer)
    train_loader  = DataLoader(train_dataset, batch_size=32, shuffle=True)


    model = OTModel().to(device)
    optimizer = AdamW(model.parameters(), lr=1e-5, weight_decay=1e-4)


    train_model(model, train_loader, optimizer, device, epochs=3)

    save_path = os.path.join(save_dir, 'ot_model.pt')
    torch.save({'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}, save_path)
    print(f"Model and optimizer saved to {save_path}")
