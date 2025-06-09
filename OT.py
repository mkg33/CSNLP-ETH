
!pip install nltk pandas tqdm requests textstat
!pip install pot transformers torch geomloss

!pip install nltk pandas tqdm requests textstat
!pip install pot transformers torch geomloss

import os
import json
import pandas as pd
import requests
import zipfile
from tqdm import tqdm
from pathlib import Path

import nltk
nltk.download('punkt')
nltk.download('punkt_tab')


ZIP_URL = "https://zenodo.org/records/14891299/files/pan25-multi-author-analysis.zip?download=1"
ZIP_PATH = "pan25-multi-author-analysis.zip"
EXTRACT_FOLDER = "pan25_data"

import pickle
from google.colab import drive
drive.mount('/content/gdrive',force_remount=True)

import ot
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


def download_file(url, save_path):
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get("content-length", 0))

    with open(save_path, "wb") as file, tqdm(
        desc="Downloading Dataset",
        total=total_size,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for chunk in response.iter_content(chunk_size=1024):
            file.write(chunk)
            bar.update(len(chunk))


def unzip_file(zip_path, extract_folder):
    os.makedirs(extract_folder, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_folder)
    print(f"Extracted to {extract_folder}")


download_file(ZIP_URL, ZIP_PATH)
unzip_file(ZIP_PATH, EXTRACT_FOLDER)


DATASET_ROOT = Path(EXTRACT_FOLDER)


SPLITS = ["train", "validation"]
DIFFICULTY_LEVELS = ["easy", "medium", "hard"]


def read_problem_data(problem_file):
    text_path = problem_file
    json_path = text_path.with_name(f"truth-{text_path.stem}.json")


    with open(text_path, "r", encoding="utf-8") as f:
        text = f.read().strip()


    with open(json_path, "r", encoding="utf-8") as f:
        truth = json.load(f)

    return text, truth["changes"]


def process_dataset(split):
    dataset = []
    total_mismatched_pairs = 0  # Counter for total mismatched pairs

    for difficulty in DIFFICULTY_LEVELS:
        print(f"Processing {split} set - {difficulty} difficulty...")
        folder_path = DATASET_ROOT / difficulty / split

        for problem_file in tqdm(folder_path.glob("problem-*.txt")):
            with open(problem_file, "r", encoding="utf-8") as f:
                lines = f.readlines()


            sentences = [line.strip() for line in lines if line.strip()]


            _, labels = read_problem_data(problem_file)

            # Check for label mismatch
            if len(labels) != len(sentences) - 1:
                mismatch_count = abs(len(labels) - (len(sentences) - 1))
                total_mismatched_pairs += mismatch_count

                print(f"\n Label mismatch in {problem_file}")
                print(f" Total sentences (lines): {len(sentences)} | Expected labels: {len(labels)}")
                print(f" Mismatched pairs in this file: {mismatch_count}")


                print("\n- Mismatched Sentence Pairs -")
                for i in range(min(len(sentences) - 1, len(labels), 5)):
                    print(f"[{i}] {sentences[i]} || {sentences[i+1]} --> Label: {labels[i]}")

                print("- End of mismatch -\n")
                continue


            for i in range(len(sentences) - 1):
                dataset.append({
                    "difficulty": difficulty,
                    "split": split,
                    "problem_id": problem_file.stem,
                    "sentence1": sentences[i],
                    "sentence2": sentences[i + 1],
                    "label": labels[i]
                })

    print(f"\n Total Mismatched Sentence-Label Pairs Across All Files: {total_mismatched_pairs}")
    return dataset

train_data = process_dataset("train")
valid_data = process_dataset("validation")


train_df = pd.DataFrame(train_data)
valid_df = pd.DataFrame(valid_data)

train_df.to_csv("train_dataset.csv", index=False)
valid_df.to_csv("valid_dataset.csv", index=False)

print("Training and validation datasets saved successfully!")


train_df = pd.read_csv("train_dataset.csv")
valid_df = pd.read_csv("valid_dataset.csv")


print("\n Label distribution in training set:")
print(train_df['label'].value_counts())

print("\n Label distribution in validation set:")
print(valid_df['label'].value_counts())

!pip install torch transformers spacy textstat scikit-learn pandas
!python -m spacy download en_core_web_sm

# features

import spacy
import textstat
import numpy as np
from collections import Counter

nlp = spacy.load("en_core_web_sm")


def extract_features(text):
    doc = nlp(text)

    # Lexical
    word_lengths = [len(token.text) for token in doc if token.is_alpha]
    avg_word_length = np.mean(word_lengths) if word_lengths else 0

    function_words = set(["the", "is", "and", "but", "or", "because", "as", "that"])
    function_word_count = sum(1 for token in doc if token.text.lower() in function_words)

    # Syntactic
    pos_counts = Counter(token.pos_ for token in doc)
    num_nouns = pos_counts.get("NOUN", 0)
    num_verbs = pos_counts.get("VERB", 0)

    sentence_lengths = [len(sent.text.split()) for sent in doc.sents]
    avg_sentence_length = np.mean(sentence_lengths) if sentence_lengths else 0

    # Stylometric
    punctuation_count = sum(1 for token in text if token in ".,!?;:-")
    readability_score = textstat.flesch_reading_ease(text)

    return [
        avg_word_length, function_word_count, num_nouns, num_verbs,
        avg_sentence_length, punctuation_count, readability_score
    ]


train_df["features1"] = train_df["sentence1"].apply(extract_features)
train_df["features2"] = train_df["sentence2"].apply(extract_features)
valid_df["features1"] = valid_df["sentence1"].apply(extract_features)
valid_df["features2"] = valid_df["sentence2"].apply(extract_features)

print(" Feature extraction completed!")

import pickle


train_df.to_csv("train_features.csv", index=False)
valid_df.to_csv("valid_features.csv", index=False)


with open("train_features.pkl", "wb") as f:
    pickle.dump(train_df, f)

with open("valid_features.pkl", "wb") as f:
    pickle.dump(valid_df, f)

from google.colab import drive
drive.mount('/content/drive')
train_df.to_pickle("/content/drive/My Drive/valid/train_features.pkl")
valid_df.to_pickle("/content/drive/My Drive/valid/valid_features.pkl")

import pickle
from google.colab import drive
drive.mount('/content/drive')

# Define file paths
train_features_path = "/content/drive/My Drive/valid/train_features.pkl"
valid_features_path = "/content/drive/My Drive/valid/valid_features.pkl"

# Load training features
with open(train_features_path, "rb") as f:
    train_df = pickle.load(f)

# Load validation features
with open(valid_features_path, "rb") as f:
    valid_df = pickle.load(f)

print(train_df.head())
print(valid_df.head())

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

# Load DeBERTa
model_name = "microsoft/deberta-v3-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)

class StyleChangeDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length=128):
        self.df = dataframe
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        enc = self.tokenizer(
            row["sentence1"], row["sentence2"],
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_token_type_ids=True,
            return_tensors="pt"
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}       # ids, mask, type_ids
        item["features1"] = torch.tensor(row["features1"], dtype=torch.float32)
        item["features2"] = torch.tensor(row["features2"], dtype=torch.float32)
        item["label"]     = torch.tensor(row["label"], dtype=torch.long)
        return item


train_dataset = StyleChangeDataset(train_df, tokenizer)
valid_dataset = StyleChangeDataset(valid_df, tokenizer)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False)

print(" Data successfully loaded into PyTorch Dataloaders!")

import torch.nn as nn
from transformers import AutoModel

import torch.nn.functional as F
from geomloss import SamplesLoss

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
from geomloss import SamplesLoss
from torch.optim import AdamW


class FactorizedAttentionModel(nn.Module):

    def __init__(self,
                 model_name="microsoft/deberta-v3-base",
                 hidden_dim=768,
                 style_dim=128,
                 content_dim=128,
                 feature_dim=7,
                 ot_blur=0.05):
        super().__init__()

        # Transformer encoder
        self.encoder = AutoModel.from_pretrained(model_name)


        self.content_proj = nn.Linear(hidden_dim, content_dim)
        self.style_proj   = nn.Linear(hidden_dim, style_dim)


        self.feature_proj = nn.Linear(feature_dim * 2, style_dim)

        # OT branch
        self.sinkhorn = SamplesLoss("sinkhorn",
                                    p=1,
                                    blur=ot_blur,    # entropic smoothing
                                    debias=True)     # Sinkhorn divergence
        self.ot_proj   = nn.Linear(1, style_dim)
        self.ot_norm   = nn.LayerNorm(style_dim)

        # Final classifier
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
                F.pad(t, (0, 0, 0, nmax - t.size(0))) if pad_left is False
                else F.pad(t, (0, nmax - t.size(0)))
                for t in tlist])

        t0 = _pad(clouds0)
        t1 = _pad(clouds1)
        w0 = _pad(w0s, pad_left=True)
        w1 = _pad(w1s, pad_left=True)
        return t0, w0, t1, w1

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
        cls = hidden[:, 0, :]


        content_vec = self.content_proj(cls)
        style_vec   = self.style_proj(cls)


        feats = torch.cat([features1, features2], dim=-1)
        feat_emb = self.feature_proj(feats)

        # OT distance
        t0, w0, t1, w1 = self._build_clouds(hidden,
                                            token_type_ids,
                                            attention_mask)

        ot_scalar = self.sinkhorn(w0, t0, w1, t1)
        ot_scalar = torch.log1p(ot_scalar)
        ot_emb = self.ot_norm(self.ot_proj(
                     ot_scalar.unsqueeze(-1)))


        gate = torch.sigmoid(torch.norm(feats, dim=-1, keepdim=True))
        content_vec = content_vec * (1 - gate)
        style_vec   = style_vec   * gate


        rep = torch.cat([style_vec, feat_emb, ot_emb], dim=-1)
        logits = self.classifier(rep).squeeze(-1)
        return logits



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = FactorizedAttentionModel().to(device)


optim = AdamW(model.parameters(), lr=1e-5, weight_decay=1e-4)

def train_model(model,
          dataloader,
          epochs=5,
          lr=1e-5,
          weight_decay=1e-4,
          device="cuda" if torch.cuda.is_available() else "cpu"):

    for ep in range(1, epochs + 1):
        model.train()
        running = 0.0

        for batch in dataloader:
            ids   = batch["input_ids"].to(device)
            attn  = batch["attention_mask"].to(device)
            ttype = batch["token_type_ids"].to(device)
            f1    = batch["features1"].to(device)
            f2    = batch["features2"].to(device)
            lbl   = batch["label"].to(device).float()

            logits = model(ids, attn, ttype, f1, f2)
            loss   = F.binary_cross_entropy_with_logits(logits, lbl)

            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()

            running += loss.item()

        print(f"Epoch {ep}/{epochs} • mean BCE: {running/len(dataloader):.4f}")

train_model(model, train_loader, epochs=3)

import torch

# Define save path in Google Drive
model_save_path = "/content/drive/My Drive/ot_model2.pt"

# Save model weights
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optim.state_dict(),
}, model_save_path)

print(f" Model saved to {model_save_path}")

import torch
from torch.optim import AdamW
from google.colab import drive
drive.mount('/content/drive')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define model path
model_load_path = "/content/drive/My Drive/ot_model2.pt"

# Load checkpoint
checkpoint = torch.load(model_load_path, map_location=device)

# Model and optimizer
model = FactorizedAttentionModel().to(device)
optimizer = AdamW(model.parameters(), lr=1e-5)

# Load saved weights
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

model.eval()

print(" Model loaded successfully!")

from sklearn.metrics import accuracy_score, f1_score
"""
def evaluate_model_batched(model, dataset, batch_size=64, dataset_name="Validation"):
    model.eval()
    all_preds, all_labels = [], []

    for i in range(0, len(dataset), batch_size):
        batch = dataset.iloc[i:i+batch_size]

        encoding = tokenizer(
            batch["sentence1"].tolist(),
            batch["sentence2"].tolist(),
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt"
        )

        input_ids = encoding["input_ids"].to(device)
        attention_mask = encoding["attention_mask"].to(device)
        features1 = torch.tensor(batch["features1"].tolist(), dtype=torch.float32).to(device)
        features2 = torch.tensor(batch["features2"].tolist(), dtype=torch.float32).to(device)
        labels = torch.tensor(batch["label"].tolist(), dtype=torch.float32).to(device)

        with torch.no_grad():

            logits, _, _ = model(input_ids, attention_mask, features1, features2)
            preds = torch.sigmoid(logits).cpu().numpy() > 0.5

        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="macro")
    print(f"{dataset_name} Set - Accuracy: {accuracy:.4f}, Macro F1-Score: {f1:.4f}")
    return accuracy, f1

# Evaluate on validation set
valid_acc, valid_f1 = evaluate_model_batched(model, valid_df, dataset_name="Validation")

import torch.nn.functional as F
for difficulty in ["easy", "medium", "hard"]:
    subset = valid_df[valid_df["difficulty"] == difficulty]
    evaluate_model_batched(model, subset, dataset_name=f"{difficulty.capitalize()} Validation")
"""

# Official evaluator

import requests
from google.colab import drive

drive.mount('/content/drive')


save_path = "/content/drive/My Drive/evaluator.py"


url = "https://raw.githubusercontent.com/pan-webis-de/pan-code/master/clef25/multi-author-analysis/evaluator/evaluator.py"


response = requests.get(url)
with open(save_path, "w") as f:
    f.write(response.text)

print("evaluator.py saved to:", save_path)

import os
import requests
import zipfile

url = "https://zenodo.org/records/14891299/files/pan25-multi-author-analysis.zip?download=1"
zip_path = "/content/drive/My Drive/pan25.zip"
extract_path = "/content/drive/My Drive/pan25_data"

"""
print("Downloading PAN25 dataset...")
r = requests.get(url)
with open(zip_path, "wb") as f:
    f.write(r.content)

print("Unzipping...")
with zipfile.ZipFile(zip_path, "r") as zip_ref:
    zip_ref.extractall(extract_path)

print("Extracted to", extract_path)
os.listdir(extract_path)
"""

import os
import json
import torch
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
import spacy
import textstat
from collections import Counter
from transformers import AutoTokenizer, AutoModel
from google.colab import drive

drive.mount('/content/drive')


with open("/content/drive/My Drive/valid/train_features.pkl", "rb") as f:
    train_df = pickle.load(f)

with open("/content/drive/My Drive/valid/valid_features.pkl", "rb") as f:
    valid_df = pickle.load(f)

# Load tokenizer
model_name = "microsoft/deberta-v3-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)

nlp = spacy.load("en_core_web_sm")


def extract_features(text):
    doc = nlp(text)
    word_lengths = [len(token.text) for token in doc if token.is_alpha]
    avg_word_length = np.mean(word_lengths) if word_lengths else 0
    function_words = {"the", "is", "and", "but", "or", "because", "as", "that"}
    function_word_count = sum(1 for token in doc if token.text.lower() in function_words)
    pos_counts = Counter(token.pos_ for token in doc)
    num_nouns = pos_counts.get("NOUN", 0)
    num_verbs = pos_counts.get("VERB", 0)
    sentence_lengths = [len(sent.text.split()) for sent in doc.sents]
    avg_sentence_length = np.mean(sentence_lengths) if sentence_lengths else 0
    punctuation_count = sum(1 for token in text if token in ".,!?;:-")
    readability_score = textstat.flesch_reading_ease(text)
    return [avg_word_length, function_word_count, num_nouns, num_verbs,
            avg_sentence_length, punctuation_count, readability_score]


import torch.nn as nn
import torch.nn.functional as F



model = FactorizedAttentionModel().to("cuda" if torch.cuda.is_available() else "cpu")
checkpoint = torch.load("/content/drive/My Drive/ot_model2.pt", map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
print("Model loaded.")


def predict_and_save(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for file_name in tqdm(sorted(os.listdir(input_folder))):
        if not file_name.startswith("problem-") or not file_name.endswith(".txt"): continue
        problem_id = file_name[8:-4]
        with open(os.path.join(input_folder, file_name), "r") as f:
            sentences = [line.strip() for line in f if line.strip()]

        changes = []
        for i in range(len(sentences) - 1):
            sent1, sent2 = sentences[i], sentences[i + 1]
            encoding = tokenizer(sent1, sent2, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)
            f1 = torch.tensor([extract_features(sent1)], dtype=torch.float32).to(device)
            f2 = torch.tensor([extract_features(sent2)], dtype=torch.float32).to(device)
            logits = model(encoding["input_ids"], encoding["attention_mask"], encoding["token_type_ids"], f1, f2)
            pred = torch.sigmoid(logits).item() > 0.5
            changes.append(int(pred))

        out_path = os.path.join(output_folder, f"solution-problem-{problem_id}.json")
        with open(out_path, "w") as f:
            json.dump({"changes": changes}, f, indent=2)

    print(f"Predictions saved to: {output_folder}")

predict_and_save("/content/drive/My Drive/pan25_data/easy/validation", "/content/drive/My Drive/pan25_output/easy")
predict_and_save("/content/drive/My Drive/pan25_data/medium/validation", "/content/drive/My Drive/pan25_output/medium")
predict_and_save("/content/drive/My Drive/pan25_data/hard/validation", "/content/drive/My Drive/pan25_output/hard")

import shutil

os.makedirs("/content/drive/My Drive/eval_result", exist_ok=True)

"""
src_dir = "/content/drive/My Drive/pan25_data/easy/validation"
dest_dir = "/content/drive/My Drive/pan25_data/easy"

for filename in os.listdir(src_dir):
    if filename.startswith("truth-problem-"):
        shutil.move(
            os.path.join(src_dir, filename),
            os.path.join(dest_dir, filename)
        )

src_dir = "/content/drive/My Drive/pan25_data/medium/validation"
dest_dir = "/content/drive/My Drive/pan25_data/medium"

for filename in os.listdir(src_dir):
    if filename.startswith("truth-problem-"):
        shutil.move(
            os.path.join(src_dir, filename),
            os.path.join(dest_dir, filename)
        )

src_dir = "/content/drive/My Drive/pan25_data/hard/validation"
dest_dir = "/content/drive/My Drive/pan25_data/hard"

for filename in os.listdir(src_dir):
    if filename.startswith("truth-problem-"):
        shutil.move(
            os.path.join(src_dir, filename),
            os.path.join(dest_dir, filename)
        )
"""

!python "/content/drive/My Drive/evaluator.py" \
  --predictions "/content/drive/My Drive/pan25_output" \
  --truth "/content/drive/My Drive/pan25_data" \
  --output "/content/drive/My Drive/eval_result"
