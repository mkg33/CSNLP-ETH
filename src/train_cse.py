import json
import logging
import os
import shutil
from datetime import datetime

from sentence_transformers import SentenceTransformer, InputExample, losses, util
import torch
from torch.utils.data import DataLoader

from sklearn.metrics import accuracy_score

log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = os.path.join(log_dir, f"train_log_{timestamp}.log")
# lr = 1e-5
batch_size = 256
max_seq_length = 192

logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
console.setFormatter(formatter)
logging.getLogger().addHandler(console)

logging.info("Starting SimCSE training script.")


def load_data_from_json(json_path):
    logging.info(f"Loading data from {json_path}")
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)["data"]
    examples = [
        InputExample(
            texts=[item["sentence1"], item["sentence2"]],
            label=float(item["label"])
        )
        for item in data
    ]
    logging.info(f"Loaded {len(examples)} examples.")
    return examples

json_easy_train = 'data/processed/easy_train.json'
json_medium_train = 'data/processed/medium_train.json'
json_hard_train = 'data/processed/hard_train.json'

model_name = 'all-MiniLM-L6-v2'

output_dir = 'output/simcse-easy-model'


train_easy = load_data_from_json(json_easy_train)
train_medium = load_data_from_json(json_medium_train)
train_hard = load_data_from_json(json_hard_train)

train_data = train_easy + train_medium + train_hard

train_dataloader = DataLoader(train_data, shuffle=True, batch_size=batch_size)

logging.info(f"Loading model {model_name}")
model = SentenceTransformer(model_name)
model.max_seq_length = max_seq_length

train_loss = losses.CosineSimilarityLoss(model)

def evaluate(model, val_data, threshold=0.5, batch_size=128, name="val"):
    logging.info(f"Evaluating on {name}...")

    sents1 = [ex.texts[0] for ex in val_data]
    sents2 = [ex.texts[1] for ex in val_data]
    labels = [int(ex.label) for ex in val_data]

    embs1 = model.encode(sents1, convert_to_tensor=True, batch_size=batch_size, show_progress_bar=False)
    embs2 = model.encode(sents2, convert_to_tensor=True, batch_size=batch_size, show_progress_bar=False)

    # Cosine similarities
    sims = util.cos_sim(embs1, embs2).diagonal()
    preds = (sims >= threshold).int().tolist()

    sims_np = sims.cpu().numpy()
    losses = [
        1 - sim if label == 1 else sim
        for sim, label in zip(sims_np, labels)
    ]
    avg_loss = sum(losses) / len(losses)

    acc = accuracy_score(labels, preds)
    logging.info(f"{name} Accuracy: {acc:.4f}, Loss: {avg_loss:.4f}")

    return acc, avg_loss

best_models = []  # (accuracy, path)


def save_best_model(model, acc, epoch, output_base='output/best_models', top_k=5):
    os.makedirs(output_base, exist_ok=True)
    model_name = f"model_epoch{epoch+1}_acc{acc:.4f}"
    model_path = os.path.join(output_base, model_name)

    model.save(model_path)
    logging.info(f"Saved new checkpoint: {model_path}")

    best_models.append((acc, model_path))
    best_models.sort(key=lambda x: x[0], reverse=True)  # Sort by accuracy descending

    while len(best_models) > top_k:
        _, path_to_remove = best_models.pop()
        shutil.rmtree(path_to_remove)
        logging.info(f"Removed checkpoint: {path_to_remove}")

warmup_steps = int(len(train_dataloader) * 0.1)
optimizer_params = {'lr': 2e-5, 'eps': 1e-6}
optimizer = torch.optim.AdamW(model.parameters(), **optimizer_params)
epochs = 10

val_easy_path = 'data/processed/easy_valid.json'
val_medium_path = 'data/processed/medium_valid.json'
val_hard_path = 'data/processed/hard_valid.json'

val_easy = load_data_from_json(val_easy_path)
val_medium = load_data_from_json(val_medium_path)
val_hard = load_data_from_json(val_hard_path)

val_sets = [("val_easy", val_easy), ("val_medium", val_medium), ("val_hard", val_hard)]

# Train
for epoch in range(epochs):
    logging.info(f"Epoch {epoch + 1}/{epochs} started")
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=1,
        warmup_steps=warmup_steps,
        optimizer_class=torch.optim.AdamW,
        optimizer_params=optimizer_params,
        show_progress_bar=True,
    )

    val_scores = {}
    for name, val_data in val_sets:
        acc, loss = evaluate(model, val_data, name=name)
        val_scores[name] = {"accuracy": acc, "loss": loss}

    save_best_model(model, val_scores["val_hard"]["accuracy"], epoch)

# === Save model ===
os.makedirs(output_dir, exist_ok=True)
logging.info(f"Model saved to {output_dir}")