import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import RobertaTokenizer, RobertaModel


def set_seed(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_and_prepare_train(train_path: str, scale: float = 2.0) -> pd.DataFrame:
    df = pd.read_csv(train_path)
    df.columns = df.columns.str.strip()

    required = [
        "user_id", "text", "timestamp", "is_words", "valence", "arousal",
        "state_change_valence", "state_change_arousal"
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in train file: {missing}")

    df["text"] = df["text"].astype(str).str.replace("â€™", "’")
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df.sort_values(["user_id", "timestamp"], inplace=True)

    df = df.dropna(subset=["state_change_valence", "state_change_arousal"]).reset_index(drop=True)

    for col in ["valence", "arousal", "state_change_valence", "state_change_arousal"]:
        df[col] = df[col] / scale

    return df


def load_and_prepare_test(marker_path: str, scale: float = 2.0) -> pd.DataFrame:
    df = pd.read_csv(marker_path)
    df.columns = df.columns.str.strip()

    required = ["user_id", "text", "timestamp", "is_words", "valence", "arousal", "is_forecasting_user"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in marker file: {missing}")

    df["text"] = df["text"].astype(str).str.replace("â€™", "’")
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df.sort_values(["user_id", "timestamp"], inplace=True)

    for col in ["valence", "arousal"]:
        df[col] = df[col] / scale

    return df


def add_surface_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["length"] = df["text"].astype(str).apply(len)
    df["exclaims"] = df["text"].astype(str).str.count("!")
    df["questions"] = df["text"].astype(str).str.count(r"\?")

    for c in ["length", "exclaims", "questions"]:
        df[c] = df[c] / (df[c].std() + 1e-6)

    return df


def build_user_map(train_df: pd.DataFrame) -> dict:
    users = train_df["user_id"].astype(str).unique().tolist()
    user_map = {u: i for i, u in enumerate(users)}
    user_map["__UNK__"] = len(user_map)
    return user_map


def apply_user_map(df: pd.DataFrame, user_map: dict) -> pd.DataFrame:
    df = df.copy()
    df["user_id_code"] = df["user_id"].astype(str).map(lambda u: user_map.get(u, user_map["__UNK__"]))
    return df


def get_tokenizer():
    return RobertaTokenizer.from_pretrained("roberta-base")


class StateChangeDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer, max_len: int = 128, is_test: bool = False):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.is_test = is_test

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        r = self.df.loc[idx]
        enc = self.tokenizer(
            r["text"],
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )

        item = {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),

            "user_id": torch.tensor(int(r["user_id_code"]), dtype=torch.long),
            "is_words": torch.tensor(float(r["is_words"]), dtype=torch.float),

            "length": torch.tensor(float(r["length"]), dtype=torch.float),
            "exclaims": torch.tensor(float(r["exclaims"]), dtype=torch.float),
            "questions": torch.tensor(float(r["questions"]), dtype=torch.float),

            # previous affect
            "prev_v": torch.tensor(float(r["valence"]), dtype=torch.float),
            "prev_a": torch.tensor(float(r["arousal"]), dtype=torch.float),
        }

        if not self.is_test:
            item["targets"] = torch.tensor(
                [float(r["state_change_valence"]), float(r["state_change_arousal"])],
                dtype=torch.float
            )
        return item


class EmotionDeltaModel(nn.Module):
    def __init__(self, num_users: int):
        super().__init__()
        self.roberta = RobertaModel.from_pretrained("roberta-base")

        for p in self.roberta.parameters():
            p.requires_grad = False

        for name, p in self.roberta.named_parameters():
            if ("encoder.layer.9" in name or "encoder.layer.10" in name or "encoder.layer.11" in name):
                p.requires_grad = True

        self.lstm = nn.LSTM(768, 128, bidirectional=True, batch_first=True)
        self.user_emb = nn.Embedding(num_users, 32)

        self.shared = nn.Sequential(
            nn.Linear(256 + 32 + 1 + 3 + 2, 256),
            nn.ReLU(),
            nn.LayerNorm(256),
            nn.Dropout(0.3),
        )

        self.dv = nn.Sequential(nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 1))
        self.da = nn.Sequential(nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 1))

    def forward(self, input_ids, attention_mask, user_id, is_words, length, exclaims, questions, prev_v, prev_a):
        x = self.roberta(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        x, _ = self.lstm(x)
        pooled = x[:, -1, :] 
        u = self.user_emb(user_id)

        feats = torch.cat([
            pooled,
            u,
            is_words.unsqueeze(1),
            length.unsqueeze(1),
            exclaims.unsqueeze(1),
            questions.unsqueeze(1),
            prev_v.unsqueeze(1),
            prev_a.unsqueeze(1),
        ], dim=1)

        h = self.shared(feats)
        return torch.cat([self.dv(h), self.da(h)], dim=1)
