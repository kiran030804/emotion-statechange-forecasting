import argparse
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from model import (
    set_seed,
    load_and_prepare_train,
    add_surface_features,
    build_user_map,
    apply_user_map,
    get_tokenizer,
    StateChangeDataset,
    EmotionDeltaModel,
)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("train_path", required=True)
    ap.add_argument("model_out", required=True)
    ap.add_argument("epochs", type=int, default=8)
    ap.add_argument("batch_size", type=int, default=16)
    ap.add_argument("lr", type=float, default=7e-5)
    ap.add_argument("weight_decay", type=float, default=0.01)
    ap.add_argument("max_len", type=int, default=128)
    ap.add_argument("scale", type=float, default=2.0)
    ap.add_argument("seed", type=int, default=42)
    return ap.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    train_df = load_and_prepare_train(args.train_path, scale=args.scale)
    train_df = add_surface_features(train_df)

    user_map = build_user_map(train_df)
    train_df = apply_user_map(train_df, user_map)
    num_users = len(user_map)

    tokenizer = get_tokenizer()
    train_ds = StateChangeDataset(train_df, tokenizer=tokenizer, max_len=args.max_len, is_test=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)

    model = EmotionDeltaModel(num_users=num_users).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    for e in range(args.epochs):
        model.train()
        total = 0.0
        for b in train_loader:
            optimizer.zero_grad()
            pred = model(
                b["input_ids"].to(device),
                b["attention_mask"].to(device),
                b["user_id"].to(device),
                b["is_words"].to(device),
                b["length"].to(device),
                b["exclaims"].to(device),
                b["questions"].to(device),
                b["prev_v"].to(device),
                b["prev_a"].to(device),
            )
            loss = criterion(pred, b["targets"].to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += loss.item()

        print(f"Full-train epoch {e+1}/{args.epochs} | loss={total/len(train_loader):.4f}")

    os.makedirs(os.path.dirname(args.model_out), exist_ok=True) if os.path.dirname(args.model_out) else None
    ckpt = {
        "state_dict": model.state_dict(),
        "user_map": user_map,
        "max_len": args.max_len,
        "scale": args.scale,
        "model_name": "roberta-base",
    }
    torch.save(ckpt, args.model_out)
    print("Saved final model to", args.model_out)


if __name__ == "__main__":
    main()
