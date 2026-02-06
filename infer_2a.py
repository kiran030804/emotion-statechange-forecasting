import argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from model import (
    load_and_prepare_test,
    add_surface_features,
    apply_user_map,
    get_tokenizer,
    StateChangeDataset,
    EmotionDeltaModel,
)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_path", required=True)
    ap.add_argument("test_path", required=True)   
    ap.add_argument("output_path", required=True)
    ap.add_argument("batch_size", type=int, default=32)
    return ap.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    ckpt = torch.load(args.model_path, map_location=device)
    user_map = ckpt["user_map"]
    max_len = int(ckpt["max_len"])
    scale = float(ckpt["scale"])

    test_df = load_and_prepare_test(args.test_path, scale=scale)
    test_df = add_surface_features(test_df)
    test_df = apply_user_map(test_df, user_map)

    forecast_df = test_df[test_df["is_forecasting_user"] == True].copy()
    forecast_df.sort_values(["user_id", "timestamp"], inplace=True)
    last_rows = forecast_df.groupby("user_id", as_index=False).tail(1).reset_index(drop=True)

    tokenizer = get_tokenizer()
    ds = StateChangeDataset(last_rows, tokenizer=tokenizer, max_len=max_len, is_test=True)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)

    model = EmotionDeltaModel(num_users=len(user_map)).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    preds = []
    with torch.no_grad():
        for b in loader:
            p = model(
                b["input_ids"].to(device),
                b["attention_mask"].to(device),
                b["user_id"].to(device),
                b["is_words"].to(device),
                b["length"].to(device),
                b["exclaims"].to(device),
                b["questions"].to(device),
                b["prev_v"].to(device),
                b["prev_a"].to(device),
            ).cpu().numpy()
            preds.append(p)

    preds = np.vstack(preds)

    pred_state_v = preds[:, 0] * scale
    pred_state_a = preds[:, 1] * scale

    out = pd.DataFrame({
        "user_id": last_rows["user_id"].values,
        "pred_state_change_valence": pred_state_v,
        "pred_state_change_arousal": pred_state_a,
    })
    out.to_csv(args.output_path, index=False)
    print(f"Wrote {args.output_path} with shape {out.shape}")
    print(out.head())


if __name__ == "__main__":
    main()
