"""
=============================================================
  Visualize Training Curves
  Đọc training_history.csv của 3 model deep learning và vẽ:
    1. plot_curves_overview.png  — 4 subplot (loss/AUROC/AUPRC/F1)
                                   so sánh 3 model trên 1 ảnh
    2. plot_curves_<model>.png   — chi tiết từng model (3 ảnh)
    3. plot_val_auroc_compare.png — chỉ val AUROC, dễ nhìn
=============================================================
"""

import os
import argparse
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# Màu nhất quán giữa các plot
COLORS = {
    "Bi-LSTM":     "#e74c3c",
    "LSTM":        "#3498db",
    "Transformer": "#2ecc71",
}


def load_history(ckpt_dir, model_name):
    csv_path = os.path.join(ckpt_dir, "training_history.csv")
    if not os.path.exists(csv_path):
        print(f"[!] Không tìm thấy {csv_path}")
        return None
    df = pd.DataFrame(pd.read_csv(csv_path))
    df["model"] = model_name
    return df


def plot_single_model(df, name, out_path):
    """Vẽ chi tiết 1 model trên 4 subplot."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    color = COLORS.get(name, "#34495e")

    # Train loss vs Val loss
    ax = axes[0, 0]
    ax.plot(df["epoch"], df["train_loss"], color=color, lw=2,
            marker="o", markersize=4, label="Train loss")
    ax.plot(df["epoch"], df["val_loss"], color=color, lw=2, ls="--",
            marker="s", markersize=4, alpha=0.7, label="Val loss")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
    ax.set_title(f"{name} — Loss Curves")
    ax.legend(); ax.grid(alpha=0.3)

    # Val AUROC
    ax = axes[0, 1]
    ax.plot(df["epoch"], df["val_auroc"], color=color, lw=2,
            marker="o", markersize=4)
    best_ep = df.loc[df["val_auroc"].idxmax(), "epoch"]
    best_v  = df["val_auroc"].max()
    ax.axvline(best_ep, color="red", ls=":", lw=1, alpha=0.6,
               label=f"Best: ep{int(best_ep)} ({best_v:.4f})")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Validation AUROC")
    ax.set_title(f"{name} — Val AUROC")
    ax.legend(); ax.grid(alpha=0.3)

    # Val AUPRC
    ax = axes[1, 0]
    ax.plot(df["epoch"], df["val_auprc"], color=color, lw=2,
            marker="o", markersize=4)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Validation AUPRC")
    ax.set_title(f"{name} — Val AUPRC")
    ax.grid(alpha=0.3)

    # Val F1
    ax = axes[1, 1]
    ax.plot(df["epoch"], df["val_f1"], color=color, lw=2,
            marker="o", markersize=4)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Validation F1")
    ax.set_title(f"{name} — Val F1")
    ax.grid(alpha=0.3)

    plt.suptitle(f"Training Curves — {name}", fontsize=14, y=1.00)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


def plot_overview(dfs, out_path):
    """4 subplot, mỗi cái có 3 đường model."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    metric_specs = [
        ("train_loss", "Train Loss",         axes[0, 0]),
        ("val_loss",   "Val Loss",           axes[0, 1]),
        ("val_auroc",  "Val AUROC",          axes[1, 0]),
        ("val_auprc",  "Val AUPRC",          axes[1, 1]),
    ]

    for metric, title, ax in metric_specs:
        for name, df in dfs.items():
            if df is None or metric not in df.columns:
                continue
            ax.plot(df["epoch"], df[metric],
                    color=COLORS.get(name, "#34495e"),
                    lw=2, marker="o", markersize=3, label=name)
        ax.set_xlabel("Epoch"); ax.set_ylabel(title)
        ax.set_title(title); ax.legend(); ax.grid(alpha=0.3)

    plt.suptitle("Training Curves Overview — 3 Deep Learning Models",
                 fontsize=14, y=1.00)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


def plot_val_auroc_compare(dfs, out_path):
    """Chỉ val_AUROC, to và rõ — ảnh chính cho báo cáo."""
    plt.figure(figsize=(11, 7))
    for name, df in dfs.items():
        if df is None:
            continue
        plt.plot(df["epoch"], df["val_auroc"],
                 color=COLORS.get(name, "#34495e"),
                 lw=2.5, marker="o", markersize=5, label=name)
        # Đánh dấu best epoch
        best_ep = df.loc[df["val_auroc"].idxmax(), "epoch"]
        best_v  = df["val_auroc"].max()
        plt.scatter([best_ep], [best_v],
                    color=COLORS.get(name), s=200, marker="*",
                    edgecolors="black", lw=1.5, zorder=10)

    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Validation AUROC", fontsize=12)
    plt.title("Validation AUROC over Epochs — Model Comparison",
              fontsize=13)
    plt.legend(fontsize=11, loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bilstm_dir",
                        default="checkpoints_v4")
    parser.add_argument("--lstm_dir",
                        default="checkpoints_lstm")
    parser.add_argument("--transformer_dir",
                        default="checkpoints_transformer")
    parser.add_argument("--out_dir",
                        default="results_compare")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    print("=" * 60)
    print("  Vẽ training curves")
    print("=" * 60)

    dfs = {
        "Bi-LSTM":     load_history(args.bilstm_dir,      "Bi-LSTM"),
        "LSTM":        load_history(args.lstm_dir,        "LSTM"),
        "Transformer": load_history(args.transformer_dir, "Transformer"),
    }
    dfs = {k: v for k, v in dfs.items() if v is not None}

    if not dfs:
        print("  Không tìm thấy file training_history.csv nào!")
        return

    print(f"\n  Models loaded: {list(dfs.keys())}\n")

    # 1. Overview
    plot_overview(dfs, os.path.join(args.out_dir, "plot_curves_overview.png"))

    # 2. Mỗi model 1 ảnh
    for name, df in dfs.items():
        safe = name.lower().replace("-", "_")
        plot_single_model(df, name,
                          os.path.join(args.out_dir, f"plot_curves_{safe}.png"))

    # 3. Val AUROC compare
    plot_val_auroc_compare(dfs,
                           os.path.join(args.out_dir,
                                        "plot_val_auroc_compare.png"))

    print(f"\n  Hoàn thành! Output: {args.out_dir}/")


if __name__ == "__main__":
    main()
