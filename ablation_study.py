"""
=============================================================
  Ablation Study — Bi-LSTM ICU Mortality
  Sweep 4 yếu tố để chứng minh chọn cấu hình hợp lý:

  1. Missing mask:  ON  vs  OFF
  2. pos_weight:    1, 3, 5, 8
  3. Hidden size:   64, 128, 256
  4. n_layers:      1, 2, 3

  Output:
    ablation/
    ├── ablation_results.csv    ← bảng đầy đủ
    ├── ablation_results.md     ← bảng markdown cho báo cáo
    ├── ablation_mask.png       ← bar chart so sánh
    ├── ablation_posweight.png
    ├── ablation_hidden.png
    ├── ablation_layers.png
    └── ablation.log

  Mỗi config train tối đa 25 epoch (đủ hội tụ cho ablation).
=============================================================
"""

import os
import sys
import time
import logging
import argparse
import importlib.util

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    confusion_matrix,
)

try:
    from tqdm import tqdm
except ImportError:
    print("Cài tqdm: pip install tqdm")
    sys.exit(1)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ══════════════════════════════════════════════════════════════════════
#  LOGGING
# ══════════════════════════════════════════════════════════════════════
def setup_logging(out_dir):
    os.makedirs(out_dir, exist_ok=True)
    log_path = os.path.join(out_dir, "ablation.log")
    fmt = logging.Formatter("%(asctime)s  %(levelname)s  %(message)s",
                            datefmt="%H:%M:%S")
    logger = logging.getLogger("Ablation")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger


# ══════════════════════════════════════════════════════════════════════
#  IMPORT class BiLSTM_ICU từ train_bilstm.py
# ══════════════════════════════════════════════════════════════════════
def import_bilstm_class(py_file):
    spec   = importlib.util.spec_from_file_location("train_bilstm_mod", py_file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.BiLSTM_ICU


# ══════════════════════════════════════════════════════════════════════
#  METRICS
# ══════════════════════════════════════════════════════════════════════
def compute_metrics(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)
    auroc = roc_auc_score(y_true, y_prob)
    auprc = average_precision_score(y_true, y_prob)
    f1    = f1_score(y_true, y_pred, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    ppv  = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    return {"AUROC": auroc, "AUPRC": auprc, "F1": f1,
            "Sensitivity": sens, "Specificity": spec, "PPV": ppv}


# ══════════════════════════════════════════════════════════════════════
#  TRAIN 1 CONFIG  (nhanh: ít epoch, dùng val làm test luôn để so sánh)
# ══════════════════════════════════════════════════════════════════════
def train_one_config(
    BiLSTM_ICU, X_train, y_train, M_train, X_val, y_val, M_val,
    X_test, y_test, M_test, device,
    hidden_size, n_layers, dropout, lr, weight_decay,
    pos_weight_value, batch_size, max_epochs, patience,
    use_mask, log, config_name,
):
    """Train 1 config và return test metrics."""
    # Datasets
    if use_mask and M_train is not None:
        train_ds = TensorDataset(X_train, M_train, y_train)
        val_ds   = TensorDataset(X_val,   M_val,   y_val)
        test_ds  = TensorDataset(X_test,  M_test,  y_test)
    else:
        train_ds = TensorDataset(X_train, y_train)
        val_ds   = TensorDataset(X_val,   y_val)
        test_ds  = TensorDataset(X_test,  y_test)

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True, num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size,
                              shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds, batch_size=batch_size,
                              shuffle=False, num_workers=0)

    # Model
    model = BiLSTM_ICU(
        n_features=X_train.shape[2],
        hidden_size=hidden_size,
        n_layers=n_layers,
        dropout=dropout,
        use_mask=use_mask,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Loss + Optimizer
    pw         = torch.tensor([pos_weight_value]).to(device)
    criterion  = nn.BCEWithLogitsLoss(pos_weight=pw)
    optimizer  = torch.optim.AdamW(model.parameters(), lr=lr,
                                   weight_decay=weight_decay)
    scheduler  = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3
    )

    # Train loop (silent — chỉ track best)
    best_auroc      = 0.0
    best_state      = None
    patience_count  = 0
    epoch_used      = 0

    pbar = tqdm(range(1, max_epochs + 1), desc=f"  {config_name:30s}",
                bar_format="{l_bar}{bar:20}{r_bar}", leave=False)
    for epoch in pbar:
        # Train
        model.train()
        for batch in train_loader:
            if len(batch) == 3:
                xb, mb, yb = [x.to(device) for x in batch]
                logits = model(xb, mb)
            else:
                xb, yb = [x.to(device) for x in batch]
                logits = model(xb)
            loss = criterion(logits, yb)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        # Validate
        model.eval()
        probs, labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                if len(batch) == 3:
                    xb, mb, yb = [x.to(device) for x in batch]
                    logits = model(xb, mb)
                else:
                    xb, yb = [x.to(device) for x in batch]
                    logits = model(xb)
                probs.append(torch.sigmoid(logits).cpu().numpy())
                labels.append(yb.cpu().numpy())
        val_auroc = roc_auc_score(np.concatenate(labels),
                                  np.concatenate(probs))
        scheduler.step(val_auroc)

        if val_auroc > best_auroc:
            best_auroc = val_auroc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= patience:
                epoch_used = epoch
                break
        epoch_used = epoch
        pbar.set_postfix(val_auroc=f"{val_auroc:.4f}")

    # Test với best model
    model.load_state_dict(best_state)
    model.eval()
    probs, labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            if len(batch) == 3:
                xb, mb, yb = [x.to(device) for x in batch]
                logits = model(xb, mb)
            else:
                xb, yb = [x.to(device) for x in batch]
                logits = model(xb)
            probs.append(torch.sigmoid(logits).cpu().numpy())
            labels.append(yb.cpu().numpy())
    metrics = compute_metrics(np.concatenate(labels), np.concatenate(probs))
    metrics["Params"]      = n_params
    metrics["EpochsUsed"]  = epoch_used
    return metrics


# ══════════════════════════════════════════════════════════════════════
#  PLOT BAR CHART CHO TỪNG ABLATION
# ══════════════════════════════════════════════════════════════════════
def plot_ablation_bars(df_subset, factor_col, factor_label, out_path,
                       title):
    """df_subset đã filter chỉ 1 ablation, x = factor, y = AUROC/AUPRC/F1/PPV."""
    metrics = ["AUROC", "AUPRC", "F1", "PPV"]
    colors  = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12"]

    fig, ax = plt.subplots(figsize=(10, 6))
    x       = np.arange(len(df_subset))
    width   = 0.2

    for i, (m, c) in enumerate(zip(metrics, colors)):
        ax.bar(x + i*width - 1.5*width, df_subset[m].values,
               width, label=m, color=c, edgecolor="white", lw=0.5)

    # Value labels
    for i, m in enumerate(metrics):
        for j, v in enumerate(df_subset[m].values):
            ax.text(j + i*width - 1.5*width, v + 0.005,
                    f"{v:.3f}", ha="center", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(df_subset[factor_col].astype(str).values)
    ax.set_xlabel(factor_label, fontsize=12)
    ax.set_ylabel("Metric value", fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.set_ylim(0, 1.0)
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Saved: {out_path}")


# ══════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",  required=True)
    parser.add_argument("--out_dir",   default="ablation")
    parser.add_argument("--bilstm_py", default="train_bilstm.py")
    parser.add_argument("--max_epochs", type=int, default=25,
                        help="Max epochs per config (ablation cần nhanh)")
    parser.add_argument("--patience",   type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=128)

    # Default config "anchor" — khi sweep 1 yếu tố, các yếu tố khác giữ nguyên
    parser.add_argument("--base_hidden",     type=int,   default=128)
    parser.add_argument("--base_layers",     type=int,   default=2)
    parser.add_argument("--base_dropout",    type=float, default=0.5)
    parser.add_argument("--base_lr",         type=float, default=3e-4)
    parser.add_argument("--base_wd",         type=float, default=1e-3)
    parser.add_argument("--base_pos_weight", type=float, default=3.0)
    args = parser.parse_args()

    log = setup_logging(args.out_dir)
    log.info("=" * 60)
    log.info("  ABLATION STUDY — Bi-LSTM ICU Mortality")
    log.info("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"  Device: {device}")
    log.info(f"  Max epochs / config: {args.max_epochs}  "
             f"|  Patience: {args.patience}")
    log.info(f"  Base config: hidden={args.base_hidden}, "
             f"layers={args.base_layers}, dropout={args.base_dropout}, "
             f"lr={args.base_lr}, wd={args.base_wd}, "
             f"pos_weight={args.base_pos_weight}")

    # Load class
    BiLSTM_ICU = import_bilstm_class(args.bilstm_py)

    # Load data once
    log.info(f"  Loading data từ {args.data_dir}")
    X_train = torch.load(os.path.join(args.data_dir, "X_train.pt"), weights_only=True)
    y_train = torch.load(os.path.join(args.data_dir, "y_train.pt"), weights_only=True)
    X_val   = torch.load(os.path.join(args.data_dir, "X_val.pt"),   weights_only=True)
    y_val   = torch.load(os.path.join(args.data_dir, "y_val.pt"),   weights_only=True)
    X_test  = torch.load(os.path.join(args.data_dir, "X_test.pt"),  weights_only=True)
    y_test  = torch.load(os.path.join(args.data_dir, "y_test.pt"),  weights_only=True)
    M_train = torch.load(os.path.join(args.data_dir, "M_train.pt"), weights_only=True)
    M_val   = torch.load(os.path.join(args.data_dir, "M_val.pt"),   weights_only=True)
    M_test  = torch.load(os.path.join(args.data_dir, "M_test.pt"),  weights_only=True)
    log.info(f"  Train: {X_train.shape}, Test: {X_test.shape}")

    # ── Định nghĩa các sweep ──────────────────────────────────────────
    experiments = []

    # 1. Missing mask
    for use_mask in [False, True]:
        experiments.append({
            "Factor":      "Missing Mask",
            "Value":       "ON" if use_mask else "OFF",
            "use_mask":    use_mask,
            "hidden_size": args.base_hidden,
            "n_layers":    args.base_layers,
            "pos_weight":  args.base_pos_weight,
        })

    # 2. pos_weight
    for pw in [1.0, 3.0, 5.0, 8.0]:
        experiments.append({
            "Factor":      "pos_weight",
            "Value":       str(pw),
            "use_mask":    True,
            "hidden_size": args.base_hidden,
            "n_layers":    args.base_layers,
            "pos_weight":  pw,
        })

    # 3. Hidden size
    for h in [64, 128, 256]:
        experiments.append({
            "Factor":      "Hidden Size",
            "Value":       str(h),
            "use_mask":    True,
            "hidden_size": h,
            "n_layers":    args.base_layers,
            "pos_weight":  args.base_pos_weight,
        })

    # 4. n_layers
    for n in [1, 2, 3]:
        experiments.append({
            "Factor":      "n_layers",
            "Value":       str(n),
            "use_mask":    True,
            "hidden_size": args.base_hidden,
            "n_layers":    n,
            "pos_weight":  args.base_pos_weight,
        })

    log.info(f"\n  Tổng số config sẽ chạy: {len(experiments)}\n")

    # ── Chạy từng experiment ──────────────────────────────────────────
    results = []
    total_start = time.time()
    for i, exp in enumerate(experiments, start=1):
        cfg_name = f"{exp['Factor']}={exp['Value']}"
        log.info(f"  [{i}/{len(experiments)}] {cfg_name}")
        t0 = time.time()
        metrics = train_one_config(
            BiLSTM_ICU,
            X_train, y_train, M_train,
            X_val,   y_val,   M_val,
            X_test,  y_test,  M_test,
            device,
            hidden_size      = exp["hidden_size"],
            n_layers         = exp["n_layers"],
            dropout          = args.base_dropout,
            lr               = args.base_lr,
            weight_decay     = args.base_wd,
            pos_weight_value = exp["pos_weight"],
            batch_size       = args.batch_size,
            max_epochs       = args.max_epochs,
            patience         = args.patience,
            use_mask         = exp["use_mask"],
            log              = log,
            config_name      = cfg_name,
        )
        elapsed = time.time() - t0
        log.info(f"    AUROC={metrics['AUROC']:.4f}  "
                 f"AUPRC={metrics['AUPRC']:.4f}  "
                 f"F1={metrics['F1']:.4f}  "
                 f"PPV={metrics['PPV']:.4f}  "
                 f"params={metrics['Params']:,}  "
                 f"ep={metrics['EpochsUsed']}  ({elapsed:.0f}s)")
        results.append({**exp, **metrics, "Time_s": round(elapsed, 1)})

    total_time = time.time() - total_start
    log.info(f"\n  Tổng thời gian: {total_time/60:.1f} phút")

    # ── Save table ────────────────────────────────────────────────────
    df = pd.DataFrame(results)
    df_save = df[["Factor", "Value", "AUROC", "AUPRC", "F1",
                  "Sensitivity", "Specificity", "PPV",
                  "Params", "EpochsUsed", "Time_s"]]
    df_save = df_save.round(4)
    csv_path = os.path.join(args.out_dir, "ablation_results.csv")
    df_save.to_csv(csv_path, index=False)
    log.info(f"  CSV: {csv_path}")

    # ── Markdown table ────────────────────────────────────────────────
    md_path = os.path.join(args.out_dir, "ablation_results.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Ablation Study Results\n\n")
        for factor in df["Factor"].unique():
            sub = df_save[df_save["Factor"] == factor].copy()
            f.write(f"## {factor}\n\n")
            f.write("| " + " | ".join(sub.columns) + " |\n")
            f.write("|" + "---|" * len(sub.columns) + "\n")
            for _, row in sub.iterrows():
                f.write("| " + " | ".join(str(v) for v in row.values) + " |\n")
            f.write("\n")
    log.info(f"  MD : {md_path}")

    # ── Plots ─────────────────────────────────────────────────────────
    plot_specs = [
        ("Missing Mask", "Missing Mask",  "ablation_mask.png",
         "Tác động của Missing Mask"),
        ("pos_weight",   "pos_weight",    "ablation_posweight.png",
         "Tác động của pos_weight"),
        ("Hidden Size",  "Hidden Size",   "ablation_hidden.png",
         "Tác động của Hidden Size"),
        ("n_layers",     "Số LSTM Layers", "ablation_layers.png",
         "Tác động của số LSTM layers"),
    ]
    for factor, xlabel, fname, title in plot_specs:
        sub = df_save[df_save["Factor"] == factor].copy()
        if len(sub) == 0:
            continue
        plot_ablation_bars(sub, "Value", xlabel,
                           os.path.join(args.out_dir, fname), title)

    # ── In bảng tóm tắt cuối ─────────────────────────────────────────
    log.info("\n")
    log.info("=" * 70)
    log.info("  TÓM TẮT KẾT QUẢ ABLATION")
    log.info("=" * 70)
    for factor in df["Factor"].unique():
        sub = df_save[df_save["Factor"] == factor]
        best_idx = sub["AUROC"].idxmax()
        best     = df_save.loc[best_idx]
        log.info(f"  {factor:15s}: best = {best['Value']:8s}  "
                 f"AUROC={best['AUROC']:.4f}  "
                 f"AUPRC={best['AUPRC']:.4f}  "
                 f"F1={best['F1']:.4f}")
    log.info("=" * 70)
    log.info(f"\n  Hoàn thành! Output: {args.out_dir}/\n")


if __name__ == "__main__":
    main()
