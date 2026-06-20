"""
=============================================================
  LSTM Training — ICU Mortality Prediction
  (Unidirectional — đối xứng với Bi-LSTM để so sánh)
  GPU: RTX 5060 | RAM: 32GB
=============================================================

Cách chạy (Windows):
    python train_lstm.py --data_dir E:\\KLTN\\mimic4_processed

Output:
    checkpoints_lstm/
    ├── epoch_XX.pt          ← checkpoint mỗi epoch
    ├── best_model.pt        ← model AUROC cao nhất
    ├── training_history.csv
    └── training.log
"""

import os
import sys
import json
import time
import logging
import argparse
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    f1_score, accuracy_score, confusion_matrix,
)

try:
    from tqdm import tqdm
except ImportError:
    print("Cài tqdm: pip install tqdm")
    sys.exit(1)


# ══════════════════════════════════════════════════════════════════════
#  LOGGING
# ══════════════════════════════════════════════════════════════════════
def setup_logging(log_dir):
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "training.log")

    fmt = logging.Formatter(
        "%(asctime)s  %(levelname)s  %(message)s", datefmt="%H:%M:%S"
    )
    logger = logging.getLogger("LSTM")
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
#  MODEL — Unidirectional LSTM with Missing Mask
# ══════════════════════════════════════════════════════════════════════
class LSTM_ICU(nn.Module):
    """
    Unidirectional LSTM cho ICU Mortality Prediction.

    Khác Bi-LSTM: chỉ đọc chuỗi theo 1 chiều (quá khứ → hiện tại).
    Dùng để so sánh xem bidirectional có lợi ích thực sự hay không.

    Input:
        x: (batch, seq_len, n_features)
        m: (batch, seq_len, n_features) optional — missing mask

    Architecture:
        [Input] → [LSTM × n_layers] → [Dropout] → [FC → logit]
    """

    def __init__(
        self,
        n_features:  int   = 69,
        hidden_size: int   = 128,
        n_layers:    int   = 2,
        dropout:     float = 0.3,
        use_mask:    bool  = True,
    ):
        super().__init__()
        self.use_mask    = use_mask
        self.n_features  = n_features
        self.hidden_size = hidden_size

        input_dim = n_features * 2 if use_mask else n_features

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
            bidirectional=False,                 # ← khác Bi-LSTM
            dropout=dropout if n_layers > 1 else 0,
        )
        self.dropout = nn.Dropout(dropout)
        # Unidirectional → hidden_size (không nhân 2)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x, m=None):
        # x: (B, T, F),  m: (B, T, F)
        if self.use_mask and m is not None:
            x = torch.cat([x, m], dim=-1)        # (B, T, F*2)

        # LSTM output: (B, T, hidden)
        lstm_out, _ = self.lstm(x)

        # Lấy output ở time step cuối
        last_output = lstm_out[:, -1, :]         # (B, hidden)

        out = self.dropout(last_output)
        out = self.fc(out)                       # (B, 1)
        return out.squeeze(-1)                   # (B,)


# ══════════════════════════════════════════════════════════════════════
#  METRICS
# ══════════════════════════════════════════════════════════════════════
def compute_metrics(y_true, y_prob, threshold=0.5):
    """Tính toàn bộ metrics cần cho báo cáo."""
    y_pred = (y_prob >= threshold).astype(int)

    auroc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.0
    auprc = average_precision_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.0

    acc  = accuracy_score(y_true, y_pred)
    f1   = f1_score(y_true, y_pred, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    ppv         = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    npv         = tn / (tn + fn) if (tn + fn) > 0 else 0.0

    return {
        "AUROC":       auroc,
        "AUPRC":       auprc,
        "Accuracy":    acc,
        "F1":          f1,
        "Sensitivity": sensitivity,
        "Specificity": specificity,
        "PPV":         ppv,
        "NPV":         npv,
        "TP": tp, "FP": fp, "TN": tn, "FN": fn,
    }


# ══════════════════════════════════════════════════════════════════════
#  TRAINING LOOP
# ══════════════════════════════════════════════════════════════════════
def train_one_epoch(model, loader, criterion, optimizer, device, epoch, log):
    model.train()
    total_loss = 0.0
    n_batches  = 0

    pbar = tqdm(
        loader,
        desc=f"  Train E{epoch:02d}",
        bar_format="{l_bar}{bar:30}{r_bar}",
        leave=True,
    )
    for batch in pbar:
        if len(batch) == 3:
            xb, mb, yb = batch
            xb, mb, yb = xb.to(device), mb.to(device), yb.to(device)
            logits = model(xb, mb)
        else:
            xb, yb = batch
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)

        loss = criterion(logits, yb)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        n_batches  += 1
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    avg_loss = total_loss / max(n_batches, 1)
    return avg_loss


@torch.no_grad()
def evaluate(model, loader, criterion, device, split_name="Val"):
    model.eval()
    total_loss = 0.0
    n_batches  = 0
    all_probs  = []
    all_labels = []

    pbar = tqdm(
        loader,
        desc=f"  {split_name:5s}     ",
        bar_format="{l_bar}{bar:30}{r_bar}",
        leave=True,
    )
    for batch in pbar:
        if len(batch) == 3:
            xb, mb, yb = batch
            xb, mb, yb = xb.to(device), mb.to(device), yb.to(device)
            logits = model(xb, mb)
        else:
            xb, yb = batch
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)

        loss = criterion(logits, yb)
        total_loss += loss.item()
        n_batches  += 1

        probs = torch.sigmoid(logits).cpu().numpy()
        all_probs.append(probs)
        all_labels.append(yb.cpu().numpy())

    avg_loss   = total_loss / max(n_batches, 1)
    all_probs  = np.concatenate(all_probs)
    all_labels = np.concatenate(all_labels)
    metrics    = compute_metrics(all_labels, all_probs)
    metrics["loss"] = avg_loss

    return metrics


# ══════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="LSTM ICU Mortality Training")
    parser.add_argument("--data_dir",    required=True, help="Thư mục chứa X/y/M .pt files")
    parser.add_argument("--ckpt_dir",    default="checkpoints_lstm", help="Lưu checkpoints")

    # ── Hyperparameters ──────────────────────────────────────────────
    parser.add_argument("--hidden_size", type=int,   default=128,   help="LSTM hidden size")
    parser.add_argument("--n_layers",    type=int,   default=2,     help="Số lớp LSTM")
    parser.add_argument("--dropout",     type=float, default=0.5,   help="Dropout rate")
    parser.add_argument("--lr",          type=float, default=3e-4,  help="Learning rate")
    parser.add_argument("--weight_decay",type=float, default=1e-3,  help="L2 regularization")
    parser.add_argument("--batch_size",  type=int,   default=128,   help="Batch size")
    parser.add_argument("--epochs",      type=int,   default=50,    help="Max epochs")
    parser.add_argument("--patience",    type=int,   default=15,    help="Early stopping patience")
    parser.add_argument("--use_mask",    type=int,   default=1,     help="1=dùng missing mask, 0=không")
    parser.add_argument("--use_sampler", type=int,   default=0,     help="1=WeightedSampler, 0=không")
    parser.add_argument("--pos_weight",  type=float, default=3.0,   help="pos_weight cho BCELoss. -1 = tự tính từ data")

    args = parser.parse_args()
    log  = setup_logging(args.ckpt_dir)

    log.info("=" * 60)
    log.info("  LSTM ICU Mortality Prediction Training")
    log.info("=" * 60)

    # ── Device ───────────────────────────────────────────────────────
    if torch.cuda.is_available():
        device = torch.device("cuda")
        if torch.cuda.device_count() > 1:
            best_gpu = max(range(torch.cuda.device_count()),
                          key=lambda i: torch.cuda.get_device_properties(i).total_memory)
            device = torch.device(f"cuda:{best_gpu}")
            torch.cuda.set_device(best_gpu)
            log.info(f"  Hybrid mode: chọn GPU {best_gpu}")
        gpu_name = torch.cuda.get_device_name(device)
        gpu_mem  = torch.cuda.get_device_properties(device).total_memory / 1e9
        log.info(f"  GPU: {gpu_name} ({gpu_mem:.1f} GB)")
    else:
        device = torch.device("cpu")
        log.info("  CPU mode (không có GPU)")
    log.info(f"  Device: {device}")

    # ── Load data ─────────────────────────────────────────────────────
    log.info(f"  Data dir: {args.data_dir}")
    X_train = torch.load(os.path.join(args.data_dir, "X_train.pt"), weights_only=True)
    y_train = torch.load(os.path.join(args.data_dir, "y_train.pt"), weights_only=True)
    X_val   = torch.load(os.path.join(args.data_dir, "X_val.pt"),   weights_only=True)
    y_val   = torch.load(os.path.join(args.data_dir, "y_val.pt"),   weights_only=True)

    n_features = X_train.shape[2]
    seq_len    = X_train.shape[1]
    log.info(f"  X_train: {tuple(X_train.shape)}  (N, {seq_len}, {n_features})")
    log.info(f"  Train mortality: {y_train.mean():.1%}")
    log.info(f"  Val   mortality: {y_val.mean():.1%}")

    # Missing mask (optional)
    use_mask = bool(args.use_mask)
    M_train, M_val = None, None
    if use_mask:
        m_path = os.path.join(args.data_dir, "M_train.pt")
        if os.path.exists(m_path):
            M_train = torch.load(m_path, weights_only=True)
            M_val   = torch.load(os.path.join(args.data_dir, "M_val.pt"), weights_only=True)
            log.info(f"  Missing mask: LOADED (input dim = {n_features} * 2 = {n_features*2})")
        else:
            log.warning(f"  M_train.pt không tìm thấy → tắt masking")
            use_mask = False

    # ── DataLoaders ───────────────────────────────────────────────────
    if use_mask and M_train is not None:
        train_ds = TensorDataset(X_train, M_train, y_train)
        val_ds   = TensorDataset(X_val,   M_val,   y_val)
    else:
        train_ds = TensorDataset(X_train, y_train)
        val_ds   = TensorDataset(X_val,   y_val)

    # Class imbalance handling
    if args.use_sampler:
        counts    = torch.bincount(y_train.long())
        weights   = 1.0 / counts.float()
        sample_w  = weights[y_train.long()]
        sampler   = WeightedRandomSampler(sample_w, len(sample_w), replacement=True)
        log.info(f"  Imbalance: WeightedRandomSampler  "
                 f"(class 0: {counts[0]:,}, class 1: {counts[1]:,})")
        train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                                  sampler=sampler, num_workers=0, pin_memory=True)
    else:
        train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                                  shuffle=True, num_workers=0, pin_memory=True)

    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                            shuffle=False, num_workers=0, pin_memory=True)

    # ── Model ─────────────────────────────────────────────────────────
    model = LSTM_ICU(
        n_features=n_features,
        hidden_size=args.hidden_size,
        n_layers=args.n_layers,
        dropout=args.dropout,
        use_mask=use_mask,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"  Model: LSTM unidirectional (hidden={args.hidden_size}, layers={args.n_layers})")
    log.info(f"  Trainable parameters: {n_params:,}")

    # ── Loss, Optimizer, Scheduler ────────────────────────────────────
    n_neg = (y_train == 0).sum().float()
    n_pos = (y_train == 1).sum().float()
    if args.pos_weight > 0:
        pw = args.pos_weight
    else:
        pw = (n_neg / n_pos).item()
    pos_weight = torch.tensor([pw]).to(device)
    criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    log.info(f"  pos_weight: {pw:.2f}  (sai nhóm tử vong bị phạt gấp {pw:.1f} lần)")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5
    )

    log.info(f"  Optimizer: AdamW (lr={args.lr}, wd={args.weight_decay})")
    log.info(f"  Scheduler: ReduceLROnPlateau (patience=5, factor=0.5)")
    log.info(f"  Batch size: {args.batch_size}")
    log.info(f"  Max epochs: {args.epochs}  |  Early stopping: {args.patience}")
    log.info("")

    # ── Training Loop ─────────────────────────────────────────────────
    best_auroc    = 0.0
    patience_cnt  = 0
    history       = []
    total_start   = time.time()

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        lr_now      = optimizer.param_groups[0]["lr"]

        log.info(f"Epoch {epoch:02d}/{args.epochs}  (lr={lr_now:.6f})")

        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch, log
        )

        val_metrics = evaluate(model, val_loader, criterion, device, "Val")
        val_auroc   = val_metrics["AUROC"]
        val_auprc   = val_metrics["AUPRC"]
        val_loss    = val_metrics["loss"]

        scheduler.step(val_auroc)
        elapsed = time.time() - epoch_start

        log.info(
            f"  → Train loss: {train_loss:.4f}  |  "
            f"Val loss: {val_loss:.4f}  |  "
            f"AUROC: {val_auroc:.4f}  |  "
            f"AUPRC: {val_auprc:.4f}  |  "
            f"F1: {val_metrics['F1']:.4f}  |  "
            f"{elapsed:.0f}s"
        )

        ckpt_path = os.path.join(args.ckpt_dir, f"epoch_{epoch:02d}.pt")
        torch.save({
            "epoch":       epoch,
            "model_state": model.state_dict(),
            "optimizer":   optimizer.state_dict(),
            "val_auroc":   val_auroc,
            "val_loss":    val_loss,
            "args":        vars(args),
        }, ckpt_path)
        log.debug(f"  Checkpoint: {ckpt_path}")

        if val_auroc > best_auroc:
            best_auroc   = val_auroc
            patience_cnt = 0
            best_path    = os.path.join(args.ckpt_dir, "best_model.pt")
            torch.save({
                "epoch":       epoch,
                "model_state": model.state_dict(),
                "val_auroc":   val_auroc,
                "val_metrics": val_metrics,
                "args":        vars(args),
            }, best_path)
            log.info(f"  ★ New best AUROC: {val_auroc:.4f} → {best_path}")
        else:
            patience_cnt += 1
            log.info(f"  No improvement ({patience_cnt}/{args.patience})")

        history.append({
            "epoch":      epoch,
            "train_loss": train_loss,
            "val_loss":   val_loss,
            "val_auroc":  val_auroc,
            "val_auprc":  val_auprc,
            "val_f1":     val_metrics["F1"],
            "lr":         lr_now,
        })

        if patience_cnt >= args.patience:
            log.info(f"\n  Early stopping tại epoch {epoch}")
            break

        log.info("")

    total_time = time.time() - total_start
    h, rem = divmod(int(total_time), 3600)
    m, s   = divmod(rem, 60)

    # ── Final Evaluation on Test ──────────────────────────────────────
    log.info("=" * 60)
    log.info("ĐÁNH GIÁ TRÊN TẬP TEST")
    log.info("=" * 60)

    best_ckpt = torch.load(
        os.path.join(args.ckpt_dir, "best_model.pt"), weights_only=False
    )
    model.load_state_dict(best_ckpt["model_state"])

    X_test = torch.load(os.path.join(args.data_dir, "X_test.pt"), weights_only=True)
    y_test = torch.load(os.path.join(args.data_dir, "y_test.pt"), weights_only=True)
    if use_mask:
        M_test  = torch.load(os.path.join(args.data_dir, "M_test.pt"), weights_only=True)
        test_ds = TensorDataset(X_test, M_test, y_test)
    else:
        test_ds = TensorDataset(X_test, y_test)

    test_loader = DataLoader(test_ds, batch_size=args.batch_size,
                             shuffle=False, num_workers=0)
    test_metrics = evaluate(model, test_loader, criterion, device, "Test")

    log.info(f"  Best epoch: {best_ckpt['epoch']}")
    log.info("")
    log.info("  ┌──────────────┬──────────┐")
    log.info("  │ Metric       │ Value    │")
    log.info("  ├──────────────┼──────────┤")
    for k in ["AUROC", "AUPRC", "Accuracy", "F1",
              "Sensitivity", "Specificity", "PPV", "NPV"]:
        log.info(f"  │ {k:<12s} │ {test_metrics[k]:.4f}   │")
    log.info("  └──────────────┴──────────┘")
    log.info("")
    log.info(f"  Confusion Matrix:")
    log.info(f"    TP={test_metrics['TP']}  FP={test_metrics['FP']}")
    log.info(f"    FN={test_metrics['FN']}  TN={test_metrics['TN']}")
    log.info("")
    log.info(f"  Thời gian training: {h}h {m}m {s}s")
    log.info(f"  Checkpoints: {args.ckpt_dir}/")

    import pandas as pd
    pd.DataFrame(history).to_csv(
        os.path.join(args.ckpt_dir, "training_history.csv"), index=False
    )
    log.info(f"  History: {args.ckpt_dir}/training_history.csv")


if __name__ == "__main__":
    main()
