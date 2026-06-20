"""
=============================================================
  Transformer Training — ICU Mortality Prediction
  (Encoder-only architecture với Positional Encoding)
  GPU: RTX 5060 | RAM: 32GB
=============================================================

Cách chạy (Windows):
    python train_transformer.py --data_dir E:\\KLTN\\mimic4_processed

Architecture:
    [Input (B, 48, 69)]
        ↓ [Concat missing mask] → (B, 48, 138)
        ↓ [Linear projection] → (B, 48, d_model)
        ↓ [+ Positional Encoding]
        ↓ [Transformer Encoder × n_layers]
        ↓ [Global pooling: mean of timesteps]
        ↓ [FC → logit]

Output:
    checkpoints_transformer/
    ├── best_model.pt
    ├── training_history.csv
    └── training.log
"""

import os
import sys
import math
import time
import logging
import argparse

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
    logger = logging.getLogger("Transformer")
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
#  POSITIONAL ENCODING
#  Sinusoidal encoding chuẩn từ paper "Attention is All You Need"
#  Giúp Transformer biết được thứ tự thời gian của các timestep
# ══════════════════════════════════════════════════════════════════════
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 100):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # (1, max_len, d_model) — broadcast được với batch
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        # x: (B, T, d_model)
        return x + self.pe[:, :x.size(1), :]


# ══════════════════════════════════════════════════════════════════════
#  MODEL — Transformer Encoder
# ══════════════════════════════════════════════════════════════════════
class Transformer_ICU(nn.Module):
    """
    Transformer Encoder cho ICU Mortality Prediction.

    Khác Bi-LSTM:
      - Bi-LSTM: xử lý tuần tự, RNN gates
      - Transformer: self-attention, có thể "nhìn" mọi timestep cùng lúc

    Args:
        n_features:  số feature gốc (69)
        d_model:     dimension của embedding (chia hết cho n_heads)
        n_heads:     số attention heads
        n_layers:    số transformer encoder layers
        dim_ff:      dimension của feedforward network
        dropout:     dropout rate
        max_len:     max sequence length (48 timestep)
        use_mask:    1 = thêm missing mask làm feature
    """

    def __init__(
        self,
        n_features: int   = 69,
        d_model:    int   = 128,
        n_heads:    int   = 8,
        n_layers:   int   = 4,
        dim_ff:     int   = 256,
        dropout:    float = 0.3,
        max_len:    int   = 48,
        use_mask:   bool  = True,
    ):
        super().__init__()
        self.use_mask  = use_mask
        self.d_model   = d_model
        input_dim      = n_features * 2 if use_mask else n_features

        # 1. Project raw features → d_model dimension
        self.input_proj = nn.Linear(input_dim, d_model)

        # 2. Positional encoding để Transformer biết thứ tự thời gian
        self.pos_encoder = PositionalEncoding(d_model, max_len=max_len + 10)

        # 3. Transformer Encoder stack
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_ff,
            dropout=dropout,
            activation="gelu",        # GELU thường tốt hơn ReLU cho Transformer
            batch_first=True,
            norm_first=True,          # Pre-LN: ổn định hơn cho training
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # 4. Classification head
        self.dropout = nn.Dropout(dropout)
        self.fc      = nn.Linear(d_model, 1)

    def forward(self, x, m=None):
        # x: (B, T, F),  m: (B, T, F)
        if self.use_mask and m is not None:
            x = torch.cat([x, m], dim=-1)         # (B, T, F*2)

        # Project + scale (chuẩn theo paper gốc)
        x = self.input_proj(x) * math.sqrt(self.d_model)  # (B, T, d_model)
        x = self.pos_encoder(x)                            # (B, T, d_model)

        # Transformer encoder
        x = self.transformer(x)                            # (B, T, d_model)

        # Global average pooling theo trục thời gian
        # (alternative: dùng CLS token — nhưng mean pooling đơn giản và effective)
        x = x.mean(dim=1)                                  # (B, d_model)

        out = self.dropout(x)
        out = self.fc(out)                                 # (B, 1)
        return out.squeeze(-1)                             # (B,)


# ══════════════════════════════════════════════════════════════════════
#  METRICS
# ══════════════════════════════════════════════════════════════════════
def compute_metrics(y_true, y_prob, threshold=0.5):
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
        "AUROC": auroc, "AUPRC": auprc, "Accuracy": acc, "F1": f1,
        "Sensitivity": sensitivity, "Specificity": specificity,
        "PPV": ppv, "NPV": npv,
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

    return total_loss / max(n_batches, 1)


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
    parser = argparse.ArgumentParser(description="Transformer ICU Mortality Training")
    parser.add_argument("--data_dir",    required=True, help="Thư mục chứa X/y/M .pt files")
    parser.add_argument("--ckpt_dir",    default="checkpoints_transformer", help="Lưu checkpoints")

    # ── Hyperparameters ──────────────────────────────────────────────
    parser.add_argument("--d_model",     type=int,   default=128,    help="Embedding dim (chia hết cho n_heads)")
    parser.add_argument("--n_heads",     type=int,   default=8,      help="Số attention heads")
    parser.add_argument("--n_layers",    type=int,   default=4,      help="Số transformer layers")
    parser.add_argument("--dim_ff",      type=int,   default=256,    help="Feedforward dimension")
    parser.add_argument("--dropout",     type=float, default=0.3,    help="Dropout rate")
    parser.add_argument("--lr",          type=float, default=1e-4,   help="LR (Transformer nhạy hơn LSTM)")
    parser.add_argument("--weight_decay",type=float, default=1e-3,   help="L2 regularization")
    parser.add_argument("--batch_size",  type=int,   default=128,    help="Batch size")
    parser.add_argument("--epochs",      type=int,   default=50,     help="Max epochs")
    parser.add_argument("--patience",    type=int,   default=15,     help="Early stopping patience")
    parser.add_argument("--warmup_epochs",type=int,  default=3,      help="Số epoch warm-up LR (Transformer cần)")
    parser.add_argument("--use_mask",    type=int,   default=1,      help="1=dùng missing mask, 0=không")
    parser.add_argument("--use_sampler", type=int,   default=0,      help="1=WeightedSampler, 0=không")
    parser.add_argument("--pos_weight",  type=float, default=3.0,    help="pos_weight cho BCELoss. -1 = tự tính")

    args = parser.parse_args()
    log  = setup_logging(args.ckpt_dir)

    log.info("=" * 60)
    log.info("  Transformer ICU Mortality Prediction Training")
    log.info("=" * 60)

    # Sanity check: d_model phải chia hết cho n_heads
    if args.d_model % args.n_heads != 0:
        log.error(f"  d_model ({args.d_model}) phải chia hết cho n_heads ({args.n_heads})!")
        sys.exit(1)

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

    # Missing mask
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
    model = Transformer_ICU(
        n_features=n_features,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        dim_ff=args.dim_ff,
        dropout=args.dropout,
        max_len=seq_len,
        use_mask=use_mask,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(
        f"  Model: Transformer  "
        f"(d_model={args.d_model}, heads={args.n_heads}, "
        f"layers={args.n_layers}, dim_ff={args.dim_ff})"
    )
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
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
        betas=(0.9, 0.98)         # betas chuẩn cho Transformer
    )

    # Warm-up + ReduceLROnPlateau (Transformer rất nhạy với LR đầu)
    def warmup_lambda(epoch):
        if epoch < args.warmup_epochs:
            return (epoch + 1) / args.warmup_epochs
        return 1.0
    warmup_sched = torch.optim.lr_scheduler.LambdaLR(optimizer, warmup_lambda)
    plateau_sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5
    )

    log.info(f"  Optimizer: AdamW (lr={args.lr}, wd={args.weight_decay}, betas=(0.9, 0.98))")
    log.info(f"  Scheduler: Warmup({args.warmup_epochs}) + ReduceLROnPlateau(patience=5)")
    log.info(f"  Batch size: {args.batch_size}")
    log.info(f"  Max epochs: {args.epochs}  |  Early stopping: {args.patience}")
    log.info("")

    # ── Training Loop ─────────────────────────────────────────────────
    best_auroc   = 0.0
    patience_cnt = 0
    history      = []
    total_start  = time.time()

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

        # Update schedulers
        if epoch <= args.warmup_epochs:
            warmup_sched.step()
        else:
            plateau_sched.step(val_auroc)

        elapsed = time.time() - epoch_start

        log.info(
            f"  → Train loss: {train_loss:.4f}  |  "
            f"Val loss: {val_loss:.4f}  |  "
            f"AUROC: {val_auroc:.4f}  |  "
            f"AUPRC: {val_auprc:.4f}  |  "
            f"F1: {val_metrics['F1']:.4f}  |  "
            f"{elapsed:.0f}s"
        )

        # Checkpoint mỗi epoch
        torch.save({
            "epoch":       epoch,
            "model_state": model.state_dict(),
            "optimizer":   optimizer.state_dict(),
            "val_auroc":   val_auroc,
            "val_loss":    val_loss,
            "args":        vars(args),
        }, os.path.join(args.ckpt_dir, f"epoch_{epoch:02d}.pt"))

        # Best model
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

    test_loader  = DataLoader(test_ds, batch_size=args.batch_size,
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
