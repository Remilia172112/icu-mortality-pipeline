"""
==========================================================================
  multi_seed_runs.py
  Thí nghiệm Multi-Seed cho mục 4.10 — KLTN ICU Mortality Prediction
==========================================================================

  Huấn luyện 5 mô hình (LR, XGBoost, LSTM, Bi-LSTM, Transformer) với 5
  random seeds (42, 123, 456, 789, 2026) trên cùng cohort/split để
  đánh giá độ ổn định thống kê.

  Output:
    multi_seed_results/multi_seed_raw.csv     (25 rows: 5 seeds x 5 models)
    multi_seed_results/multi_seed_summary.csv (mean ± std cho mỗi model)
    multi_seed_results/boxplot_auroc.png      (box plot AUROC)

  Ước tính thời gian: 20 - 30 phút trên RTX 5060.

  Cách dùng:
    python multi_seed_runs.py
    python multi_seed_runs.py --epochs 30 --patience 10   # rút gọn
    python multi_seed_runs.py --seeds 42 123              # chỉ 2 seed (test)

  Yêu cầu:
    - Folder dữ liệu đã chuẩn bị từ pipeline tiền xử lý:
        DATA_DIR/X_train.pt, y_train.pt, M_train.pt
        DATA_DIR/X_val.pt,   y_val.pt,   M_val.pt
        DATA_DIR/X_test.pt,  y_test.pt,  M_test.pt
    - Mặc định DATA_DIR = E:/KLTN/mimic4_processed
      (sửa qua biến môi trường DATA_DIR hoặc tham số --data-dir)
"""
import argparse
import logging
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    confusion_matrix,
    accuracy_score,
)
from xgboost import XGBClassifier

# ── Logging ──────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("multi_seed")

# ── Defaults ─────────────────────────────────────────────────────────────
DEFAULT_DATA_DIR = Path(
    os.environ.get("DATA_DIR", r"E:/KLTN/mimic4_processed")
)
OUTPUT_DIR = Path("multi_seed_results")
OUTPUT_DIR.mkdir(exist_ok=True)
DEFAULT_SEEDS = [42, 123, 456, 789, 2024]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ════════════════════════════════════════════════════════════════════════
#   UTILITIES
# ════════════════════════════════════════════════════════════════════════
def set_seed(seed: int):
    """Cố định toàn bộ nguồn ngẫu nhiên."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_metrics(y_true: np.ndarray, y_proba: np.ndarray, threshold: float = 0.5) -> dict:
    y_pred = (y_proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        "AUROC": float(roc_auc_score(y_true, y_proba)),
        "AUPRC": float(average_precision_score(y_true, y_proba)),
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "F1": float(f1_score(y_true, y_pred)),
        "Sensitivity": float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0,
        "Specificity": float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0,
        "PPV": float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0,
        "NPV": float(tn / (tn + fn)) if (tn + fn) > 0 else 0.0,
    }


# ════════════════════════════════════════════════════════════════════════
#   DATA LOADING & FEATURE EXTRACTION
# ════════════════════════════════════════════════════════════════════════
def load_data(data_dir: Path):
    """Load 9 tensor files đã chuẩn hóa từ pipeline tiền xử lý."""
    log.info(f"Loading data tu {data_dir} ...")

    def load(name):
        t = torch.load(data_dir / f"{name}.pt", weights_only=True)
        return t.numpy() if isinstance(t, torch.Tensor) else t

    X_train = load("X_train")
    y_train = load("y_train")
    M_train = load("M_train")
    X_val   = load("X_val")
    y_val   = load("y_val")
    M_val   = load("M_val")
    X_test  = load("X_test")
    y_test  = load("y_test")
    M_test  = load("M_test")

    log.info(
        f"  Train: X={X_train.shape}  |  Val: X={X_val.shape}  |  Test: X={X_test.shape}"
    )
    log.info(f"  Test mortality: {y_test.mean():.1%}")
    return (X_train, y_train, M_train), (X_val, y_val, M_val), (X_test, y_test, M_test)


def extract_features(X: np.ndarray, M: np.ndarray) -> np.ndarray:
    """Trích 7 thống kê × F features → vector (N, F*7)."""
    return np.concatenate(
        [
            X.mean(axis=1),
            X.std(axis=1),
            X.min(axis=1),
            X.max(axis=1),
            X[:, 0, :],
            X[:, -1, :],
            1.0 - M.mean(axis=1),
        ],
        axis=1,
    )


# ════════════════════════════════════════════════════════════════════════
#   BASELINE MODELS
# ════════════════════════════════════════════════════════════════════════
def train_logreg(X_tr, y_tr, X_te, seed=42):
    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    C=0.1,
                    class_weight="balanced",
                    max_iter=1000,
                    solver="lbfgs",
                    random_state=seed,
                ),
            ),
        ]
    )
    model.fit(X_tr, y_tr)
    return model.predict_proba(X_te)[:, 1]


def train_xgb(X_tr, y_tr, X_va, y_va, X_te, seed=42):
    model = XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=8.0,
        eval_metric="auc",
        early_stopping_rounds=20,
        random_state=seed,
        n_jobs=-1,
        verbosity=0,
        tree_method="hist",
    )
    model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    return model.predict_proba(X_te)[:, 1]


# ════════════════════════════════════════════════════════════════════════
#   DEEP LEARNING MODELS
# ════════════════════════════════════════════════════════════════════════
class LSTM_ICU(nn.Module):
    def __init__(self, n_features=69, hidden=128, n_layers=2,
                 dropout=0.5, bidirectional=False):
        super().__init__()
        input_dim = n_features * 2  # concat X + missing mask
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden,
            num_layers=n_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        out_dim = hidden * (2 if bidirectional else 1)
        self.fc = nn.Linear(out_dim, 1)

    def forward(self, x, m):
        x = torch.cat([x, m], dim=-1)
        out, _ = self.lstm(x)
        return self.fc(self.dropout(out[:, -1, :])).squeeze(-1)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1), :]


class Transformer_ICU(nn.Module):
    def __init__(self, n_features=69, d_model=128, n_heads=8,
                 n_layers=4, dim_ff=256, dropout=0.3):
        super().__init__()
        input_dim = n_features * 2
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pe = PositionalEncoding(d_model, 60)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.tr = nn.TransformerEncoder(encoder_layer, num_layers=n_layers, enable_nested_tensor=False)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(d_model, 1)
        self.d_model = d_model

    def forward(self, x, m):
        x = torch.cat([x, m], dim=-1)
        x = self.input_proj(x) * math.sqrt(self.d_model)
        x = self.pe(x)
        x = self.tr(x)
        x = x.mean(dim=1)
        return self.fc(self.dropout(x)).squeeze(-1)


def train_dl_model(
    model: nn.Module,
    train_data, val_data, test_data,
    pos_weight: float = 3.0,
    lr: float = 3e-4,
    weight_decay: float = 1e-3,
    epochs: int = 50,
    patience: int = 15,
    batch_size: int = 128,
    warmup_epochs: int = 0,
):
    Xtr, ytr, Mtr = train_data
    Xva, yva, Mva = val_data
    Xte, yte, Mte = test_data

    def make_loader(X, M, y, shuffle):
        ds = TensorDataset(
            torch.from_numpy(X).float(),
            torch.from_numpy(M).float(),
            torch.from_numpy(y).float(),
        )
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0)

    train_loader = make_loader(Xtr, Mtr, ytr, shuffle=True)
    val_loader   = make_loader(Xva, Mva, yva, shuffle=False)
    test_loader  = make_loader(Xte, Mte, yte, shuffle=False)

    model = model.to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight], device=DEVICE)
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    def warmup_lambda(epoch):
        if warmup_epochs > 0 and epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        return 1.0

    warmup_sched = torch.optim.lr_scheduler.LambdaLR(optimizer, warmup_lambda)
    plateau_sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5
    )

    best_auroc = 0.0
    best_state = None
    no_improve = 0

    for epoch in range(1, epochs + 1):
        # Train
        model.train()
        for X, M, y in train_loader:
            X, M, y = X.to(DEVICE), M.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            logits = model(X, M)
            loss = criterion(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        # Validate
        model.eval()
        probs, ys = [], []
        with torch.no_grad():
            for X, M, y in val_loader:
                X, M = X.to(DEVICE), M.to(DEVICE)
                probs.append(torch.sigmoid(model(X, M)).cpu().numpy())
                ys.append(y.numpy())
        val_probs = np.concatenate(probs)
        val_ys = np.concatenate(ys)
        val_auroc = roc_auc_score(val_ys, val_probs)

        if epoch <= warmup_epochs:
            warmup_sched.step()
        else:
            plateau_sched.step(val_auroc)

        if val_auroc > best_auroc:
            best_auroc = val_auroc
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                log.info(f"     Early stop @ epoch {epoch}, best val AUROC={best_auroc:.4f}")
                break

    # Load best + predict test
    model.load_state_dict(best_state)
    model.eval()
    probs, ys = [], []
    with torch.no_grad():
        for X, M, y in test_loader:
            X, M = X.to(DEVICE), M.to(DEVICE)
            probs.append(torch.sigmoid(model(X, M)).cpu().numpy())
            ys.append(y.numpy())
    return np.concatenate(ys), np.concatenate(probs)


# ════════════════════════════════════════════════════════════════════════
#   PLOTTING
# ════════════════════════════════════════════════════════════════════════
def make_boxplot(df: pd.DataFrame, output_path: Path):
    """Tao box plot AUROC theo mo hinh."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        log.warning("  matplotlib chua cai - bo qua boxplot")
        return

    models_order = ["Logistic Regression", "XGBoost", "LSTM", "Bi-LSTM", "Transformer"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Box plot AUROC
    data_auroc = [df[df["model"] == m]["AUROC"].values for m in models_order]
    axes[0].boxplot(data_auroc, tick_labels=models_order, showmeans=True)
    axes[0].set_ylabel("AUROC")
    axes[0].set_title("Phân bố AUROC qua 5 seeds")
    axes[0].grid(True, alpha=0.3)
    axes[0].tick_params(axis="x", rotation=15)

    # Box plot F1
    data_f1 = [df[df["model"] == m]["F1"].values for m in models_order]
    axes[1].boxplot(data_f1, tick_labels=models_order, showmeans=True)
    axes[1].set_ylabel("F1-score")
    axes[1].set_title("Phân bố F1-score qua 5 seeds")
    axes[1].grid(True, alpha=0.3)
    axes[1].tick_params(axis="x", rotation=15)

    plt.tight_layout()
    plt.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close()
    log.info(f"  Boxplot: {output_path}")


# ════════════════════════════════════════════════════════════════════════
#   MAIN
# ════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="Multi-Seed Experiment")
    parser.add_argument("--data-dir", type=str, default=str(DEFAULT_DATA_DIR),
                        help="Folder chua X_train.pt, y_train.pt, ...")
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS,
                        help="Random seeds (default: 42 123 456 789 2024)")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Max epochs cho DL models")
    parser.add_argument("--patience", type=int, default=15,
                        help="Early stopping patience")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--skip-models", type=str, nargs="*", default=[],
                        choices=["lr", "xgb", "lstm", "bilstm", "transformer"],
                        help="Bo qua mot so mo hinh")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        log.error(f"DATA_DIR khong ton tai: {data_dir}")
        sys.exit(1)

    log.info("=" * 60)
    log.info("  MULTI-SEED EXPERIMENT - ICU Mortality Prediction")
    log.info("=" * 60)
    log.info(f"  Device:    {DEVICE}")
    log.info(f"  Data dir:  {data_dir}")
    log.info(f"  Seeds:     {args.seeds}")
    log.info(f"  Epochs:    {args.epochs} (patience={args.patience})")
    log.info(f"  Output:    {OUTPUT_DIR.absolute()}")
    log.info("=" * 60)

    # Load data 1 lan
    train_data, val_data, test_data = load_data(data_dir)
    X_tr, y_tr, M_tr = train_data
    X_va, y_va, M_va = val_data
    X_te, y_te, M_te = test_data

    log.info("Feature engineering cho baselines...")
    X_tr_feat = extract_features(X_tr, M_tr)
    X_va_feat = extract_features(X_va, M_va)
    X_te_feat = extract_features(X_te, M_te)
    log.info(f"  Feature vector: {X_tr_feat.shape[1]} chieu")

    all_results = []
    t_total = time.time()

    for i_seed, seed in enumerate(args.seeds, start=1):
        log.info("")
        log.info("=" * 60)
        log.info(f"  SEED {i_seed}/{len(args.seeds)}  =  {seed}")
        log.info("=" * 60)

        # ── 1. Logistic Regression ──────────────────────────────────
        if "lr" not in args.skip_models:
            set_seed(seed)
            t0 = time.time()
            proba = train_logreg(X_tr_feat, y_tr, X_te_feat, seed=seed)
            m = compute_metrics(y_te, proba)
            log.info(
                f"  [1/5] LR          AUROC={m['AUROC']:.4f}  "
                f"F1={m['F1']:.4f}  PPV={m['PPV']:.4f}  ({time.time()-t0:.1f}s)"
            )
            all_results.append({"seed": seed, "model": "Logistic Regression", **m})

        # ── 2. XGBoost ──────────────────────────────────────────────
        if "xgb" not in args.skip_models:
            set_seed(seed)
            t0 = time.time()
            proba = train_xgb(X_tr_feat, y_tr, X_va_feat, y_va, X_te_feat, seed=seed)
            m = compute_metrics(y_te, proba)
            log.info(
                f"  [2/5] XGBoost     AUROC={m['AUROC']:.4f}  "
                f"F1={m['F1']:.4f}  PPV={m['PPV']:.4f}  ({time.time()-t0:.1f}s)"
            )
            all_results.append({"seed": seed, "model": "XGBoost", **m})

        # ── 3. LSTM (2 layers, unidirectional) ──────────────────────
        if "lstm" not in args.skip_models:
            set_seed(seed)
            t0 = time.time()
            model = LSTM_ICU(hidden=128, n_layers=2, dropout=0.5, bidirectional=False)
            y_true, y_proba = train_dl_model(
                model, train_data, val_data, test_data,
                pos_weight=3.0, lr=3e-4,
                epochs=args.epochs, patience=args.patience,
                batch_size=args.batch_size,
            )
            m = compute_metrics(y_true, y_proba)
            log.info(
                f"  [3/5] LSTM        AUROC={m['AUROC']:.4f}  "
                f"F1={m['F1']:.4f}  PPV={m['PPV']:.4f}  ({time.time()-t0:.1f}s)"
            )
            all_results.append({"seed": seed, "model": "LSTM", **m})

        # ── 4. Bi-LSTM (1 layer, bidirectional — optimal config) ────
        if "bilstm" not in args.skip_models:
            set_seed(seed)
            t0 = time.time()
            model = LSTM_ICU(hidden=128, n_layers=1, dropout=0.5, bidirectional=True)
            y_true, y_proba = train_dl_model(
                model, train_data, val_data, test_data,
                pos_weight=3.0, lr=3e-4,
                epochs=args.epochs, patience=args.patience,
                batch_size=args.batch_size,
            )
            m = compute_metrics(y_true, y_proba)
            log.info(
                f"  [4/5] Bi-LSTM     AUROC={m['AUROC']:.4f}  "
                f"F1={m['F1']:.4f}  PPV={m['PPV']:.4f}  ({time.time()-t0:.1f}s)"
            )
            all_results.append({"seed": seed, "model": "Bi-LSTM", **m})

        # ── 5. Transformer ──────────────────────────────────────────
        if "transformer" not in args.skip_models:
            set_seed(seed)
            t0 = time.time()
            model = Transformer_ICU(
                d_model=128, n_heads=8, n_layers=4, dim_ff=256, dropout=0.3
            )
            y_true, y_proba = train_dl_model(
                model, train_data, val_data, test_data,
                pos_weight=3.0, lr=1e-4, warmup_epochs=3,
                epochs=args.epochs, patience=args.patience,
                batch_size=args.batch_size,
            )
            m = compute_metrics(y_true, y_proba)
            log.info(
                f"  [5/5] Transformer AUROC={m['AUROC']:.4f}  "
                f"F1={m['F1']:.4f}  PPV={m['PPV']:.4f}  ({time.time()-t0:.1f}s)"
            )
            all_results.append({"seed": seed, "model": "Transformer", **m})

        # Save sau moi seed (de phong crash)
        df = pd.DataFrame(all_results)
        df.to_csv(OUTPUT_DIR / "multi_seed_raw.csv", index=False)

    # ── Tong hop cuoi cung ──────────────────────────────────────────
    df = pd.DataFrame(all_results)
    df.to_csv(OUTPUT_DIR / "multi_seed_raw.csv", index=False)

    # Mean ± std
    summary = df.groupby("model").agg(["mean", "std"]).round(4)
    summary.to_csv(OUTPUT_DIR / "multi_seed_summary.csv")

    # 95% CI
    n = df["seed"].nunique()
    ci_factor = 1.96 / np.sqrt(n)
    ci_data = []
    for model in df["model"].unique():
        sub = df[df["model"] == model]
        for metric in ["AUROC", "AUPRC", "F1", "PPV", "Sensitivity"]:
            mean = sub[metric].mean()
            std = sub[metric].std()
            ci_data.append({
                "Model": model,
                "Metric": metric,
                "Mean": round(mean, 4),
                "Std": round(std, 4),
                "CI_low": round(mean - ci_factor * std, 4),
                "CI_high": round(mean + ci_factor * std, 4),
            })
    pd.DataFrame(ci_data).to_csv(OUTPUT_DIR / "multi_seed_ci95.csv", index=False)

    # Boxplot
    make_boxplot(df, OUTPUT_DIR / "boxplot_auroc.png")

    total_time = time.time() - t_total
    log.info("")
    log.info("=" * 60)
    log.info(f"  HOAN THANH! Tong thoi gian: {total_time/60:.1f} phut")
    log.info("=" * 60)
    log.info(f"  Raw data:  {OUTPUT_DIR}/multi_seed_raw.csv")
    log.info(f"  Summary:   {OUTPUT_DIR}/multi_seed_summary.csv")
    log.info(f"  95% CI:    {OUTPUT_DIR}/multi_seed_ci95.csv")
    log.info(f"  Boxplot:   {OUTPUT_DIR}/boxplot_auroc.png")
    log.info("")
    log.info("  Tom tat:")
    print("\n", summary.to_string())


if __name__ == "__main__":
    main()