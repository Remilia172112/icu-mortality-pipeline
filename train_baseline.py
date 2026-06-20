"""
=============================================================
  Baseline Models — ICU Mortality Prediction
  So sánh với Bi-LSTM (train_bilstm.py)

  Models:
    1. Logistic Regression  (linear baseline)
    2. XGBoost              (tree-based SOTA baseline)

  Input:  cùng .pt files với train_bilstm.py
  Output: results/baseline_results.txt
          results/baseline_comparison.csv
          results/roc_curves.png

Cách chạy:
    python train_baseline.py --data_dir E:\\KLTN\\mimic4_processed

Cài thư viện nếu thiếu:
    pip install xgboost scikit-learn matplotlib
=============================================================
"""

import os
import sys
import time
import logging
import argparse

import numpy as np
import torch

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    f1_score, accuracy_score, confusion_matrix,
    roc_curve,
)
from sklearn.pipeline import Pipeline

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("⚠  XGBoost chưa cài. Chạy: pip install xgboost")
    print("   Sẽ chỉ chạy Logistic Regression.\n")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("⚠  matplotlib chưa cài → bỏ qua vẽ ROC curve.\n")

import pandas as pd


# ══════════════════════════════════════════════════════════════════════
#  LOGGING
# ══════════════════════════════════════════════════════════════════════
def setup_logging(out_dir):
    os.makedirs(out_dir, exist_ok=True)
    log_path = os.path.join(out_dir, "baseline.log")
    fmt = logging.Formatter("%(asctime)s  %(levelname)s  %(message)s",
                            datefmt="%H:%M:%S")
    logger = logging.getLogger("Baseline")
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
#  FEATURE ENGINEERING
#  Bi-LSTM dùng sequence (N, 48, 69) — baseline cần vector 2D (N, F)
#  → flatten + tính thống kê theo thời gian
# ══════════════════════════════════════════════════════════════════════
def extract_features(X, M=None):
    """
    X: numpy (N, T, F)
    M: numpy (N, T, F) missing mask — 1=thật, 0=điền
    Trả về (N, F*6) hoặc (N, F*7 nếu có mask)

    Thống kê mỗi feature theo trục thời gian:
      mean, std, min, max, first, last  [+ missing_rate nếu có mask]
    """
    # (N, T, F) → stats theo axis=1 (time)
    mean_v  = X.mean(axis=1)                          # (N, F)
    std_v   = X.std(axis=1)                           # (N, F)
    min_v   = X.min(axis=1)                           # (N, F)
    max_v   = X.max(axis=1)                           # (N, F)
    first_v = X[:, 0, :]                              # (N, F)  — giờ đầu nhập viện
    last_v  = X[:, -1, :]                             # (N, F)  — giờ cuối (48h)

    parts = [mean_v, std_v, min_v, max_v, first_v, last_v]

    if M is not None:
        # Tỷ lệ missing mỗi feature (0=hoàn toàn missing, 1=đầy đủ)
        missing_rate = 1.0 - M.mean(axis=1)           # (N, F)
        parts.append(missing_rate)

    return np.concatenate(parts, axis=1)              # (N, F*6 or F*7)


# ══════════════════════════════════════════════════════════════════════
#  METRICS  (giống train_bilstm.py để so sánh fair)
# ══════════════════════════════════════════════════════════════════════
def compute_metrics(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)
    auroc  = roc_auc_score(y_true, y_prob)
    auprc  = average_precision_score(y_true, y_prob)
    acc    = accuracy_score(y_true, y_pred)
    f1     = f1_score(y_true, y_pred, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    ppv         = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    npv         = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    return {
        "AUROC": auroc, "AUPRC": auprc, "Accuracy": acc, "F1": f1,
        "Sensitivity": sensitivity, "Specificity": specificity,
        "PPV": ppv, "NPV": npv,
        "TP": int(tp), "FP": int(fp), "TN": int(tn), "FN": int(fn),
    }


def print_metrics(log, name, m):
    log.info(f"\n  ── {name} ──────────────────────────")
    log.info(f"  ┌──────────────┬──────────┐")
    log.info(f"  │ Metric       │ Value    │")
    log.info(f"  ├──────────────┼──────────┤")
    for k in ["AUROC","AUPRC","Accuracy","F1","Sensitivity","Specificity","PPV","NPV"]:
        log.info(f"  │ {k:<12s} │ {m[k]:.4f}   │")
    log.info(f"  └──────────────┴──────────┘")
    log.info(f"  Confusion: TP={m['TP']}  FP={m['FP']}  FN={m['FN']}  TN={m['TN']}")


# ══════════════════════════════════════════════════════════════════════
#  ROC CURVE PLOT
# ══════════════════════════════════════════════════════════════════════
def plot_roc(results, out_path):
    """results: list of (name, y_true, y_prob)"""
    if not HAS_MPL:
        return
    plt.figure(figsize=(7, 6))
    colors = ["#e74c3c", "#3498db", "#2ecc71", "#9b59b6"]
    for i, (name, y_true, y_prob) in enumerate(results):
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc = roc_auc_score(y_true, y_prob)
        plt.plot(fpr, tpr, color=colors[i % len(colors)],
                 lw=2, label=f"{name}  (AUROC={auc:.4f})")
    plt.plot([0,1],[0,1], "k--", lw=1, label="Random")
    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate", fontsize=12)
    plt.title("ROC Curves — ICU Mortality Prediction", fontsize=13)
    plt.legend(fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


# ══════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="Baseline Models — ICU Mortality")
    parser.add_argument("--data_dir",  required=True,
                        help="Thư mục chứa X/y/M .pt files (giống train_bilstm.py)")
    parser.add_argument("--out_dir",   default="results",
                        help="Thư mục lưu kết quả (default: results/)")
    parser.add_argument("--use_mask",  type=int, default=1,
                        help="1=dùng missing mask làm feature, 0=không")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Ngưỡng phân loại (default: 0.5)")
    args = parser.parse_args()

    log = setup_logging(args.out_dir)
    log.info("=" * 60)
    log.info("  Baseline Models — ICU Mortality Prediction")
    log.info("=" * 60)

    # ── Load data ─────────────────────────────────────────────────────
    log.info(f"  Data dir: {args.data_dir}")

    def load_pt(name):
        return torch.load(os.path.join(args.data_dir, name),
                          weights_only=True).numpy()

    X_train = load_pt("X_train.pt")
    y_train = load_pt("y_train.pt").astype(int)
    X_val   = load_pt("X_val.pt")
    y_val   = load_pt("y_val.pt").astype(int)
    X_test  = load_pt("X_test.pt")
    y_test  = load_pt("y_test.pt").astype(int)

    N, T, F = X_train.shape
    log.info(f"  X_train: {X_train.shape}  (N={N}, T={T}, F={F})")
    log.info(f"  Train mortality: {y_train.mean():.1%}")
    log.info(f"  Test  mortality: {y_test.mean():.1%}")

    # Missing mask
    M_train = M_val = M_test = None
    if args.use_mask:
        m_path = os.path.join(args.data_dir, "M_train.pt")
        if os.path.exists(m_path):
            M_train = load_pt("M_train.pt")
            M_val   = load_pt("M_val.pt")
            M_test  = load_pt("M_test.pt")
            log.info("  Missing mask: LOADED → thêm missing_rate features")
        else:
            log.warning("  M_train.pt không tìm thấy → bỏ mask features")

    # ── Feature extraction ────────────────────────────────────────────
    log.info("\n  Extracting temporal features (mean/std/min/max/first/last)...")
    X_tr = extract_features(X_train, M_train)
    X_va = extract_features(X_val,   M_val)
    X_te = extract_features(X_test,  M_test)
    log.info(f"  Feature vector shape: {X_tr.shape}  "
             f"({F} features × {X_tr.shape[1]//F} stats)")

    # Kết hợp train+val để train baseline (không cần val loss)
    X_trainval = np.concatenate([X_tr, X_va], axis=0)
    y_trainval = np.concatenate([y_train, y_val], axis=0)
    log.info(f"  Train+Val: {X_trainval.shape[0]:,} samples\n")

    # Class weight cho imbalance
    n_neg = (y_trainval == 0).sum()
    n_pos = (y_trainval == 1).sum()
    scale = n_neg / n_pos
    log.info(f"  Class imbalance — class_weight scale: {scale:.1f}x\n")

    all_results   = {}   # name → metrics dict
    roc_data      = []   # for ROC plot

    # ══════════════════════════════════════════════════════════════════
    #  MODEL 1: Logistic Regression
    # ══════════════════════════════════════════════════════════════════
    log.info("  [1/2] Logistic Regression ...")
    t0 = time.time()

    lr_model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    LogisticRegression(
            C=0.1,                    # regularization mạnh hơn
            class_weight="balanced",  # xử lý imbalance
            max_iter=1000,
            solver="lbfgs",
            n_jobs=-1,
            random_state=42,
        )),
    ])
    lr_model.fit(X_trainval, y_trainval)
    lr_prob = lr_model.predict_proba(X_te)[:, 1]

    elapsed = time.time() - t0
    m_lr = compute_metrics(y_test, lr_prob, args.threshold)
    print_metrics(log, f"Logistic Regression  ({elapsed:.1f}s)", m_lr)
    all_results["Logistic Regression"] = m_lr
    roc_data.append(("Logistic Regression", y_test, lr_prob))

    # ══════════════════════════════════════════════════════════════════
    #  MODEL 2: XGBoost
    # ══════════════════════════════════════════════════════════════════
    if HAS_XGB:
        log.info("\n  [2/2] XGBoost ...")
        t0 = time.time()

        xgb_model = XGBClassifier(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale,   # xử lý imbalance
            eval_metric="auc",
            early_stopping_rounds=20,
            random_state=42,
            n_jobs=-1,
            verbosity=0,
        )
        xgb_model.fit(
            X_tr, y_train,
            eval_set=[(X_va, y_val)],
            verbose=False,
        )
        xgb_prob = xgb_model.predict_proba(X_te)[:, 1]

        elapsed = time.time() - t0
        m_xgb = compute_metrics(y_test, xgb_prob, args.threshold)
        print_metrics(log, f"XGBoost  ({elapsed:.1f}s, "
                      f"best iter={xgb_model.best_iteration})", m_xgb)
        all_results["XGBoost"] = m_xgb
        roc_data.append(("XGBoost", y_test, xgb_prob))
    else:
        log.warning("  [2/2] XGBoost bị bỏ qua (chưa cài).")

    # ══════════════════════════════════════════════════════════════════
    #  BẢNG SO SÁNH TỔNG HỢP
    # ══════════════════════════════════════════════════════════════════
    # Thêm kết quả Bi-LSTM v4 để so sánh trực tiếp
    bilstm_v4 = {
        "AUROC": 0.9221, "AUPRC": 0.7048, "Accuracy": 0.9088,
        "F1": 0.6207, "Sensitivity": 0.6728, "Specificity": 0.9382,
        "PPV": 0.5761, "NPV": 0.9583,
        "TP": 1275, "FP": 938, "TN": 14244, "FN": 620,
    }
    all_results["Bi-LSTM (v4)"] = bilstm_v4

    metrics_keys = ["AUROC","AUPRC","Accuracy","F1",
                    "Sensitivity","Specificity","PPV","NPV"]

    log.info("\n")
    log.info("=" * 60)
    log.info("  BẢNG SO SÁNH TỔNG HỢP (Test Set)")
    log.info("=" * 60)

    header = f"  {'Metric':<14}" + "".join(
        f"{name:>22}" for name in all_results.keys()
    )
    log.info(header)
    log.info("  " + "-" * (14 + 22 * len(all_results)))

    for k in metrics_keys:
        vals = [all_results[name][k] for name in all_results]
        best = max(vals)
        row  = f"  {k:<14}"
        for name, v in zip(all_results.keys(), vals):
            marker = " ★" if abs(v - best) < 1e-6 else "  "
            row += f"{v:>18.4f}{marker}"
        log.info(row)

    log.info("  " + "-" * (14 + 22 * len(all_results)))
    log.info("  ★ = best cho metric đó\n")

    # ── Lưu CSV ───────────────────────────────────────────────────────
    rows = []
    for model_name, m in all_results.items():
        row = {"Model": model_name}
        row.update({k: round(m[k], 4) for k in metrics_keys})
        rows.append(row)

    df = pd.DataFrame(rows).set_index("Model")
    csv_path = os.path.join(args.out_dir, "baseline_comparison.csv")
    df.to_csv(csv_path)
    log.info(f"  Kết quả lưu: {csv_path}")

    # ── ROC curve ─────────────────────────────────────────────────────
    if HAS_MPL and roc_data:
        roc_path = os.path.join(args.out_dir, "roc_curves.png")
        plot_roc(roc_data, roc_path)
        log.info(f"  ROC curve: {roc_path}")

    log.info("\n  Hoàn thành!")


if __name__ == "__main__":
    main()
