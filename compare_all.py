"""
=============================================================
  Compare All Models — ICU Mortality Prediction
  So sánh 5 model trên cùng tập Test:
    1. Logistic Regression  (baseline)
    2. XGBoost              (baseline)
    3. LSTM                 (unidirectional)
    4. Bi-LSTM              (bidirectional)
    5. Transformer          (encoder-only + PE)

  Output (results_compare/):
    ├── comparison_table.csv      ← bảng metrics chi tiết
    ├── comparison_table.md       ← bảng markdown cho báo cáo
    ├── roc_curves.png            ← 5 đường ROC trên cùng plot
    ├── pr_curves.png             ← 5 đường Precision-Recall
    └── compare.log
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

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    f1_score, accuracy_score, confusion_matrix,
    roc_curve, precision_recall_curve,
)

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

try:
    from tqdm import tqdm
except ImportError:
    print("Cài tqdm: pip install tqdm")
    sys.exit(1)


# ══════════════════════════════════════════════════════════════════════
#  LOGGING
# ══════════════════════════════════════════════════════════════════════
def setup_logging(out_dir):
    os.makedirs(out_dir, exist_ok=True)
    log_path = os.path.join(out_dir, "compare.log")
    fmt = logging.Formatter("%(asctime)s  %(levelname)s  %(message)s",
                            datefmt="%H:%M:%S")
    logger = logging.getLogger("Compare")
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
#  METRICS
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


# ══════════════════════════════════════════════════════════════════════
#  FEATURE ENGINEERING (cho baseline)
# ══════════════════════════════════════════════════════════════════════
def extract_features(X, M=None):
    """X: (N,T,F) → (N, F*6 hoặc F*7) bằng cách aggregate theo thời gian."""
    parts = [
        X.mean(axis=1), X.std(axis=1),
        X.min(axis=1),  X.max(axis=1),
        X[:, 0, :],     X[:, -1, :],
    ]
    if M is not None:
        parts.append(1.0 - M.mean(axis=1))
    return np.concatenate(parts, axis=1)


# ══════════════════════════════════════════════════════════════════════
#  DYNAMIC IMPORT cho các module model
#  → load class LSTM_ICU, BiLSTM_ICU, Transformer_ICU từ file .py
# ══════════════════════════════════════════════════════════════════════
def import_module_from_file(name, path):
    spec   = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ══════════════════════════════════════════════════════════════════════
#  PREDICT bằng PyTorch model (load từ best_model.pt)
# ══════════════════════════════════════════════════════════════════════
@torch.no_grad()
def predict_pytorch(model, X_test, M_test, y_test, device, use_mask,
                    batch_size=128, desc="Predict"):
    model.eval()
    if use_mask and M_test is not None:
        ds = TensorDataset(X_test, M_test, y_test)
    else:
        ds = TensorDataset(X_test, y_test)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)

    all_probs, all_labels = [], []
    pbar = tqdm(loader, desc=f"  {desc:14s}",
                bar_format="{l_bar}{bar:30}{r_bar}", leave=False)
    for batch in pbar:
        if len(batch) == 3:
            xb, mb, yb = batch
            xb, mb, yb = xb.to(device), mb.to(device), yb.to(device)
            logits = model(xb, mb)
        else:
            xb, yb = batch
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
        probs = torch.sigmoid(logits).cpu().numpy()
        all_probs.append(probs)
        all_labels.append(yb.cpu().numpy())
    return np.concatenate(all_probs), np.concatenate(all_labels)


# ══════════════════════════════════════════════════════════════════════
#  LOAD + PREDICT từng PyTorch model
# ══════════════════════════════════════════════════════════════════════
def run_pytorch_model(name, py_file, class_name,
                      ckpt_path, X_test, M_test, y_test, device, log):
    """Load class + checkpoint, predict trên test."""
    log.info(f"  → Loading {name} từ {ckpt_path}")
    if not os.path.exists(ckpt_path):
        log.warning(f"  [!] {ckpt_path} không tồn tại — bỏ qua {name}")
        return None

    # Import module chứa class model
    module = import_module_from_file(f"mod_{name.lower()}", py_file)
    ModelCls = getattr(module, class_name)

    # Load checkpoint
    ckpt = torch.load(ckpt_path, weights_only=False, map_location=device)
    args = ckpt.get("args", {})
    log.debug(f"    args saved: {args}")

    # Tái tạo model với args gốc
    if class_name == "BiLSTM_ICU" or class_name == "LSTM_ICU":
        model = ModelCls(
            n_features  = X_test.shape[2],
            hidden_size = args.get("hidden_size", 128),
            n_layers    = args.get("n_layers", 2),
            dropout     = args.get("dropout", 0.3),
            use_mask    = bool(args.get("use_mask", 1)),
        )
    elif class_name == "Transformer_ICU":
        model = ModelCls(
            n_features = X_test.shape[2],
            d_model    = args.get("d_model", 128),
            n_heads    = args.get("n_heads", 8),
            n_layers   = args.get("n_layers", 4),
            dim_ff     = args.get("dim_ff", 256),
            dropout    = args.get("dropout", 0.3),
            max_len    = X_test.shape[1],
            use_mask   = bool(args.get("use_mask", 1)),
        )
    else:
        raise ValueError(f"Unknown class: {class_name}")

    model.load_state_dict(ckpt["model_state"])
    model.to(device)

    use_mask = bool(args.get("use_mask", 1))
    probs, labels = predict_pytorch(
        model, X_test, M_test if use_mask else None,
        y_test, device, use_mask, desc=name
    )

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "name":     name,
        "probs":    probs,
        "labels":   labels,
        "params":   n_params,
        "best_ep":  ckpt.get("epoch", -1),
    }


# ══════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="So sánh 5 model trên test set")
    parser.add_argument("--data_dir", required=True,
                        help="Thư mục chứa X/y/M .pt files")
    parser.add_argument("--out_dir",  default="results_compare",
                        help="Thư mục lưu kết quả")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Ngưỡng phân loại")

    # Đường dẫn checkpoints — chỉnh nếu tên thư mục khác
    parser.add_argument("--bilstm_ckpt",
                        default="checkpoints_v4/best_model.pt")
    parser.add_argument("--lstm_ckpt",
                        default="checkpoints_lstm/best_model.pt")
    parser.add_argument("--transformer_ckpt",
                        default="checkpoints_transformer/best_model.pt")

    # Đường dẫn file .py chứa class model
    parser.add_argument("--bilstm_py",      default="train_bilstm.py")
    parser.add_argument("--lstm_py",        default="train_lstm.py")
    parser.add_argument("--transformer_py", default="train_transformer.py")

    args = parser.parse_args()
    log  = setup_logging(args.out_dir)
    log.info("=" * 60)
    log.info("  COMPARE ALL MODELS — ICU Mortality Prediction")
    log.info("=" * 60)

    # ── Device ───────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"  Device: {device}")

    # ── Load data ────────────────────────────────────────────────────
    log.info(f"  Data dir: {args.data_dir}")
    X_train = torch.load(os.path.join(args.data_dir, "X_train.pt"), weights_only=True)
    y_train = torch.load(os.path.join(args.data_dir, "y_train.pt"), weights_only=True)
    X_val   = torch.load(os.path.join(args.data_dir, "X_val.pt"),   weights_only=True)
    y_val   = torch.load(os.path.join(args.data_dir, "y_val.pt"),   weights_only=True)
    X_test  = torch.load(os.path.join(args.data_dir, "X_test.pt"),  weights_only=True)
    y_test  = torch.load(os.path.join(args.data_dir, "y_test.pt"),  weights_only=True)

    # Missing mask
    m_path = os.path.join(args.data_dir, "M_train.pt")
    if os.path.exists(m_path):
        M_train = torch.load(m_path, weights_only=True)
        M_val   = torch.load(os.path.join(args.data_dir, "M_val.pt"),  weights_only=True)
        M_test  = torch.load(os.path.join(args.data_dir, "M_test.pt"), weights_only=True)
        log.info("  Missing mask: LOADED")
    else:
        M_train = M_val = M_test = None
        log.warning("  Missing mask: NOT FOUND")

    log.info(f"  Train: {X_train.shape}  |  Test: {X_test.shape}")
    log.info(f"  Test mortality: {y_test.float().mean():.1%}\n")

    y_test_np = y_test.numpy().astype(int)
    results   = []   # list of dicts: {name, probs, labels, params, best_ep}

    # ══════════════════════════════════════════════════════════════════
    #  1. Logistic Regression
    # ══════════════════════════════════════════════════════════════════
    log.info("─" * 60)
    log.info("  [1/5] Logistic Regression")
    log.info("─" * 60)
    t0 = time.time()

    M_tr_np = M_train.numpy() if M_train is not None else None
    M_te_np = M_test.numpy()  if M_test  is not None else None
    M_va_np = M_val.numpy()   if M_val   is not None else None

    X_tr_feat = extract_features(X_train.numpy(), M_tr_np)
    X_va_feat = extract_features(X_val.numpy(),   M_va_np)
    X_te_feat = extract_features(X_test.numpy(),  M_te_np)

    X_trval = np.concatenate([X_tr_feat, X_va_feat], axis=0)
    y_trval = np.concatenate([y_train.numpy().astype(int),
                              y_val.numpy().astype(int)], axis=0)

    lr_model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    LogisticRegression(
            C=0.1, class_weight="balanced",
            max_iter=1000, solver="lbfgs", random_state=42,
        )),
    ])
    lr_model.fit(X_trval, y_trval)
    lr_probs = lr_model.predict_proba(X_te_feat)[:, 1]
    log.info(f"    Train+predict: {time.time()-t0:.1f}s")
    results.append({
        "name":  "Logistic Regression",
        "probs": lr_probs, "labels": y_test_np,
        "params": "—", "best_ep": "—",
    })

    # ══════════════════════════════════════════════════════════════════
    #  2. XGBoost
    # ══════════════════════════════════════════════════════════════════
    if HAS_XGB:
        log.info("─" * 60)
        log.info("  [2/5] XGBoost")
        log.info("─" * 60)
        t0 = time.time()

        n_neg = (y_trval == 0).sum()
        n_pos = (y_trval == 1).sum()
        scale = n_neg / n_pos

        xgb_model = XGBClassifier(
            n_estimators=500, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=scale,
            eval_metric="auc", early_stopping_rounds=20,
            random_state=42, n_jobs=-1, verbosity=0,
        )
        xgb_model.fit(
            X_tr_feat, y_train.numpy().astype(int),
            eval_set=[(X_va_feat, y_val.numpy().astype(int))],
            verbose=False,
        )
        xgb_probs = xgb_model.predict_proba(X_te_feat)[:, 1]
        log.info(f"    Train+predict: {time.time()-t0:.1f}s "
                 f"(best iter={xgb_model.best_iteration})")
        results.append({
            "name":  "XGBoost",
            "probs": xgb_probs, "labels": y_test_np,
            "params": "—", "best_ep": f"iter={xgb_model.best_iteration}",
        })
    else:
        log.warning("  [2/5] XGBoost SKIPPED (chưa cài).")

    # ══════════════════════════════════════════════════════════════════
    #  3-5. PyTorch models
    # ══════════════════════════════════════════════════════════════════
    pytorch_configs = [
        ("LSTM",        args.lstm_py,        "LSTM_ICU",        args.lstm_ckpt),
        ("Bi-LSTM",     args.bilstm_py,      "BiLSTM_ICU",      args.bilstm_ckpt),
        ("Transformer", args.transformer_py, "Transformer_ICU", args.transformer_ckpt),
    ]

    for i, (name, py, cls, ckpt) in enumerate(pytorch_configs, start=3):
        log.info("─" * 60)
        log.info(f"  [{i}/5] {name}")
        log.info("─" * 60)
        t0 = time.time()
        try:
            r = run_pytorch_model(
                name, py, cls, ckpt,
                X_test, M_test, y_test, device, log
            )
            if r is not None:
                log.info(f"    Loaded + predict: {time.time()-t0:.1f}s "
                         f"(params={r['params']:,}, best_ep={r['best_ep']})")
                results.append(r)
        except Exception as e:
            log.error(f"    [X] Lỗi load {name}: {e}")

    # ══════════════════════════════════════════════════════════════════
    #  TÍNH METRICS + IN BẢNG
    # ══════════════════════════════════════════════════════════════════
    log.info("\n")
    log.info("=" * 80)
    log.info("  KẾT QUẢ TEST SET")
    log.info("=" * 80)

    metrics_keys = ["AUROC","AUPRC","Accuracy","F1",
                    "Sensitivity","Specificity","PPV","NPV"]
    rows = []
    for r in results:
        m = compute_metrics(r["labels"], r["probs"], args.threshold)
        row = {"Model": r["name"]}
        row.update({k: round(m[k], 4) for k in metrics_keys})
        row["TP"]   = m["TP"]; row["FP"] = m["FP"]
        row["FN"]   = m["FN"]; row["TN"] = m["TN"]
        row["Params"]    = r["params"]
        row["Best Epoch"] = r["best_ep"]
        rows.append(row)

    df = pd.DataFrame(rows).set_index("Model")

    # ── In bảng đẹp ra console + log ──────────────────────────────────
    log.info("")
    log.info(f"  {'Metric':<14}" + "".join(f"{r['name']:>16}" for r in results))
    log.info("  " + "-" * (14 + 16 * len(results)))
    for k in metrics_keys:
        vals = [compute_metrics(r["labels"], r["probs"], args.threshold)[k]
                for r in results]
        best = max(vals)
        line = f"  {k:<14}"
        for v in vals:
            mark = " ★" if abs(v - best) < 1e-6 else "  "
            line += f"{v:>14.4f}{mark}"
        log.info(line)
    log.info("  " + "-" * (14 + 16 * len(results)))
    log.info("  ★ = best cho metric đó")

    # ── Save CSV ──────────────────────────────────────────────────────
    csv_path = os.path.join(args.out_dir, "comparison_table.csv")
    df.to_csv(csv_path)
    log.info(f"\n  CSV  : {csv_path}")

    # ── Save Markdown (paste thẳng vào báo cáo) ───────────────────────
    md_path = os.path.join(args.out_dir, "comparison_table.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Bảng so sánh các mô hình — ICU Mortality Prediction\n\n")
        f.write(f"Test set: {len(y_test_np):,} samples  |  "
                f"Mortality rate: {y_test_np.mean():.1%}  |  "
                f"Threshold: {args.threshold}\n\n")
        f.write("| Metric | " + " | ".join(r["name"] for r in results) + " |\n")
        f.write("|--------" + "|--------" * len(results) + "|\n")
        for k in metrics_keys:
            vals = [compute_metrics(r["labels"], r["probs"], args.threshold)[k]
                    for r in results]
            best = max(vals)
            row = f"| {k} |"
            for v in vals:
                cell = f" **{v:.4f}** ★" if abs(v-best) < 1e-6 else f" {v:.4f}"
                row += cell + " |"
            f.write(row + "\n")
    log.info(f"  MD   : {md_path}")

    # ══════════════════════════════════════════════════════════════════
    #  PLOTS — ROC và PR curves
    # ══════════════════════════════════════════════════════════════════
    if HAS_MPL and len(results) >= 2:
        colors = ["#e74c3c","#3498db","#2ecc71","#9b59b6","#f39c12","#1abc9c"]

        # ── ROC Curves ────────────────────────────────────────────────
        plt.figure(figsize=(8, 7))
        for i, r in enumerate(results):
            fpr, tpr, _ = roc_curve(r["labels"], r["probs"])
            auc = roc_auc_score(r["labels"], r["probs"])
            plt.plot(fpr, tpr, color=colors[i % len(colors)], lw=2,
                     label=f"{r['name']}  (AUROC={auc:.4f})")
        plt.plot([0,1],[0,1], "k--", lw=1, alpha=0.5, label="Random")
        plt.xlabel("False Positive Rate", fontsize=12)
        plt.ylabel("True Positive Rate",  fontsize=12)
        plt.title("ROC Curves — ICU Mortality Prediction", fontsize=13)
        plt.legend(fontsize=10, loc="lower right")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        roc_path = os.path.join(args.out_dir, "roc_curves.png")
        plt.savefig(roc_path, dpi=150)
        plt.close()
        log.info(f"  ROC  : {roc_path}")

        # ── PR Curves ─────────────────────────────────────────────────
        plt.figure(figsize=(8, 7))
        baseline = y_test_np.mean()
        for i, r in enumerate(results):
            prec, rec, _ = precision_recall_curve(r["labels"], r["probs"])
            auprc        = average_precision_score(r["labels"], r["probs"])
            plt.plot(rec, prec, color=colors[i % len(colors)], lw=2,
                     label=f"{r['name']}  (AUPRC={auprc:.4f})")
        plt.axhline(y=baseline, color="k", linestyle="--", lw=1, alpha=0.5,
                    label=f"Baseline ({baseline:.3f})")
        plt.xlabel("Recall (Sensitivity)", fontsize=12)
        plt.ylabel("Precision (PPV)",      fontsize=12)
        plt.title("Precision-Recall Curves — ICU Mortality Prediction", fontsize=13)
        plt.legend(fontsize=10, loc="lower left")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        pr_path = os.path.join(args.out_dir, "pr_curves.png")
        plt.savefig(pr_path, dpi=150)
        plt.close()
        log.info(f"  PR   : {pr_path}")

    log.info("\n  Hoàn thành!\n")


if __name__ == "__main__":
    main()
