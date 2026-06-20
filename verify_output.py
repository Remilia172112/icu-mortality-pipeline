"""Kiểm tra kết quả preprocessing."""
import sys, os, json

def main():
    out = sys.argv[1] if len(sys.argv) > 1 else "."
    print(f"Thu muc output: {out}")
    print()

    required = [
        "X_train.pt", "y_train.pt", "M_train.pt",
        "X_val.pt",   "y_val.pt",   "M_val.pt",
        "X_test.pt",  "y_test.pt",  "M_test.pt",
        "scaler.pkl", "feature_names.json", "cohort.parquet",
    ]
    all_ok = True
    for f in required:
        p = os.path.join(out, f)
        if os.path.exists(p):
            size_mb = os.path.getsize(p) / 1e6
            print(f"  [OK]  {f:<30s}  {size_mb:.0f} MB")
        else:
            print(f"  [!!]  {f:<30s}  KHONG TIM THAY")
            all_ok = False

    print()
    if not all_ok:
        print(">>> Mot so file bi thieu. Chay lai 2_run.bat hoac 3_rerun_from_cache.bat")
        return

    import torch
    X_train = torch.load(os.path.join(out, "X_train.pt"), weights_only=True)
    y_train = torch.load(os.path.join(out, "y_train.pt"), weights_only=True)
    X_val   = torch.load(os.path.join(out, "X_val.pt"),   weights_only=True)
    y_val   = torch.load(os.path.join(out, "y_val.pt"),   weights_only=True)
    X_test  = torch.load(os.path.join(out, "X_test.pt"),  weights_only=True)
    y_test  = torch.load(os.path.join(out, "y_test.pt"),  weights_only=True)
    M_train = torch.load(os.path.join(out, "M_train.pt"), weights_only=True)

    with open(os.path.join(out, "feature_names.json"), encoding="utf-8") as f:
        feat = json.load(f)

    print("=== KET QUA TENSOR ===")
    print(f"X_train : {tuple(X_train.shape)}   mortality = {y_train.mean():.1%}")
    print(f"X_val   : {tuple(X_val.shape)}     mortality = {y_val.mean():.1%}")
    print(f"X_test  : {tuple(X_test.shape)}    mortality = {y_test.mean():.1%}")
    print(f"M_train : {tuple(M_train.shape)}   (missing mask)")
    print()
    print(f"Features: {feat['n_total']} total")
    print(f"  Chart  ({feat['n_chart']}): {feat['chart']}")
    print(f"  Lab    ({feat['n_lab']}):   {feat['lab']}")
    print(f"  Treat  ({feat['n_treatment']}): {feat['treatment']}")
    print()

    nan_count = X_train.isnan().sum().item()
    print(f"NaN trong X_train: {nan_count}  (phai = 0)")
    print()
    if nan_count == 0:
        print(">>> Du lieu sach! San sang train LSTM / Transformer.")
    else:
        print(">>> [CANH BAO] Van con NaN. Kiem tra lai pipeline.")

    # Kiểm tra stratified split
    print()
    print("=== KIEM TRA STRATIFIED SPLIT ===")
    for name, y in [("Train", y_train), ("Val", y_val), ("Test", y_test)]:
        mort = y.mean().item() * 100
        print(f"  {name}: {mort:.1f}% mortality  (N={len(y):,})")
    print("  (Ty le phai gan bang nhau giua 3 tap)")


if __name__ == "__main__":
    main()
