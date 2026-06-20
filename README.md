# ICU Mortality Prediction using Deep Learning
# Đánh Giá Mức Độ Sinh Tồn Của Bệnh Nhân Tại ICU Bằng Học Sâu

> 🇻🇳 [Tiếng Việt](#-tiếng-việt) · 🇬🇧 [English](#-english)

A deep learning pipeline for predicting in-hospital mortality of ICU patients using the **MIMIC-IV v3.1** dataset, built as a graduation thesis at **Saigon University, Faculty of Information Technology**.

Một pipeline học sâu dự đoán khả năng tử vong trong viện của bệnh nhân ICU sử dụng bộ dữ liệu **MIMIC-IV v3.1**, thực hiện trong khuôn khổ khóa luận tốt nghiệp tại **Trường Đại học Sài Gòn, Khoa Công nghệ Thông tin**.

---

## 🇬🇧 English

### Overview

| | |
|---|---|
| **Topic** | ICU Mortality Prediction using Deep Learning |
| **University** | Saigon University, Faculty of Information Technology |
| **Dataset** | MIMIC-IV v3.1 (Beth Israel Deaconess Medical Center, Boston) |
| **Main model** | Bi-LSTM (Bidirectional Long Short-Term Memory) |
| **Also compared** | Logistic Regression, XGBoost, LSTM, Transformer |
| **Stack** | Python 3.10, PyTorch 2.x |
| **Dev machine** | Windows 11, 32GB RAM, RTX 5060 (hybrid GPU mode) |

The pipeline turns raw MIMIC-IV CSV tables into 48-hour, 69-feature patient tensors and trains sequence models to predict in-hospital mortality, with an emphasis on correct handling of **missing data**, **class imbalance**, and **train/val/test leakage prevention**.

### Repository structure

```
PREPROCESSOR/
├── ablation/                       # ablation study outputs
├── checkpoints/                    # Bi-LSTM checkpoints
├── checkpoints_lstm/                # LSTM checkpoints
├── checkpoints_transformer/         # Transformer checkpoints
├── checkpoints_v2/ , checkpoints_v4/ # Bi-LSTM version checkpoints
├── multi_seed_results/              # 5-seed experiment outputs
├── results/                         # evaluation results
├── results_compare/                 # cross-model comparison results
│
├── 1_setup.bat                      # create venv + install deps
├── 2_run.bat                        # run preprocessing (~1-2h)
├── 3_rerun_from_cache.bat           # resume from cache if interrupted
├── 4_verify.bat                     # sanity-check output tensors
├── 5_train.bat                      # train Bi-LSTM
├── 6_baseline.bat                   # train baseline (LR / XGBoost)
├── 7_train_lstm.bat                 # train LSTM
├── 8_train_transformer.bat          # train Transformer
├── 9_compare_all.bat                # compare all trained models
├── 10_curves_and_ablation.bat       # plot curves + run ablation study
├── 11_run_multi_seed.bat            # run 5 models x 5 seeds experiment
│
├── ablation_study.py                # ablation study script
├── compare_all.py                   # aggregate & compare model results
├── config.txt                       # DATA_DIR / OUT_DIR / HOURS config
├── csv_to_parquet.py                 # stream CSV -> Parquet (39GB-safe)
├── HUONG_DAN.txt                     # usage guide (Vietnamese)
├── MIMIC4_EDA.ipynb                  # exploratory data analysis notebook
├── mimic4_preprocessing.py           # main script: raw CSV -> tensors (v4)
├── multi_seed_runs.py                # 5 models x 5 seeds experiment script
├── plot_training_curves.py           # plot training curves
└── README.md

OUTLINE/
├── README.md
├── train_baseline.py                # Logistic Regression / XGBoost training
├── train_bilstm.py                  # Bi-LSTM training script
├── train_lstm.py                    # LSTM training script
├── train_transformer.py             # Transformer training script
└── verify_output.py                  # verify preprocessing output
```

### Data pipeline (7 steps)

Input layout expected from MIMIC-IV v3.1:

```
DATA_DIR/
├── hosp/   patients.csv, admissions.csv, labevents.csv (~6GB), diagnoses_icd.csv, ...
└── icu/    icustays.csv, chartevents.csv (~30GB), inputevents.csv (~2GB), outputevents.csv, ...
```

1. **Cohort construction** — merge patients + admissions + ICU stays, keep adults (age ≥ 18) and first ICU stay per admission, label = `hospital_expire_flag`. Result: ~73,000–85,000 ICU stays, ~11–15% mortality.
2. **Chart vitals (33 features)** — chunked read of `chartevents.csv` (10M rows/chunk), itemid → feature name mapping, unit conversion (°F→°C, lbs→kg, creatinine µmol/L→mg/dL), GCS text-to-score parsing, physiological-range outlier filtering, restricted to hours 0–48 of ICU stay, pivoted to wide hourly format.
3. **Lab tests (30 features)** — chunked read of `labevents.csv` (5M rows/chunk), window from −6h to +48h relative to ICU admission, covering renal/liver/coagulation/electrolyte/blood-gas/endocrine/inflammation panels.
4. **Vasopressors (4 features)** — binary hourly indicators for Epinephrine, Norepinephrine, Dobutamine, Dopamine from `inputevents`.
5. **Urine output & CRRT (2 features)** — hourly urine output and continuous renal replacement therapy status.
6. **Merge → split → impute → scale**:
   - Merge all tables into a 69-feature × 48-hour grid per admission.
   - **Stratified split by PatientID** (70/10/20, train/val/test) performed *before* any imputation, to avoid leakage and keep ~11–15% mortality rate across all splits.
   - **Missing mask** `M[i,t,f]` built before imputation (à la GRU-D, Che et al. 2018).
   - **Linear interpolation** per patient (with forward/backward fill at sequence edges).
   - **Median fill** using train-set medians only.
   - **StandardScaler** fit on train only, applied to all splits.
7. **Tensor packaging** — `X_*.pt` (N, 48, 69), `M_*.pt` (N, 48, 69), `y_*.pt` (N,), plus `scaler.pkl`, `feature_names.json`, and processing logs.

### Model architecture (Bi-LSTM)

```
Input: concat(X, M) → (batch, 48, 138)
   ↓
Bidirectional LSTM (hidden=128, layers=2, dropout=0.3)
   ↓
Last timestep → (batch, 256)
   ↓
Dropout(0.3) → Linear(256, 1) → logit
   ↓
BCEWithLogitsLoss (pos_weight ≈ 6.0)
```

**Class imbalance handling (two layers):**
- `WeightedRandomSampler` to balance each training batch (~50/50).
- `pos_weight` in the loss to penalize missed-death predictions ~6× more heavily.

**Training setup:** AdamW (lr=1e-3, weight_decay=1e-4), `ReduceLROnPlateau` on AUROC, gradient clipping (max_norm=1.0), early stopping (patience=10), per-epoch checkpoints + best-AUROC checkpoint, auto GPU detection with hybrid-mode VRAM selection.

**Metrics:** AUROC, AUPRC, Accuracy, F1, Sensitivity, Specificity, PPV, NPV, confusion matrix.

### Results (test set, n = 17,077, mortality 11.1%)

Single-seed (seed = 42):

| Model | AUROC | AUPRC | F1 | PPV | Sensitivity |
|---|---|---|---|---|---|
| Logistic Regression | 0.9197 | 0.6979 | 0.5686 | 0.4396 | **0.8047** |
| XGBoost | **0.9361** | **0.7396** | **0.6215** | 0.5145 | 0.7847 |
| LSTM | 0.9201 | 0.6953 | 0.6032 | 0.5318 | 0.6966 |
| Bi-LSTM | 0.9221 | 0.7048 | 0.6207 | **0.5761** | 0.6728 |
| Transformer | 0.9227 | 0.7015 | 0.6101 | 0.5338 | 0.7119 |

Multi-seed (5 seeds: 42, 123, 456, 789, 2026 — mean ± std):

| Model | AUROC | F1 | Notes |
|---|---|---|---|
| Logistic Regression | 0.9197 ± 0.0000 | 0.5679 | Deterministic |
| **XGBoost** | **0.9350 ± 0.0009** | 0.6203 | Most stable, highest AUROC |
| LSTM | 0.9216 ± 0.0015 | 0.6177 | — |
| **Bi-LSTM** | 0.9254 ± 0.0010 | **0.6254** | Highest F1 |
| Transformer | 0.9249 ± 0.0017 | 0.6167 | Highest variance |

XGBoost's 95% CI does not overlap with any deep learning model, indicating a statistically meaningful edge; Bi-LSTM and Transformer CIs overlap, indicating comparable AUROC between the two.

### Proposed deployment: two-tier alert system

| Tier | Model | Role | Key metric |
|---|---|---|---|
| Tier 1 | Logistic Regression | Broad screening — minimize missed cases | Sensitivity 0.8047 |
| Tier 2 | Bi-LSTM (1-layer) | Confirmation — minimize false alarms | PPV 0.5761 |

This balances sensitivity (wide screening) against precision (actionable alerts), aimed at practical hospital deployment in Vietnam.

### Key concepts applied

- **Data leakage prevention** — split before any statistic (median, scaler) is computed from the full dataset.
- **MNAR (Missing Not At Random)** — in ICU data, missingness itself is clinically informative (e.g., a test wasn't ordered because the patient was stable); the missing mask lets the model learn this pattern.
- **Linear interpolation vs. forward-fill** — interpolation produces a clinically more plausible trend between two real measurements than a flat carried-forward value.
- **Stratified split by PatientID** (not AdmissionID) — prevents the same patient appearing in both train and test.
- **pos_weight** — penalizes false negatives (predicted survival, actual death) ~6× more than false positives, reflecting the clinical cost asymmetry.

### Getting started

```bash
pip install pyarrow pandas numpy scikit-learn joblib tqdm torch
```

1. Edit `config.txt` with your `DATA_DIR` (MIMIC-IV root) and `OUT_DIR`.
2. Run `1_setup.bat` to create the environment.
3. Run `2_run.bat` to execute preprocessing (~5–6 hours for the full dataset; `3_rerun_from_cache.bat` resumes if interrupted).
4. Run `4_verify.bat` to sanity-check the output tensors.
5. Run `5_train.bat` (or `python train_bilstm.py`) to train the Bi-LSTM.
6. Optionally run `run_multi_seed.bat` for the 5-seed robustness experiment.

> **Note:** MIMIC-IV requires PhysioNet credentialing (CITI training + signed data use agreement) before access. This repository does not include the dataset itself.

### Known issues & fixes

| Issue | Cause | Fix |
|---|---|---|
| `KeyError: 'itemid'` | Column dropped too early during chunk processing | Keep `itemid` through unit conversion, drop only after |
| Calcium / Creatinine itemid collision | Same MIMIC itemid (220615) mapped to both | Creatinine from chart (÷88.42 for unit conversion), Calcium from labevents (50893) |
| `KeyError: 'AdmissionID'` after groupby | pandas 2.0+ promotes group key to index | Use `groupby().transform()` instead of `.apply()` |
| `MemoryError` converting 39GB CSV to Parquet | Concatenating all chunks in RAM before writing | Stream with PyArrow `ParquetWriter`, write chunk by chunk |
| RTX hybrid mode picks iGPU | Default device selection | Auto-detect and select the GPU with the most VRAM |

### Acknowledgements

- Dataset: MIMIC-IV v3.1, courtesy of MIT and Beth Israel Deaconess Medical Center (BIDMC), via PhysioNet.

---

## 🇻🇳 Tiếng Việt

### Tổng quan

| | |
|---|---|
| **Đề tài** | Đánh giá mức độ sinh tồn của bệnh nhân tại ICU bằng học sâu |
| **Trường** | Đại học Sài Gòn, Khoa Công nghệ Thông tin |
| **Dataset** | MIMIC-IV v3.1 (Beth Israel Deaconess Medical Center, Boston) |
| **Mô hình chính** | Bi-LSTM (Bidirectional Long Short-Term Memory) |
| **So sánh thêm** | Logistic Regression, XGBoost, LSTM, Transformer |
| **Công nghệ** | Python 3.10, PyTorch 2.x |
| **Máy phát triển** | Windows 11, 32GB RAM, RTX 5060 (hybrid mode) |

Pipeline chuyển đổi dữ liệu CSV thô của MIMIC-IV thành tensor bệnh nhân gồm 48 giờ × 69 đặc trưng, huấn luyện các mô hình sequence để dự đoán tử vong trong viện, với trọng tâm là xử lý đúng **dữ liệu thiếu**, **mất cân bằng lớp**, và **chống rò rỉ dữ liệu** giữa các tập train/val/test.

### Cấu trúc thư mục

```
PREPROCESSOR/
├── ablation/                       # kết quả ablation study
├── checkpoints/                    # checkpoint mô hình Bi-LSTM
├── checkpoints_lstm/                # checkpoint mô hình LSTM
├── checkpoints_transformer/         # checkpoint mô hình Transformer
├── checkpoints_v2/ , checkpoints_v4/ # checkpoint mô hình Bi-LSTM v2 v4
├── multi_seed_results/              # kết quả thí nghiệm 5 seed
├── results/                         # kết quả đánh giá
├── results_compare/                 # kết quả so sánh giữa các mô hình
│
├── 1_setup.bat                      # tạo venv + cài thư viện
├── 2_run.bat                        # chạy preprocessing (~5-6 giờ)
├── 3_rerun_from_cache.bat           # chạy lại từ cache nếu bị ngắt
├── 4_verify.bat                     # kiểm tra tensor đầu ra
├── 5_train.bat                      # huấn luyện Bi-LSTM
├── 6_baseline.bat                   # huấn luyện baseline (LR / XGBoost)
├── 7_train_lstm.bat                 # huấn luyện LSTM
├── 8_train_transformer.bat          # huấn luyện Transformer
├── 9_compare_all.bat                # so sánh tất cả mô hình đã huấn luyện
├── 10_curves_and_ablation.bat       # vẽ đường cong huấn luyện + chạy ablation
├── 11_run_multi_seed.bat            # chạy thí nghiệm 5 mô hình x 5 seed
│
├── ablation_study.py                # script ablation study
├── compare_all.py                   # tổng hợp & so sánh kết quả mô hình
├── config.txt                       # cấu hình DATA_DIR / OUT_DIR / HOURS
├── csv_to_parquet.py                 # nén CSV streaming (an toàn với 39GB)
├── HUONG_DAN.txt                     # hướng dẫn sử dụng
├── MIMIC4_EDA.ipynb                  # notebook phân tích khám phá dữ liệu
├── mimic4_preprocessing.py           # script chính: CSV thô -> tensor (v4)
├── multi_seed_runs.py                # script thí nghiệm 5 mô hình x 5 seed
├── plot_training_curves.py           # vẽ đường cong huấn luyện
└── README.md

OUTLINE/
├── README.md
├── train_baseline.py                # huấn luyện Logistic Regression / XGBoost
├── train_bilstm.py                  # script huấn luyện Bi-LSTM
├── train_lstm.py                    # script huấn luyện LSTM
├── train_transformer.py             # script huấn luyện Transformer
└── verify_output.py                  # kiểm tra kết quả tiền xử lý
```

### Pipeline tiền xử lý (7 bước)

Cấu trúc thư mục dữ liệu đầu vào của MIMIC-IV v3.1:

```
DATA_DIR/
├── hosp/   patients.csv, admissions.csv, labevents.csv (~6GB), diagnoses_icd.csv, ...
└── icu/    icustays.csv, chartevents.csv (~30GB), inputevents.csv (~2GB), outputevents.csv, ...
```

1. **Xây dựng cohort** — merge patients + admissions + ICU stays, lọc tuổi ≥ 18 và lần nhập ICU đầu tiên của mỗi admission, nhãn = `hospital_expire_flag`. Kết quả: ~73.000–85.000 ICU stays, tỷ lệ tử vong ~11–15%.
2. **Dấu hiệu sinh tồn — 33 features** — đọc `chartevents.csv` theo chunk (10 triệu dòng/chunk), mapping itemid sang tên feature, chuyển đơn vị (°F→°C, lbs→kg, creatinine µmol/L→mg/dL), parse GCS từ text sang số, lọc outlier theo ngưỡng sinh lý, chỉ giữ 0–48 giờ đầu ICU, pivot sang dạng wide theo giờ.
3. **Xét nghiệm — 30 features** — đọc `labevents.csv` theo chunk (5 triệu dòng/chunk), cửa sổ thời gian từ −6h đến +48h so với lúc nhập ICU, bao gồm các nhóm thận/gan/đông máu/điện giải/khí máu/nội tiết/viêm.
4. **Thuốc vận mạch — 4 features** — chỉ báo nhị phân theo giờ cho Epinephrine, Norepinephrine, Dobutamine, Dopamine từ `inputevents`.
5. **Nước tiểu & CRRT — 2 features** — lượng nước tiểu theo giờ và trạng thái lọc máu liên tục (CRRT).
6. **Merge → chia tập → điền thiếu → chuẩn hóa**:
   - Gộp các bảng thành lưới 69 đặc trưng × 48 giờ cho mỗi lần nhập viện.
   - **Chia tập theo PatientID** (70/10/20, train/val/test) thực hiện *trước* mọi bước điền dữ liệu, để tránh rò rỉ và giữ tỷ lệ tử vong nhất quán ~11–15% ở cả 3 tập.
   - **Missing mask** `M[i,t,f]` được tạo trước khi điền thiếu (theo hướng GRU-D, Che et al. 2018).
   - **Nội suy tuyến tính** theo từng bệnh nhân (kèm forward/backward fill ở hai đầu chuỗi).
   - **Điền median** chỉ tính từ tập train.
   - **StandardScaler** fit chỉ trên train, áp dụng cho cả 3 tập.
7. **Đóng gói tensor** — `X_*.pt` (N, 48, 69), `M_*.pt` (N, 48, 69), `y_*.pt` (N,), kèm `scaler.pkl`, `feature_names.json`, và log xử lý.

### Kiến trúc mô hình (Bi-LSTM)

```
Input: ghép(X, M) → (batch, 48, 138)
   ↓
Bi-LSTM (hidden=128, layers=2, dropout=0.3)
   ↓
Output bước thời gian cuối → (batch, 256)
   ↓
Dropout(0.3) → Linear(256, 1) → logit
   ↓
BCEWithLogitsLoss (pos_weight ≈ 6.0)
```

**Xử lý mất cân bằng lớp (2 lớp bảo vệ):**
- `WeightedRandomSampler` để cân bằng mỗi batch huấn luyện (~50/50).
- `pos_weight` trong loss function để phạt nặng hơn ~6 lần các dự đoán bỏ sót ca tử vong.

**Cấu hình huấn luyện:** AdamW (lr=1e-3, weight_decay=1e-4), `ReduceLROnPlateau` theo AUROC, gradient clipping (max_norm=1.0), early stopping (patience=10), checkpoint mỗi epoch + checkpoint AUROC tốt nhất, tự động phát hiện GPU và chọn GPU có VRAM lớn nhất ở chế độ hybrid.

**Chỉ số đánh giá:** AUROC, AUPRC, Accuracy, F1, Sensitivity, Specificity, PPV, NPV, confusion matrix.

### Kết quả (tập test, n = 17.077, tỷ lệ tử vong 11,1%)

Single-seed (seed = 42):

| Mô hình | AUROC | AUPRC | F1 | PPV | Sensitivity |
|---|---|---|---|---|---|
| Logistic Regression | 0.9197 | 0.6979 | 0.5686 | 0.4396 | **0.8047** |
| XGBoost | **0.9361** | **0.7396** | **0.6215** | 0.5145 | 0.7847 |
| LSTM | 0.9201 | 0.6953 | 0.6032 | 0.5318 | 0.6966 |
| Bi-LSTM | 0.9221 | 0.7048 | 0.6207 | **0.5761** | 0.6728 |
| Transformer | 0.9227 | 0.7015 | 0.6101 | 0.5338 | 0.7119 |

Multi-seed (5 seed: 42, 123, 456, 789, 2026 — mean ± std):

| Mô hình | AUROC | F1 | Nhận xét |
|---|---|---|---|
| Logistic Regression | 0.9197 ± 0.0000 | 0.5679 | Deterministic |
| **XGBoost** | **0.9350 ± 0.0009** | 0.6203 | Ổn định nhất, AUROC cao nhất |
| LSTM | 0.9216 ± 0.0015 | 0.6177 | — |
| **Bi-LSTM** | 0.9254 ± 0.0010 | **0.6254** | F1 cao nhất |
| Transformer | 0.9249 ± 0.0017 | 0.6167 | Biến thiên cao nhất |

Khoảng tin cậy 95% của XGBoost không trùng với các mô hình deep learning, cho thấy sự vượt trội có ý nghĩa thống kê; khoảng tin cậy của Bi-LSTM và Transformer trùng nhau, cho thấy hai mô hình này tương đương về AUROC.

### Đề xuất triển khai: hệ thống cảnh báo hai tầng

| Tầng | Mô hình | Vai trò | Chỉ số chính |
|---|---|---|---|
| Tầng 1 | Logistic Regression | Sàng lọc rộng — không bỏ sót ca | Sensitivity 0.8047 |
| Tầng 2 | Bi-LSTM (1-layer) | Xác nhận — không quá tải cảnh báo | PPV 0.5761 |

Cách kết hợp này cân bằng giữa độ nhạy (sàng lọc rộng) và độ chính xác (cảnh báo có giá trị), phù hợp với triển khai thực tế tại bệnh viện Việt Nam.

### Các khái niệm quan trọng

- **Chống rò rỉ dữ liệu (data leakage)** — chia tập trước khi tính bất kỳ thống kê nào (median, scaler) từ toàn bộ dữ liệu.
- **MNAR (Missing Not At Random)** — trong dữ liệu ICU, việc thiếu dữ liệu mang ý nghĩa lâm sàng (ví dụ: không đo vì bệnh nhân ổn định); missing mask giúp mô hình học được pattern này.
- **Nội suy tuyến tính so với forward-fill** — nội suy tạo ra xu hướng hợp lý hơn về y khoa giữa hai điểm đo thật, so với giá trị lặp lại phẳng của forward-fill.
- **Chia tập theo PatientID** (không theo AdmissionID) — tránh việc cùng một bệnh nhân xuất hiện ở cả tập train và test.
- **pos_weight** — phạt dự đoán âm tính giả (dự đoán sống nhưng thực tế tử vong) nặng hơn ~6 lần so với chiều ngược lại, phản ánh đúng mức độ nghiêm trọng về mặt y tế.

### Bắt đầu sử dụng

```bash
pip install pyarrow pandas numpy scikit-learn joblib tqdm torch
```

1. Sửa `config.txt` với `DATA_DIR` (thư mục gốc MIMIC-IV) và `OUT_DIR` của bạn.
2. Chạy `1_setup.bat` để tạo môi trường.
3. Chạy `2_run.bat` để thực hiện tiền xử lý (~5–6 giờ với toàn bộ dataset; dùng `3_rerun_from_cache.bat` để chạy lại nếu bị ngắt).
4. Chạy `4_verify.bat` để kiểm tra tensor đầu ra.
5. Chạy `5_train.bat` (hoặc `python train_bilstm.py`) để huấn luyện Bi-LSTM.
6. Có thể chạy thêm `run_multi_seed.bat` cho thí nghiệm 5 seed.

> **Lưu ý:** MIMIC-IV yêu cầu chứng nhận trên PhysioNet (hoàn thành CITI training + ký thỏa thuận sử dụng dữ liệu) trước khi truy cập. Repository này không bao gồm dataset.

### Các lỗi đã gặp và cách khắc phục

| Lỗi | Nguyên nhân | Cách khắc phục |
|---|---|---|
| `KeyError: 'itemid'` | Cột bị drop quá sớm trong quá trình xử lý chunk | Giữ `itemid` đến khi xử lý xong đơn vị, drop sau |
| Xung đột itemid giữa Calcium / Creatinine | Cùng itemid MIMIC (220615) được map cho cả hai | Creatinine lấy từ chart (chia 88.42 để đổi đơn vị), Calcium lấy từ labevents (50893) |
| `KeyError: 'AdmissionID'` sau groupby | pandas 2.0+ promote group key thành index | Dùng `groupby().transform()` thay vì `.apply()` |
| `MemoryError` khi nén CSV 39GB sang Parquet | Concat toàn bộ chunk vào RAM trước khi ghi | Dùng PyArrow `ParquetWriter` streaming, ghi từng chunk |
| Hybrid mode chọn iGPU thay vì RTX | Mặc định chọn thiết bị sai | Tự động phát hiện và chọn GPU có VRAM lớn nhất |

### Ghi nhận

- Dataset: MIMIC-IV v3.1, do MIT và Beth Israel Deaconess Medical Center (BIDMC) cung cấp qua PhysioNet.
