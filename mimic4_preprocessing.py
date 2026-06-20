"""
=============================================================
  MIMIC-IV Preprocessing — Khóa luận ICU Mortality
=============================================================
  Cách chạy:
      python mimic4_preprocessing.py --data_dir E:\\MIMIC4 --out_dir E:\\mimic4_processed

  Chạy lại từ cache (khi bị ngắt giữa chừng):
      python mimic4_preprocessing.py --data_dir E:\\MIMIC4 --out_dir E:\\mimic4_processed
          --skip_chart --skip_lab --skip_vaso --skip_urine

═══════════════════════════════════════════════════════════════
  THIẾT KẾ PIPELINE — TRÁNH DATA LEAKAGE
═══════════════════════════════════════════════════════════════

  THỨ TỰ ĐÚNG:

    Raw data
      │
      ├─ Step 1-5: Trích xuất raw values (chart, lab, vaso, urine)
      │            Chỉ dùng kiến thức miền (ngưỡng sinh lý cố định)
      │            để lọc outlier — không học gì từ data.
      │
      ├─ Step 6a: SPLIT trước theo PatientID
      │            → train_adm / val_adm / test_adm
      │
      ├─ Step 6b: ffill/bfill PER PATIENT (trong window 48h)
      │            Không phải data leakage vì chỉ dùng data
      │            của chính bệnh nhân đó, không nhìn sang người khác.
      │
      ├─ Step 6c: Tính median TRÊN TRAIN → fill NaN còn lại
      │            Val/Test dùng median của Train (không tính lại).
      │
      └─ Step 6d: StandardScaler fit TRÊN TRAIN → transform tất cả
                   Val/Test dùng mean/std của Train (không fit lại).

  TẠI SAO ffill/bfill KHÔNG PHẢI DATA LEAKAGE:
    ffill/bfill chỉ lan truyền giá trị trong chuỗi thời gian của
    CHÍNH bệnh nhân đó (ví dụ: nhịp tim lúc 8h fill sang 9h nếu
    9h không đo). Nó không sử dụng thông tin từ bệnh nhân khác,
    nên không "rò rỉ" thông tin từ val/test sang train.

  THAY ĐỔI TỪ v2:
    [FIX-1] KeyError 'itemid' — giữ itemid trong chunks, drop sau.
    [FIX-2] Ca/Creatinine conflict itemid 220615 — tách riêng.
    [ADD]   File log preprocessing_YYYYMMDD_HHMMSS.log.
    [v3]    Đảm bảo SPLIT xảy ra TRƯỚC impute median và scaling.
            Median và scaler params chỉ tính trên train.
"""

import os, gc, json, argparse, warnings, logging
from datetime import datetime
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tqdm import tqdm

warnings.filterwarnings("ignore")


# ══════════════════════════════════════════════════════════════════════
#  LOGGING — vừa in ra màn hình, vừa ghi vào file
# ══════════════════════════════════════════════════════════════════════
def setup_logging(out_dir: str) -> logging.Logger:
    os.makedirs(out_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path  = os.path.join(out_dir, f"preprocessing_{timestamp}.log")

    fmt = logging.Formatter(
        "%(asctime)s  %(levelname)s  %(message)s", datefmt="%H:%M:%S"
    )
    logger = logging.getLogger("MIMIC4-BVQ11")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    ch = logging.StreamHandler()          # màn hình — INFO+
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    fh = logging.FileHandler(log_path, encoding="utf-8")  # file — DEBUG+
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    logger.info(f"Log file: {log_path}")
    return logger


log = logging.getLogger("MIMIC4-BVQ11")


# ══════════════════════════════════════════════════════════════════════
#  BẢNG MAPPING — theo file Selected_Item của BVQ11
# ══════════════════════════════════════════════════════════════════════

# [FIX-2] itemid 220615 chỉ map vào "Creatinine_raw" (µmol/L).
# Calcium KHÔNG lấy từ chart (conflict), lấy từ labevents (50893).
CHART_ITEM2VAR = {
    # Dấu hiệu sinh tồn
    220045: "HR",
    220210: "RR",     224690: "RR",   224422: "RR",
    223761: "Temp_F", 223762: "Temp_C",
    220179: "SBP",    220050: "SBP",
    220180: "DBP",    220051: "DBP",
    220052: "MBP",    220181: "MBP",
    220277: "SpO2",
    # GCS
    220739: "GCS_eye",
    223901: "GCS_motor",
    223900: "GCS_verbal",
    # Công thức máu
    220546: "WBC",
    220545: "HCT",
    227457: "PLT",
    # Khí máu
    220274: "pH",
    223835: "FiO2",
    220224: "PaO2",
    220235: "PaCO2",
    227443: "HCO3",
    # Tim
    227445: "CKMB",
    227446: "BNP",
    # Gan
    220587: "AST",
    220644: "ALT",
    225690: "Bili_total",
    225651: "Bili_direct",
    # Đông máu
    225636: "DDimer",
    # Thận — [FIX-2] 220615 duy nhất map vào Creatinine_raw
    225624: "BUN",
    220615: "Creatinine_raw",   # µmol/L → chia 88.42 → mg/dL
    # Nội tiết
    227463: "Cortisol",
    # Điện giải (Ca bị xóa khỏi chart, lấy từ lab)
    220645: "Na",
    227442: "K",
    220602: "Cl",
    220635: "Mg",
    # Glucose, nhân trắc
    220621: "Glucose",
    226707: "Height_in",   # inches → *2.54 → cm
    226730: "Height_cm",   # cm giữ nguyên
    226512: "Weight_kg",   # kg giữ nguyên
    226531: "Weight_lbs",  # lbs → *0.453592 → kg
}

GCS_EYE_MAP = {
    "None": 0, "1 No Response": 1, "2 To pain": 2, "To Pain": 2,
    "3 To speech": 3, "To Speech": 3, "4 Spontaneously": 4, "Spontaneously": 4,
}
GCS_MOTOR_MAP = {
    "1 No Response": 1, "No response": 1,
    "2 Abnorm extensn": 2, "Abnormal extension": 2,
    "3 Abnorm flexion": 3, "Abnormal Flexion": 3,
    "4 Flex-withdraws": 4, "Flex-withdraws": 4,
    "5 Localizes Pain": 5, "Localizes Pain": 5,
    "6 Obeys Commands": 6, "Obeys Commands": 6,
}
GCS_VERBAL_MAP = {
    "No Response-ETT": 1, "No Response": 1, "1 No Response": 1, "1.0 ET/Trach": 1,
    "2 Incomp sounds": 2, "Incomprehensible sounds": 2,
    "3 Inapprop words": 3, "Inappropriate Words": 3,
    "4 Confused": 4, "Confused": 4,
    "5 Oriented": 5, "Oriented": 5,
}

LAB_ITEM2VAR = {
    51301: "WBC_lab",   51221: "HCT_lab",   51265: "PLT_lab",
    50820: "pH_lab",    50821: "PaO2_lab",  50818: "PaCO2_lab", 50882: "HCO3_lab",
    50911: "CKMB_lab",  50963: "BNP_lab",
    50878: "AST_lab",   50861: "ALT_lab",   50862: "Albumin_lab",
    50885: "Bili_total_lab", 50883: "Bili_direct_lab", 50884: "Bili_indirect_lab",
    51274: "PT_lab",    51275: "PTT_lab",   50915: "DDimer_lab",
    51006: "BUN_lab",   50912: "Creatinine_lab",
    50983: "Na_lab",    50971: "K_lab",     50902: "Cl_lab",
    50893: "Ca_lab",    # [FIX-2] Calcium từ lab
    50960: "Mg_lab",
    50993: "TSH_lab",   51001: "FT3_lab",   50995: "FT4_lab",
    50889: "CRP_lab",   50813: "Lactate_lab",
}

VASOPRESSOR_ITEMS = {
    221289: "Epinephrine", 221906: "Norepinephrine",
    221653: "Dobutamine",  221662: "Dopamine",
}
URINE_ITEMS = [226559, 226560, 226561, 226563, 226564,
               226565, 226567, 226631, 226632, 227489]
CRRT_ITEM = 225802

CHART_FEATURES = [
    "HR", "RR", "Temp_C", "SBP", "DBP", "MBP", "SpO2",
    "GCS_total",
    "WBC", "HCT", "PLT",
    "pH", "FiO2", "PaO2", "PaCO2", "HCO3",
    "CKMB", "BNP",
    "AST", "ALT", "Bili_total", "Bili_direct",
    "DDimer",
    "BUN", "Creatinine",
    "Cortisol",
    "Na", "K", "Cl", "Mg",
    "Glucose",
    "Height", "Weight",
]
LAB_FEATURES = [
    "WBC_lab", "HCT_lab", "PLT_lab",
    "pH_lab", "PaO2_lab", "PaCO2_lab", "HCO3_lab",
    "CKMB_lab", "BNP_lab",
    "AST_lab", "ALT_lab", "Albumin_lab",
    "Bili_total_lab", "Bili_direct_lab", "Bili_indirect_lab",
    "PT_lab", "PTT_lab", "DDimer_lab",
    "BUN_lab", "Creatinine_lab",
    "Na_lab", "K_lab", "Cl_lab", "Ca_lab", "Mg_lab",
    "TSH_lab", "FT3_lab", "FT4_lab",
    "CRP_lab", "Lactate_lab",
]
TREATMENT_FEATURES = [
    "Epinephrine_used", "Norepinephrine_used",
    "Dobutamine_used",  "Dopamine_used",
    "Urine_output_mL",  "CRRT_active",
]
ALL_FEATURES = CHART_FEATURES + LAB_FEATURES + TREATMENT_FEATURES
# Tổng: 33 + 30 + 6 = 69 features

VALID_RANGE = {
    "HR": (10, 300), "RR": (2, 80), "Temp_C": (25, 45), "Temp_F": (77, 113),
    "SBP": (30, 300), "DBP": (10, 200), "MBP": (20, 250), "SpO2": (50, 100),
    "GCS_total": (3, 15),
    "WBC": (0.1, 500), "HCT": (5, 70), "PLT": (1, 3000),
    "pH": (6.5, 8.5), "FiO2": (0.1, 1.05),
    "PaO2": (10, 700), "PaCO2": (10, 200), "HCO3": (5, 60),
    "CKMB": (0, 10000), "BNP": (1, 50000),
    "AST": (1, 50000), "ALT": (1, 50000),
    "Bili_total": (0.1, 100), "Bili_direct": (0.1, 100),
    "DDimer": (0.1, 100000),
    "BUN": (1, 500), "Creatinine": (0.1, 50),
    "Cortisol": (0.1, 200),
    "Na": (100, 180), "K": (1, 10), "Cl": (70, 150), "Mg": (0.5, 10),
    "Glucose": (10, 2000), "Height": (50, 250), "Weight": (10, 400),
    "WBC_lab": (0.1, 500), "HCT_lab": (5, 70), "PLT_lab": (1, 3000),
    "pH_lab": (6.5, 8.5), "PaO2_lab": (10, 700), "PaCO2_lab": (10, 200),
    "HCO3_lab": (5, 60), "Creatinine_lab": (0.1, 50), "BUN_lab": (1, 500),
    "Albumin_lab": (0.5, 8), "Lactate_lab": (0.1, 30), "CRP_lab": (0, 500),
    "Ca_lab": (3, 20), "Na_lab": (100, 180), "K_lab": (1, 10),
    "Cl_lab": (70, 150), "Mg_lab": (0.5, 10),
}


# ══════════════════════════════════════════════════════════════════════
#  STEP 1 — COHORT
# ══════════════════════════════════════════════════════════════════════
def build_cohort(data_dir):
    log.info("=" * 60)
    log.info("STEP 1: Xây dựng cohort bệnh nhân ICU")
    log.info("=" * 60)

    patients = pd.read_csv(
        os.path.join(data_dir, "hosp", "patients.csv"),
        usecols=["subject_id", "gender", "anchor_age", "dod"],
        parse_dates=["dod"],
    )
    admissions = pd.read_csv(
        os.path.join(data_dir, "hosp", "admissions.csv"),
        usecols=["subject_id", "hadm_id", "admittime", "dischtime",
                 "deathtime", "hospital_expire_flag"],
        parse_dates=["admittime", "dischtime", "deathtime"],
    )
    icustays = pd.read_csv(
        os.path.join(data_dir, "icu", "icustays.csv"),
        usecols=["subject_id", "hadm_id", "stay_id", "intime", "outtime", "los"],
        parse_dates=["intime", "outtime"],
    )

    icustays = (
        icustays.sort_values(["subject_id", "hadm_id", "intime"])
        .groupby(["subject_id", "hadm_id"], as_index=False)
        .first()
    )
    cohort = (
        icustays
        .merge(admissions, on=["subject_id", "hadm_id"], how="inner")
        .merge(patients,   on="subject_id",               how="inner")
    )
    cohort = cohort[cohort["anchor_age"] >= 18].copy()
    cohort["Outcome"]   = cohort["hospital_expire_flag"].astype(int)
    cohort["Sex"]       = (cohort["gender"] == "M").astype(int)
    cohort["LOS_hours"] = cohort["los"].astype(float) * 24
    cohort = cohort.rename(columns={
        "subject_id": "PatientID", "hadm_id": "AdmissionID",
        "stay_id": "StayID",       "intime":  "ICUInTime",
        "outtime": "ICUOutTime",   "anchor_age": "Age",
    })[["PatientID", "AdmissionID", "StayID",
        "ICUInTime", "ICUOutTime",
        "Age", "Sex", "Outcome", "LOS_hours"]].reset_index(drop=True)

    log.info(f"  ICU stays      : {len(cohort):,}")
    log.info(f"  Unique patients: {cohort['PatientID'].nunique():,}")
    log.info(f"  Mortality      : {cohort['Outcome'].sum():,}  "
             f"({cohort['Outcome'].mean()*100:.1f}%)")
    return cohort


# ══════════════════════════════════════════════════════════════════════
#  STEP 2 — CHARTEVENTS
# ══════════════════════════════════════════════════════════════════════
def process_chartevents(data_dir, cohort, hours=48):
    log.info("=" * 60)
    log.info(f"STEP 2: Xử lý chartevents (first {hours}h ICU)")
    log.info(f"  File: {os.path.join(data_dir, 'icu', 'chartevents.csv')}")
    log.info("=" * 60)

    stay2intime = cohort.set_index("StayID")["ICUInTime"].to_dict()
    valid_stays = set(cohort["StayID"].tolist())
    valid_items = set(CHART_ITEM2VAR.keys())

    chunks    = []
    row_total = 0

    for chunk in tqdm(
        pd.read_csv(
            os.path.join(data_dir, "icu", "chartevents.csv"),
            usecols=["subject_id", "hadm_id", "stay_id",
                     "charttime", "itemid", "value", "valuenum"],
            chunksize=10_000_000,
            low_memory=False,
        ),
        total=44,
        desc="  chartevents",
    ):
        chunk = chunk[
            chunk["stay_id"].isin(valid_stays) &
            chunk["itemid"].isin(valid_items)
        ].copy()
        if chunk.empty:
            continue

        chunk["charttime"]  = pd.to_datetime(chunk["charttime"])
        chunk["icu_intime"] = chunk["stay_id"].map(stay2intime)
        chunk["hours_in"]   = (
            chunk["charttime"] - chunk["icu_intime"]
        ).dt.total_seconds() / 3600
        chunk = chunk[(chunk["hours_in"] >= 0) & (chunk["hours_in"] < hours)]
        if chunk.empty:
            continue

        chunk["Variable"]    = chunk["itemid"].map(CHART_ITEM2VAR)
        chunk["hour_bucket"] = chunk["hours_in"].astype(int).clip(0, hours - 1)

        # [FIX-1] Giữ "itemid" để xử lý đơn vị sau concat
        row_total += len(chunk)
        chunks.append(chunk[[
            "subject_id", "hadm_id", "stay_id",
            "hour_bucket", "Variable", "itemid", "value", "valuenum"
        ]])

    log.info(f"  Tổng rows sau lọc: {row_total:,}")
    if not chunks:
        raise RuntimeError("Không có dữ liệu chartevents nào khớp với cohort!")

    log.info("  Ghép chunks...")
    df = pd.concat(chunks, ignore_index=True)
    del chunks; gc.collect()
    log.debug(f"  Shape sau concat: {df.shape}")

    # ── Xử lý đơn vị ──────────────────────────────────────────

    # Nhiệt độ F → C
    m = df["Variable"] == "Temp_F"
    df.loc[m, "valuenum"] = (pd.to_numeric(df.loc[m, "valuenum"], errors="coerce") - 32) * 5.0 / 9.0
    df.loc[m, "Variable"] = "Temp_C"

    # Weight lbs → kg
    m = df["Variable"] == "Weight_lbs"
    df.loc[m, "valuenum"] = pd.to_numeric(df.loc[m, "valuenum"], errors="coerce") * 0.453592
    df.loc[m, "Variable"] = "Weight"
    df.loc[df["Variable"] == "Weight_kg", "Variable"] = "Weight"

    # Height inches → cm
    m = df["Variable"] == "Height_in"
    df.loc[m, "valuenum"] = pd.to_numeric(df.loc[m, "valuenum"], errors="coerce") * 2.54
    df.loc[m, "Variable"] = "Height"
    df.loc[df["Variable"] == "Height_cm", "Variable"] = "Height"

    # [FIX-2] Creatinine_raw µmol/L → mg/dL
    m = df["Variable"] == "Creatinine_raw"
    df.loc[m, "valuenum"] = pd.to_numeric(df.loc[m, "valuenum"], errors="coerce") / 88.42
    df.loc[m, "Variable"] = "Creatinine"

    # GCS text → số
    for var, mapping in [
        ("GCS_eye",    GCS_EYE_MAP),
        ("GCS_motor",  GCS_MOTOR_MAP),
        ("GCS_verbal", GCS_VERBAL_MAP),
    ]:
        m = df["Variable"] == var
        df.loc[m, "valuenum"] = df.loc[m, "value"].map(mapping)

    # Chuyển số, drop NaN, drop cột không cần
    df["valuenum"] = pd.to_numeric(df["valuenum"], errors="coerce")
    df = df.dropna(subset=["valuenum"])
    df = df.drop(columns=["itemid", "value"])  # bỏ sau khi đã xử lý

    # ── Lọc outlier ──────────────────────────────────────────
    before = len(df)
    for feat, (lo, hi) in VALID_RANGE.items():
        m = df["Variable"] == feat
        if m.any():
            df = df[~(m & ((df["valuenum"] < lo) | (df["valuenum"] > hi)))]
    log.info(f"  Outlier removed: {before - len(df):,} rows")

    # ── GCS total ────────────────────────────────────────────
    gcs_vars = ["GCS_eye", "GCS_motor", "GCS_verbal"]
    gcs_mask = df["Variable"].isin(gcs_vars)
    if gcs_mask.any():
        gcs_total = (
            df[gcs_mask]
            .groupby(["subject_id", "hadm_id", "stay_id", "hour_bucket"])
            ["valuenum"].sum().reset_index()
        )
        gcs_total["Variable"] = "GCS_total"
        df = pd.concat([df[~gcs_mask], gcs_total], ignore_index=True)
        log.debug(f"  GCS_total rows: {len(gcs_total):,}")

    # ── Pivot wide ───────────────────────────────────────────
    log.info("  Pivot sang dạng wide...")
    wide = (
        df.groupby(["subject_id", "hadm_id", "hour_bucket", "Variable"])
        ["valuenum"].mean().reset_index()
        .pivot_table(
            index=["subject_id", "hadm_id", "hour_bucket"],
            columns="Variable", values="valuenum",
        ).reset_index()
    )
    wide.columns.name = None
    wide = wide.rename(columns={"subject_id": "PatientID", "hadm_id": "AdmissionID"})

    for col in CHART_FEATURES:
        if col not in wide.columns:
            wide[col] = np.nan
            log.debug(f"  '{col}' không có data → cột NaN")

    log.info(f"  Chart wide: {wide.shape}  |  admissions: {wide['AdmissionID'].nunique():,}")
    return wide[["PatientID", "AdmissionID", "hour_bucket"] + CHART_FEATURES]


# ══════════════════════════════════════════════════════════════════════
#  STEP 3 — LABEVENTS
# ══════════════════════════════════════════════════════════════════════
def process_labevents(data_dir, cohort, hours=48):
    log.info("=" * 60)
    log.info(f"STEP 3: Xử lý labevents (first {hours}h, bao gồm -6h trước ICU)")
    log.info(f"  File: {os.path.join(data_dir, 'hosp', 'labevents.csv')}")
    log.info("=" * 60)

    adm2intime = cohort.set_index("AdmissionID")["ICUInTime"].to_dict()
    valid_adm  = set(cohort["AdmissionID"].tolist())
    valid_labs = set(LAB_ITEM2VAR.keys())
    chunks     = []
    row_total  = 0

    for chunk in tqdm(
        pd.read_csv(
            os.path.join(data_dir, "hosp", "labevents.csv"),
            usecols=["subject_id", "hadm_id", "itemid", "charttime", "valuenum"],
            chunksize=10_000_000, low_memory=False,
        ),
        total= 16,
        desc="  labevents",
    ):
        chunk = chunk.dropna(subset=["hadm_id", "valuenum"])
        chunk["hadm_id"] = chunk["hadm_id"].astype(int)
        chunk = chunk[
            chunk["hadm_id"].isin(valid_adm) & chunk["itemid"].isin(valid_labs)
        ].copy()
        if chunk.empty:
            continue

        chunk["charttime"]  = pd.to_datetime(chunk["charttime"])
        chunk["icu_intime"] = chunk["hadm_id"].map(adm2intime)
        chunk = chunk.dropna(subset=["icu_intime"])
        chunk["hours_in"] = (
            chunk["charttime"] - chunk["icu_intime"]
        ).dt.total_seconds() / 3600
        chunk = chunk[(chunk["hours_in"] >= -6) & (chunk["hours_in"] < hours)]
        if chunk.empty:
            continue

        chunk["Variable"]    = chunk["itemid"].map(LAB_ITEM2VAR)
        chunk["hour_bucket"] = (
            chunk["hours_in"].apply(lambda x: max(0, int(x))).clip(0, hours - 1)
        )
        row_total += len(chunk)
        chunks.append(chunk[["hadm_id", "hour_bucket", "Variable", "valuenum"]])

    log.info(f"  Rows sau lọc: {row_total:,}")
    if not chunks:
        log.warning("  Không có dữ liệu labevents!")
        return pd.DataFrame(columns=["AdmissionID", "hour_bucket"] + LAB_FEATURES)

    log.info("  Ghép chunks...")
    df = pd.concat(chunks, ignore_index=True)
    del chunks; gc.collect()

    before = len(df)
    for feat, (lo, hi) in VALID_RANGE.items():
        m = df["Variable"] == feat
        if m.any():
            df = df[~(m & ((df["valuenum"] < lo) | (df["valuenum"] > hi)))]
    log.info(f"  Outlier removed: {before - len(df):,} rows")

    log.info("  Pivot sang dạng wide...")
    wide = (
        df.groupby(["hadm_id", "hour_bucket", "Variable"])["valuenum"]
        .mean().reset_index()
        .pivot_table(
            index=["hadm_id", "hour_bucket"],
            columns="Variable", values="valuenum",
        ).reset_index()
    )
    wide.columns.name = None
    wide = wide.rename(columns={"hadm_id": "AdmissionID"})

    for col in LAB_FEATURES:
        if col not in wide.columns:
            wide[col] = np.nan
            log.debug(f"  '{col}' không có data → cột NaN")

    log.info(f"  Lab wide: {wide.shape}")
    return wide[["AdmissionID", "hour_bucket"] + LAB_FEATURES]


# ══════════════════════════════════════════════════════════════════════
#  STEP 4 — THUỐC VẬN MẠCH
# ══════════════════════════════════════════════════════════════════════
def process_vasopressors(data_dir, cohort, hours=48):
    log.info("=" * 60)
    log.info("STEP 4: Xử lý thuốc vận mạch (inputevents)")
    log.info("=" * 60)

    stay2intime = cohort.set_index("StayID")["ICUInTime"].to_dict()
    adm_map     = cohort.set_index("StayID")["AdmissionID"].to_dict()
    valid_stays = set(cohort["StayID"].tolist())
    vaso_cols   = [v + "_used" for v in VASOPRESSOR_ITEMS.values()]

    df = pd.read_csv(
        os.path.join(data_dir, "icu", "inputevents.csv"),
        usecols=["stay_id", "starttime", "endtime", "itemid", "amount"],
        low_memory=False,
    )
    df = df[
        df["stay_id"].isin(valid_stays) & df["itemid"].isin(VASOPRESSOR_ITEMS)
    ].copy()
    log.info(f"  Tìm thấy {len(df):,} lệnh thuốc vận mạch")

    if df.empty:
        return pd.DataFrame(columns=["AdmissionID", "hour_bucket"] + vaso_cols)

    df["starttime"]  = pd.to_datetime(df["starttime"])
    df["endtime"]    = pd.to_datetime(df["endtime"])
    df["icu_intime"] = df["stay_id"].map(stay2intime)
    df = df.dropna(subset=["icu_intime"])
    df["start_h"] = ((df["starttime"] - df["icu_intime"]).dt.total_seconds() / 3600).clip(0, hours - 1).astype(int)
    df["end_h"]   = ((df["endtime"]   - df["icu_intime"]).dt.total_seconds() / 3600).clip(0, hours - 1).astype(int)
    df["VarName"] = df["itemid"].map(VASOPRESSOR_ITEMS)

    rows = []
    for _, r in df.iterrows():
        adm_id = adm_map.get(r["stay_id"])
        if adm_id is None:
            continue
        for h in range(int(r["start_h"]), min(int(r["end_h"]) + 1, hours)):
            rows.append({"AdmissionID": adm_id, "hour_bucket": h,
                         "VarName": r["VarName"] + "_used", "val": 1})

    if not rows:
        return pd.DataFrame(columns=["AdmissionID", "hour_bucket"] + vaso_cols)

    wide = (
        pd.DataFrame(rows)
        .groupby(["AdmissionID", "hour_bucket", "VarName"])["val"]
        .max().reset_index()
        .pivot_table(index=["AdmissionID", "hour_bucket"],
                     columns="VarName", values="val", fill_value=0)
        .reset_index()
    )
    wide.columns.name = None
    for col in vaso_cols:
        if col not in wide.columns:
            wide[col] = 0

    log.info(f"  Vasopressor wide: {wide.shape}")
    return wide[["AdmissionID", "hour_bucket"] + vaso_cols]


# ══════════════════════════════════════════════════════════════════════
#  STEP 5 — NƯỚC TIỂU & CRRT
# ══════════════════════════════════════════════════════════════════════
def process_urine_crrt(data_dir, cohort, hours=48):
    log.info("=" * 60)
    log.info("STEP 5: Nước tiểu (outputevents) + CRRT (procedureevents)")
    log.info("=" * 60)

    stay2intime = cohort.set_index("StayID")["ICUInTime"].to_dict()
    adm_map     = cohort.set_index("StayID")["AdmissionID"].to_dict()
    valid_stays = set(cohort["StayID"].tolist())

    # Nước tiểu
    udf = pd.read_csv(
        os.path.join(data_dir, "icu", "outputevents.csv"),
        usecols=["stay_id", "charttime", "itemid", "value"],
        low_memory=False,
    )
    udf = udf[udf["stay_id"].isin(valid_stays) & udf["itemid"].isin(URINE_ITEMS)].copy()
    udf["charttime"]   = pd.to_datetime(udf["charttime"])
    udf["icu_intime"]  = udf["stay_id"].map(stay2intime)
    udf["hours_in"]    = (udf["charttime"] - udf["icu_intime"]).dt.total_seconds() / 3600
    udf = udf[(udf["hours_in"] >= 0) & (udf["hours_in"] < hours)]
    udf["hour_bucket"] = udf["hours_in"].astype(int).clip(0, hours - 1)
    udf["AdmissionID"] = udf["stay_id"].map(adm_map)
    udf["value"]       = pd.to_numeric(udf["value"], errors="coerce")
    urine_wide = (
        udf.groupby(["AdmissionID", "hour_bucket"])["value"]
        .sum().reset_index().rename(columns={"value": "Urine_output_mL"})
    )
    log.info(f"  Urine rows: {len(urine_wide):,}")

    # CRRT
    pdf = pd.read_csv(
        os.path.join(data_dir, "icu", "procedureevents.csv"),
        usecols=["stay_id", "starttime", "endtime", "itemid"],
        low_memory=False,
    )
    pdf = pdf[pdf["stay_id"].isin(valid_stays) & (pdf["itemid"] == CRRT_ITEM)].copy()
    log.info(f"  CRRT sessions: {len(pdf):,}")
    rows = []
    if not pdf.empty:
        pdf["starttime"]  = pd.to_datetime(pdf["starttime"])
        pdf["endtime"]    = pd.to_datetime(pdf["endtime"])
        pdf["icu_intime"] = pdf["stay_id"].map(stay2intime)
        pdf = pdf.dropna(subset=["icu_intime"])
        pdf["start_h"] = ((pdf["starttime"] - pdf["icu_intime"]).dt.total_seconds() / 3600).clip(0, hours - 1).astype(int)
        pdf["end_h"]   = ((pdf["endtime"]   - pdf["icu_intime"]).dt.total_seconds() / 3600).clip(0, hours - 1).astype(int)
        for _, r in pdf.iterrows():
            adm_id = adm_map.get(r["stay_id"])
            if adm_id is None:
                continue
            for h in range(int(r["start_h"]), min(int(r["end_h"]) + 1, hours)):
                rows.append({"AdmissionID": adm_id, "hour_bucket": h, "CRRT_active": 1})

    crrt_wide = (
        pd.DataFrame(rows).groupby(["AdmissionID", "hour_bucket"])["CRRT_active"]
        .max().reset_index()
        if rows else pd.DataFrame(columns=["AdmissionID", "hour_bucket", "CRRT_active"])
    )

    result = pd.merge(urine_wide, crrt_wide, on=["AdmissionID", "hour_bucket"], how="outer")
    if "Urine_output_mL" not in result.columns:
        result["Urine_output_mL"] = np.nan
    if "CRRT_active" not in result.columns:
        result["CRRT_active"] = np.nan

    log.info(f"  Urine+CRRT wide: {result.shape}")
    return result


# ══════════════════════════════════════════════════════════════════════
#  STEP 6 — MERGE, IMPUTE, SPLIT, SCALE, SAVE
# ══════════════════════════════════════════════════════════════════════
def build_sequences(chart, lab, vaso, urine_crrt, cohort,
                    hours=48, out_dir="output",
                    test_size=0.2, val_size=0.1):
    log.info("=" * 60)
    log.info("STEP 6: Merge → SPLIT → Impute → Scale → Tensor")
    log.info("  Thứ tự đúng: split TRƯỚC, học tham số TRÊN TRAIN")
    log.info("=" * 60)

    # ── 6a. Merge raw data ────────────────────────────────────────────
    log.info("  [6a] Merge 4 bảng raw...")
    merged = (
        chart
        .merge(lab,        on=["AdmissionID", "hour_bucket"], how="outer")
        .merge(vaso,       on=["AdmissionID", "hour_bucket"], how="outer")
        .merge(urine_crrt, on=["AdmissionID", "hour_bucket"], how="outer")
    )
    # Binary treatment features: không đo = không dùng = 0
    for col in ["Epinephrine_used", "Norepinephrine_used",
                "Dobutamine_used", "Dopamine_used", "CRRT_active"]:
        if col in merged.columns:
            merged[col] = merged[col].fillna(0)

    # Full time grid: mỗi admission × đủ `hours` giờ
    all_adm   = cohort["AdmissionID"].unique()
    full_grid = pd.MultiIndex.from_product(
        [all_adm, range(hours)], names=["AdmissionID", "hour_bucket"]
    ).to_frame(index=False)
    df = full_grid.merge(merged, on=["AdmissionID", "hour_bucket"], how="left")
    df = df.sort_values(["AdmissionID", "hour_bucket"]).reset_index(drop=True)
    log.info(f"  Full grid shape: {df.shape}")

    feat_cols = ALL_FEATURES

    # ── 6b. SPLIT TRƯỚC — đây là bước quan trọng nhất ────────────────
    # Chia theo PatientID (không phải AdmissionID) để tránh data leakage:
    # một bệnh nhân nhập ICU nhiều lần → tất cả lần đó vào cùng 1 tập.
    log.info("  [6b] STRATIFIED SPLIT theo PatientID...")
    log.info("       (giữ tỷ lệ tử vong cân bằng giữa train/val/test)")
    # Tạo label per-patient: nếu bệnh nhân chết ở BẤT KỲ lần nhập ICU nào → 1
    pid_outcome = cohort.groupby("PatientID")["Outcome"].max()
    unique_pids = pid_outcome.index.values
    pid_labels  = pid_outcome.values  # 0 hoặc 1

    # Stratified split: duy trì tỷ lệ mortality nhất quán
    train_pids, test_pids, train_y, _ = train_test_split(
        unique_pids, pid_labels,
        test_size=test_size, random_state=42, stratify=pid_labels
    )
    train_pids, val_pids, _, _ = train_test_split(
        train_pids, train_y,
        test_size=val_size / (1 - test_size), random_state=42, stratify=train_y
    )
    log.info(f"  Patients — Train: {len(train_pids):,} | "
             f"Val: {len(val_pids):,} | Test: {len(test_pids):,}")

    adm2pid     = cohort.set_index("AdmissionID")["PatientID"].to_dict()
    adm2outcome = cohort.set_index("AdmissionID")["Outcome"].to_dict()
    train_adm = {a for a, p in adm2pid.items() if p in set(train_pids)}
    val_adm   = {a for a, p in adm2pid.items() if p in set(val_pids)}
    test_adm  = {a for a, p in adm2pid.items() if p in set(test_pids)}
    log.info(f"  Admissions — Train: {len(train_adm):,} | "
             f"Val: {len(val_adm):,} | Test: {len(test_adm):,}")

    # Tạo mask boolean để dùng nhiều lần
    train_mask = df["AdmissionID"].isin(train_adm)
    val_mask   = df["AdmissionID"].isin(val_adm)
    test_mask  = df["AdmissionID"].isin(test_adm)

    # ── 6c. ffill/bfill PER PATIENT ───────────────────────────────────
    # An toàn với data leakage vì:
    #   - Chỉ lan truyền giá trị trong chuỗi 48h của CHÍNH bệnh nhân đó
    #   - Không nhìn sang bệnh nhân khác, không dùng thống kê toàn tập
    #   - Áp dụng như nhau cho cả 3 tập (không học gì từ train)
    #
    # Dùng groupby().transform() thay vì .apply() để tránh lỗi pandas 2.0+
    # (.apply() đôi khi promote group key lên index, làm mất cột AdmissionID)
    # ── 6c. MISSING MASK — đánh dấu dữ liệu thật vs dữ liệu điền ────
    # Kỹ thuật đặc thù y tế ICU: việc "không đo" là thông tin lâm sàng
    # quan trọng (bác sĩ không đo SpO2 vì bệnh nhân ổn định).
    # Tạo mask TRƯỚC khi impute để mô hình DL biết ô nào là thật/giả.
    # Ref: GRU-D paper (Che et al., 2018)
    log.info("  [6c] Tạo Missing Mask (trước khi impute)...")
    df = df.sort_values(["AdmissionID", "hour_bucket"]).reset_index(drop=True)

    mask_cols = [f"{col}_mask" for col in feat_cols]
    for col in feat_cols:
        df[f"{col}_mask"] = (~df[col].isna()).astype(np.float32)
    log.info(f"  Missing mask: {len(mask_cols)} cột (1=đo thật, 0=sẽ điền)")

    # ── 6d. LINEAR INTERPOLATION per-patient ──────────────────────────
    # Thay vì ffill/bfill (đường ngang phẳng), dùng nội suy tuyến tính
    # → tạo đường dốc tự nhiên giữa 2 điểm đo: y tế hợp lý hơn.
    # Ví dụ: HR(7h)=80, HR(9h)=100 → HR(8h)=90 thay vì 80 (ffill)
    # Sau interpolation, dùng ffill+bfill cho đầu/cuối chuỗi.
    log.info("  [6d] Linear Interpolation per-patient (mượt hơn ffill)...")
    for col in feat_cols:
        # Bước 1: Nội suy tuyến tính giữa các điểm đo thật
        df[col] = df.groupby("AdmissionID")[col].transform(
            lambda s: s.interpolate(method="linear", limit_direction="both")
        )
        # Bước 2: ffill + bfill cho đầu/cuối chuỗi (interpolate không xử lý biên)
        df[col] = df.groupby("AdmissionID")[col].transform(
            lambda s: s.ffill().bfill()
        )

    # Tái tạo mask
    train_mask = df["AdmissionID"].isin(train_adm)
    val_mask   = df["AdmissionID"].isin(val_adm)
    test_mask  = df["AdmissionID"].isin(test_adm)

    # ── 6e. Tính median TRÊN TRAIN → fill NaN còn lại ─────────────────
    # NaN còn lại = bệnh nhân không có BẤT KỲ đo lường nào cho feature đó
    # trong 48h. Fill bằng giá trị "bình thường" từ train.
    log.info("  [6e] Train median fill (NaN còn lại sau interpolation)...")
    train_median = df.loc[train_mask, feat_cols].median()

    nan_train = df.loc[train_mask, feat_cols].isna().sum().sum()
    nan_val   = df.loc[val_mask,   feat_cols].isna().sum().sum()
    nan_test  = df.loc[test_mask,  feat_cols].isna().sum().sum()
    log.info(f"  NaN trước median fill — "
             f"Train: {nan_train:,} | Val: {nan_val:,} | Test: {nan_test:,}")

    df.loc[train_mask, feat_cols] = df.loc[train_mask, feat_cols].fillna(train_median)
    df.loc[val_mask,   feat_cols] = df.loc[val_mask,   feat_cols].fillna(train_median)
    df.loc[test_mask,  feat_cols] = df.loc[test_mask,  feat_cols].fillna(train_median)

    nan_after = df[feat_cols].isna().sum().sum()
    log.info(f"  NaN sau median fill: {nan_after:,}  (phải = 0)")

    pd.Series(train_median).to_csv(
        os.path.join(out_dir, "train_median.csv"), header=["median"]
    )

    # ── 6f. StandardScaler: fit TRÊN TRAIN, transform tất cả ──────────
    # Đây là nguyên tắc cốt lõi: scaler "học" mean/std từ train,
    # ép val/test phải dùng cùng thước đo đó.
    log.info("  [6f] StandardScaler: fit trên train → transform val + test...")
    scaler = StandardScaler()
    df.loc[train_mask, feat_cols] = scaler.fit_transform(
        df.loc[train_mask, feat_cols]
    )
    df.loc[val_mask,  feat_cols] = scaler.transform(
        df.loc[val_mask,  feat_cols]
    )
    df.loc[test_mask, feat_cols] = scaler.transform(
        df.loc[test_mask, feat_cols]
    )
    log.info(f"  Scaler mean[:3]: {scaler.mean_[:3].round(3).tolist()}")
    log.info(f"  Scaler std[:3] : {scaler.scale_[:3].round(3).tolist()}")
    joblib.dump(scaler, os.path.join(out_dir, "scaler.pkl"))

    with open(os.path.join(out_dir, "feature_names.json"), "w", encoding="utf-8") as f:
        json.dump({
            "features": feat_cols, "n_total": len(feat_cols),
            "n_chart": len(CHART_FEATURES), "n_lab": len(LAB_FEATURES),
            "n_treatment": len(TREATMENT_FEATURES),
            "chart": CHART_FEATURES, "lab": LAB_FEATURES,
            "treatment": TREATMENT_FEATURES,
            "mask_cols": mask_cols,
            "note": "M_*.pt chứa missing mask (1=đo thật, 0=đã điền). "
                    "Concat X+M → input (N, 48, F*2) nếu muốn dùng masking.",
        }, f, ensure_ascii=False, indent=2)

    try:
        import torch; use_torch = True
    except ImportError:
        use_torch = False

    def _save_split(adm_set, name):
        sub = df[df["AdmissionID"].isin(adm_set)]
        X_list, M_list, y_list, adm_list = [], [], [], []
        for adm_id, grp in sub.groupby("AdmissionID"):
            grp = grp.sort_values("hour_bucket")
            seq  = grp[feat_cols].values      # (hours, F) — features
            mask = grp[mask_cols].values       # (hours, F) — missing mask
            if seq.shape[0] < hours:
                pad_len = hours - seq.shape[0]
                seq  = np.vstack([seq,  np.zeros((pad_len, len(feat_cols)))])
                mask = np.vstack([mask, np.zeros((pad_len, len(feat_cols)))])
            else:
                seq  = seq[:hours]
                mask = mask[:hours]
            X_list.append(seq)
            M_list.append(mask)
            y_list.append(adm2outcome.get(adm_id, 0))
            adm_list.append(adm_id)

        X = np.array(X_list, dtype=np.float32)    # (N, 48, F)
        M = np.array(M_list, dtype=np.float32)    # (N, 48, F) — missing mask
        y = np.array(y_list, dtype=np.float32)    # (N,)
        nan_X = np.isnan(X).sum()
        log.info(f"  {name:5s}: N={len(X):,}  shape={X.shape}  "
                 f"mortality={y.mean()*100:.1f}%  NaN={nan_X}")

        if use_torch:
            import torch
            torch.save(torch.tensor(X), os.path.join(out_dir, f"X_{name}.pt"))
            torch.save(torch.tensor(M), os.path.join(out_dir, f"M_{name}.pt"))
            torch.save(torch.tensor(y), os.path.join(out_dir, f"y_{name}.pt"))
        else:
            np.save(os.path.join(out_dir, f"X_{name}.npy"), X)
            np.save(os.path.join(out_dir, f"M_{name}.npy"), M)
            np.save(os.path.join(out_dir, f"y_{name}.npy"), y)
        pd.Series(adm_list, name="AdmissionID").to_csv(
            os.path.join(out_dir, f"adm_ids_{name}.csv"), index=False)

    _save_split(train_adm, "train")
    _save_split(val_adm,   "val")
    _save_split(test_adm,  "test")
    log.info(f"  ✓ Tensor shape: (N, {hours}, {len(feat_cols)})")


# ══════════════════════════════════════════════════════════════════════
#  STEP 7 — ICD (optional)
# ══════════════════════════════════════════════════════════════════════
def process_icd(data_dir, cohort, out_dir):
    log.info("=" * 60)
    log.info("STEP 7: ICD codes")
    log.info("=" * 60)
    valid_adm = set(cohort["AdmissionID"].tolist())
    for fname in ["diagnoses_icd", "procedures_icd"]:
        path = os.path.join(data_dir, "hosp", f"{fname}.csv")
        if not os.path.exists(path):
            log.warning(f"  {fname}.csv không tìm thấy, bỏ qua.")
            continue
        df = pd.read_csv(path, low_memory=False)
        df = df.rename(columns={"subject_id": "PatientID", "hadm_id": "AdmissionID"})
        df = df[df["AdmissionID"].isin(valid_adm)]
        out_path = os.path.join(out_dir, f"mimic4_{fname}.parquet")
        df.to_parquet(out_path, index=False)
        log.info(f"  {fname}: {len(df):,} rows → {out_path}")


# ══════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",   required=True)
    parser.add_argument("--out_dir",    required=True)
    parser.add_argument("--hours",      type=int,   default=48)
    parser.add_argument("--test_size",  type=float, default=0.2)
    parser.add_argument("--val_size",   type=float, default=0.1)
    parser.add_argument("--skip_chart", action="store_true")
    parser.add_argument("--skip_lab",   action="store_true")
    parser.add_argument("--skip_vaso",  action="store_true")
    parser.add_argument("--skip_urine", action="store_true")
    parser.add_argument("--icd",        action="store_true")
    args = parser.parse_args()

    global log
    log = setup_logging(args.out_dir)

    log.info("=" * 60)
    log.info("MIMIC-IV Preprocessing — BVQ11 v2")
    log.info("=" * 60)
    log.info(f"Data dir : {args.data_dir}")
    log.info(f"Output   : {args.out_dir}")
    log.info(f"Window   : {args.hours}h  |  Features: {len(ALL_FEATURES)}")
    log.info(f"  {len(CHART_FEATURES)} chart + {len(LAB_FEATURES)} lab + "
             f"{len(TREATMENT_FEATURES)} điều trị")

    t0 = datetime.now()

    cohort = build_cohort(args.data_dir)
    cohort.to_parquet(os.path.join(args.out_dir, "cohort.parquet"), index=False)

    def _load_or_run(name, fn, cache_file, skip_flag, *fn_args):
        cache = os.path.join(args.out_dir, cache_file)
        if skip_flag and os.path.exists(cache):
            log.info(f"{name}: Load từ cache ({cache_file})")
            result = pd.read_parquet(cache)
            log.info(f"  Cache shape: {result.shape}")
            return result
        result = fn(*fn_args)
        result.to_parquet(cache, index=False)
        log.info(f"  Cache lưu: {cache}")
        return result

    chart      = _load_or_run("STEP 2", process_chartevents, "chart_wide.parquet",
                               args.skip_chart, args.data_dir, cohort, args.hours)
    lab        = _load_or_run("STEP 3", process_labevents,   "lab_wide.parquet",
                               args.skip_lab,   args.data_dir, cohort, args.hours)
    vaso       = _load_or_run("STEP 4", process_vasopressors,"vaso_wide.parquet",
                               args.skip_vaso,  args.data_dir, cohort, args.hours)
    urine_crrt = _load_or_run("STEP 5", process_urine_crrt,  "urine_crrt.parquet",
                               args.skip_urine, args.data_dir, cohort, args.hours)

    build_sequences(chart, lab, vaso, urine_crrt, cohort,
                    hours=args.hours, out_dir=args.out_dir,
                    test_size=args.test_size, val_size=args.val_size)

    if args.icd:
        process_icd(args.data_dir, cohort, args.out_dir)

    elapsed = datetime.now() - t0
    h, rem  = divmod(int(elapsed.total_seconds()), 3600)
    m, s    = divmod(rem, 60)

    log.info("")
    log.info("=" * 60)
    log.info(f"HOÀN TẤT — Thời gian: {h}h {m}m {s}s")
    log.info("=" * 60)
    log.info(f"Kết quả: {args.out_dir}")
    log.info("  X_train/val/test.pt    — features tensor (N, 48, 69)")
    log.info("  M_train/val/test.pt    — missing mask (N, 48, 69) [1=thật, 0=điền]")
    log.info("  y_train/val/test.pt    — labels")
    log.info("  scaler.pkl             — StandardScaler")
    log.info("  feature_names.json     — 69 features")
    log.info("  preprocessing_*.log    — log chi tiết")


if __name__ == "__main__":
    main()
