"""
=============================================================
  Nén các file CSV MIMIC-IV thành Parquet
  Giảm dung lượng ~70-80% để upload Colab nhanh hơn
=============================================================

Cách chạy trên Windows:
    python csv_to_parquet.py --data_dir "E:\KLTN\mimiciv\3.1"

Cấu trúc đầu vào:
    E:\KLTN\mimiciv\3.1\
    ├── hosp\
    │   ├── patients.csv
    │   ├── admissions.csv
    │   ├── labevents.csv       
    │   ├── diagnoses_icd.csv
    │   ├── procedures_icd.csv
    │   ├── d_labitems.csv
    │   └── d_icd_diagnoses.csv
    └── icu\
        ├── icustays.csv
        ├── chartevents.csv      
        ├── inputevents.csv
        ├── outputevents.csv
        ├── procedureevents.csv
        └── d_items.csv

Kết quả đầu ra:
    E:\KLTN\mimiciv\3.1\parquet\
    ├── hosp\
    │   ├── patients.parquet
    │   └── ...
    └── icu\
        ├── icustays.parquet
        └── ...
"""

import os
import sys
import argparse
import time
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path

def convert_one(csv_path: str, parquet_path: str, chunksize: int = 5_000_000):
    """Chuyển 1 file CSV sang Parquet. File lớn ghi từng chunk (không gom vào RAM)."""
    os.makedirs(os.path.dirname(parquet_path), exist_ok=True)
    
    file_size = os.path.getsize(csv_path) / (1024**3)  # GB
    total_rows = 0
    
    if file_size > 1.0:
        # File lớn: đọc chunk → ghi thẳng ra parquet (KHÔNG concat vào RAM)
        print(f"  [LARGE {file_size:.1f} GB] Ghi trực tiếp từng chunk → parquet...")
        writer = None
        
        for i, chunk in enumerate(pd.read_csv(csv_path, low_memory=False, chunksize=chunksize)):
            table = pa.Table.from_pandas(chunk, preserve_index=False)
            
            if writer is None:
                writer = pq.ParquetWriter(parquet_path, table.schema)
            
            writer.write_table(table)
            total_rows += len(chunk)
            print(f"    Chunk {i+1}: {len(chunk):,} rows  (tổng: {total_rows:,})")
            del table, chunk  # giải phóng RAM ngay
        
        if writer is not None:
            writer.close()
    else:
        df = pd.read_csv(csv_path, low_memory=False)
        total_rows = len(df)
        df.to_parquet(parquet_path, engine="pyarrow", index=False)
        del df
    
    parquet_size = os.path.getsize(parquet_path) / (1024**2)  # MB
    csv_size     = file_size * 1024  # MB
    ratio        = (1 - parquet_size / (csv_size + 0.001)) * 100
    
    print(f"  ✓ {os.path.basename(csv_path)}: "
          f"{csv_size:.0f} MB → {parquet_size:.0f} MB  ({ratio:.0f}% giảm)  "
          f"({total_rows:,} rows)")
    return total_rows

def main():
    parser = argparse.ArgumentParser(description="Nén CSV MIMIC-IV → Parquet")
    parser.add_argument("--data_dir", required=True,
                        help="Thư mục gốc MIMIC-IV (chứa hosp/ và icu/)")
    parser.add_argument("--out_dir", default=None,
                        help="Thư mục output (mặc định: data_dir/parquet)")
    args = parser.parse_args()
    
    data_dir = args.data_dir
    out_dir  = args.out_dir or os.path.join(data_dir, "parquet")
    
    # Danh sách các file cần nén (thứ tự từ nhỏ → lớn)
    files = [
        # hosp — nhỏ trước
        ("hosp", "patients.csv"),
        ("hosp", "admissions.csv"),
        ("hosp", "d_labitems.csv"),
        ("hosp", "d_icd_diagnoses.csv"),
        ("hosp", "d_icd_procedures.csv"),
        ("hosp", "diagnoses_icd.csv"),
        ("hosp", "procedures_icd.csv"),
        ("hosp", "prescriptions.csv"),
        ("hosp", "labevents.csv"),          # ~6 GB
        # icu — nhỏ trước
        ("icu", "d_items.csv"),
        ("icu", "icustays.csv"),
        ("icu", "procedureevents.csv"),
        ("icu", "outputevents.csv"),
        ("icu", "inputevents.csv"),
        ("icu", "chartevents.csv"),         # ~30 GB — cuối cùng
    ]
    
    print("=" * 60)
    print("  Nén CSV MIMIC-IV → Parquet")
    print(f"  Input : {data_dir}")
    print(f"  Output: {out_dir}")
    print("=" * 60)
    
    total_start  = time.time()
    total_rows   = 0
    success      = 0
    skipped      = 0
    
    for folder, filename in files:
        csv_path     = os.path.join(data_dir, folder, filename)
        parquet_name = filename.replace(".csv", ".parquet")
        parquet_path = os.path.join(out_dir, folder, parquet_name)
        
        if not os.path.exists(csv_path):
            # Thử tìm file .csv.gz
            gz_path = csv_path + ".gz"
            if os.path.exists(gz_path):
                csv_path = gz_path
            else:
                print(f"  [SKIP] {folder}/{filename} — không tìm thấy")
                skipped += 1
                continue
        
        if os.path.exists(parquet_path):
            print(f"  [SKIP] {folder}/{parquet_name} — đã tồn tại")
            skipped += 1
            continue
        
        print(f"\n  Converting: {folder}/{filename}...")
        t0 = time.time()
        try:
            rows = convert_one(csv_path, parquet_path)
            total_rows += rows
            success += 1
            elapsed = time.time() - t0
            print(f"  Thời gian: {elapsed:.0f}s")
        except Exception as e:
            print(f"  [ERROR] {e}")
    
    total_elapsed = time.time() - total_start
    m, s = divmod(int(total_elapsed), 60)
    
    print("\n" + "=" * 60)
    print(f"  XONG! {success} files converted, {skipped} skipped")
    print(f"  Tổng rows: {total_rows:,}")
    print(f"  Thời gian: {m}m {s}s")
    print(f"  Output: {out_dir}")
    print("=" * 60)
    
    # Tính tổng dung lượng
    total_size = 0
    for root, dirs, fnames in os.walk(out_dir):
        for f in fnames:
            if f.endswith(".parquet"):
                total_size += os.path.getsize(os.path.join(root, f))
    print(f"  Tổng dung lượng parquet: {total_size / (1024**3):.2f} GB")


if __name__ == "__main__":
    main()
