"""
BƯỚC 1: Thu thập dữ liệu EPL từ football-data.co.uk
=====================================================
Chạy: python collect_data.py
Output: data/epl_raw.csv (gộp tất cả mùa)
        data/epl_seasons/  (từng mùa riêng lẻ)
"""

import requests
import pandas as pd
import os
import time

# ── Cấu hình ──────────────────────────────────────────────────────────────────
SEASONS = [
    "1516", "1617", "1718", "1819", "1920",
    "2021", "2122", "2223", "2324", "2425", 
]

BASE_URL = "https://www.football-data.co.uk/mmz4281/{season}/E0.csv"

# Các cột cần giữ lại (bỏ qua cột tỉ lệ cược)
KEEP_COLS = [
    "Div",        # Division (E0 = EPL)
    "Date",       # Ngày thi đấu
    "HomeTeam",   # Đội nhà
    "AwayTeam",   # Đội khách
    "FTHG",       # Full Time Home Goals
    "FTAG",       # Full Time Away Goals
    "FTR",        # Full Time Result (H/D/A)  ← LABEL
    "HTHG",       # Half Time Home Goals
    "HTAG",       # Half Time Away Goals
    "HTR",        # Half Time Result
    "HS",         # Home Shots
    "AS",         # Away Shots
    "HST",        # Home Shots on Target
    "AST",        # Away Shots on Target
    "HC",         # Home Corners
    "AC",         # Away Corners
    "HY",         # Home Yellow Cards
    "AY",         # Away Yellow Cards
    "HR",         # Home Red Cards
    "AR",         # Away Red Cards
]

OUTPUT_DIR = "data"
SEASONS_DIR = "data/epl_seasons"


# ── Hàm download từng mùa ─────────────────────────────────────────────────────
def download_season(season: str) -> pd.DataFrame | None:
    url = BASE_URL.format(season=season)
    label = f"20{season[:2]}/{season[2:]}"  # vd: "2015/16"

    try:
        response = requests.get(url, timeout=15)
        response.raise_for_status()

        # Đọc CSV từ content trả về
        from io import StringIO
        df = pd.read_csv(StringIO(response.text), encoding="utf-8", on_bad_lines="skip")

        # Chỉ giữ các cột cần thiết (bỏ cột không có)
        available = [c for c in KEEP_COLS if c in df.columns]
        df = df[available].copy()

        # Thêm cột mùa giải
        df["Season"] = label

        # Bỏ hàng không có kết quả (cuối file hay có hàng trống)
        df = df.dropna(subset=["FTR", "HomeTeam", "AwayTeam"])

        print(f"  ✅ {label}: {len(df)} trận")
        return df

    except requests.exceptions.HTTPError:
        print(f"  ⚠️  {label}: Chưa có dữ liệu (mùa chưa bắt đầu hoặc URL sai)")
        return None
    except Exception as e:
        print(f"  ❌ {label}: Lỗi - {e}")
        return None


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(SEASONS_DIR, exist_ok=True)

    print("=" * 55)
    print("  EPL Data Collector — football-data.co.uk")
    print("=" * 55)

    all_seasons = []

    for season in SEASONS:
        df = download_season(season)

        if df is not None and len(df) > 0:
            # Lưu từng mùa riêng
            label = f"20{season[:2]}_{season[2:]}"
            path = f"{SEASONS_DIR}/epl_{label}.csv"
            df.to_csv(path, index=False)

            all_seasons.append(df)

        time.sleep(0.5)  # Tránh spam server

    if not all_seasons:
        print("\n❌ Không download được dữ liệu nào!")
        return

    # Gộp tất cả mùa
    df_all = pd.concat(all_seasons, ignore_index=True)

    # Parse ngày tháng
    df_all["Date"] = pd.to_datetime(df_all["Date"], format="mixed", dayfirst=True, errors="coerce")

    # Sắp xếp theo thời gian
    df_all = df_all.sort_values("Date").reset_index(drop=True)

    # Lưu file gộp
    raw_path = f"{OUTPUT_DIR}/epl_raw.csv"
    df_all.to_csv(raw_path, index=False)

    # ── Thống kê ──────────────────────────────────────────────────────────────
    print()
    print("=" * 55)
    print("  📊 THỐNG KÊ DỮ LIỆU ĐÃ DOWNLOAD")
    print("=" * 55)
    print(f"  Tổng số trận    : {len(df_all):,}")
    print(f"  Số mùa giải     : {df_all['Season'].nunique()}")
    print(f"  Số đội          : {df_all['HomeTeam'].nunique()}")
    print(f"  Từ ngày         : {df_all['Date'].min().date()}")
    print(f"  Đến ngày        : {df_all['Date'].max().date()}")
    print()
    print("  Phân bổ kết quả:")
    result_counts = df_all["FTR"].value_counts()
    total = len(df_all)
    for label, count in result_counts.items():
        name = {"H": "Home Win", "D": "Draw", "A": "Away Win"}[label]
        pct = count / total * 100
        bar = "█" * int(pct / 2)
        print(f"    {name:12s}: {count:4d} ({pct:.1f}%) {bar}")
    print()
    print(f"  ✅ Đã lưu: {raw_path}")
    print(f"  ✅ Từng mùa: {SEASONS_DIR}/")
    print("=" * 55)
    print()
    print("  👉 Bước tiếp theo: chạy clean_data.py")
    print("=" * 55)


if __name__ == "__main__":
    main()