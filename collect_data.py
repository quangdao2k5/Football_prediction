"""
BƯỚC 1: Thu thập dữ liệu EPL từ football-data.co.uk
=====================================================
Chạy: python collect_data.py
      python collect_data.py --current-only
Output: data/epl_raw.csv (gộp tất cả mùa)
        data/epl_seasons/  (từng mùa riêng lẻ)
"""

import argparse
import requests
import pandas as pd
import os
import time

# ── Cấu hình ──────────────────────────────────────────────────────────────────
SEASONS = [
    "1516", "1617", "1718", "1819", "1920",
    "2021", "2122", "2223", "2324", "2425",
    "2526",
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
RAW_PATH = f"{OUTPUT_DIR}/epl_raw.csv"


def season_label(season: str) -> str:
    return f"20{season[:2]}/{season[2:]}"


def season_file_label(season: str) -> str:
    return f"20{season[:2]}_{season[2:]}"


def save_season_file(season: str, df: pd.DataFrame) -> str:
    label = season_file_label(season)
    path = f"{SEASONS_DIR}/epl_{label}.csv"
    df.to_csv(path, index=False)
    return path


def print_dataset_stats(df_all: pd.DataFrame, raw_path: str):
    print()
    print("=" * 55)
    print("  THONG KE DU LIEU DA DOWNLOAD")
    print("=" * 55)
    print(f"  Tong so tran    : {len(df_all):,}")
    print(f"  So mua giai     : {df_all['Season'].nunique()}")
    print(f"  So doi          : {df_all['HomeTeam'].nunique()}")
    print(f"  Tu ngay         : {df_all['Date'].min().date()}")
    print(f"  Den ngay        : {df_all['Date'].max().date()}")
    print()
    print("  Phan bo ket qua:")
    result_counts = df_all["FTR"].value_counts()
    total = len(df_all)
    for label, count in result_counts.items():
        name = {"H": "Home Win", "D": "Draw", "A": "Away Win"}[label]
        pct = count / total * 100
        bar = "#" * int(pct / 2)
        print(f"    {name:12s}: {count:4d} ({pct:.1f}%) {bar}")
    print()
    print(f"  [DONE] Da luu: {raw_path}")
    print(f"  [DONE] Tung mua tai: {SEASONS_DIR}/")
    print("=" * 55)
    print()
    print("  Tiep theo: chay python clean_data.py")
    print("=" * 55)


# ── Hàm download từng mùa ─────────────────────────────────────────────────────
def download_season(season: str) -> pd.DataFrame | None:
    url = BASE_URL.format(season=season)
    label = season_label(season)  # vd: "2015/16"

    try:
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"}
        
        for attempt in range(3):
            try:
                response = requests.get(url, headers=headers, timeout=30)
                response.raise_for_status()
                break
            except Exception as e:
                if attempt == 2:
                    raise e
                time.sleep(2)

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

        # Parse date ngay cho tung mua (tranh loi khi gop mixed format)
        # football-data.co.uk dung dd/mm/yyyy hoac dd/mm/yy tuy mua
        df["Date"] = pd.to_datetime(df["Date"], format="mixed", dayfirst=True, errors="coerce")

        print(f"  [OK] {label}: {len(df)} tran")
        return df

    except requests.exceptions.HTTPError:
        print(f"  [WAIT] {label}: Chua co du lieu (mua chua bat dau hoac URL sai)")
        return None
    except Exception as e:
        print(f"  [ERROR] {label}: Loi - {e}")
        return None


def collect_full():
    """Download lai tat ca cac mua va ghi de epl_raw.csv."""
    all_seasons = []

    for season in SEASONS:
        df = download_season(season)

        if df is not None and len(df) > 0:
            save_season_file(season, df)
            all_seasons.append(df)

        time.sleep(0.5)  # Tranh spam server

    if not all_seasons:
        print("\n[ERROR] Khong download duoc du lieu nao!")
        return

    df_all = pd.concat(all_seasons, ignore_index=True)
    df_all = df_all.sort_values("Date").reset_index(drop=True)
    df_all.to_csv(RAW_PATH, index=False)
    print_dataset_stats(df_all, RAW_PATH)


def update_current_season(season: str):
    """
    Weekly update: chi download lai 1 mua, thay the phan mua do trong epl_raw.csv.
    Tranh tai lai toan bo lich su nhung van khong bi append trung tran.
    """
    if not os.path.exists(RAW_PATH):
        print(f"[ERROR] Chua co {RAW_PATH}. Hay chay full collect truoc:")
        print("  python collect_data.py")
        return

    target_label = season_label(season)
    print(f"  Che do current-only: chi cap nhat mua {target_label}")
    print()

    df_new = download_season(season)
    if df_new is None or len(df_new) == 0:
        print(f"\n[ERROR] Khong download duoc du lieu mua {target_label}.")
        return

    save_season_file(season, df_new)

    df_existing = pd.read_csv(RAW_PATH)
    df_existing["Date"] = pd.to_datetime(df_existing["Date"], errors="coerce")

    old_count = int((df_existing["Season"] == target_label).sum())
    df_keep = df_existing[df_existing["Season"] != target_label].copy()
    df_all = pd.concat([df_keep, df_new], ignore_index=True)
    df_all = df_all.sort_values("Date").reset_index(drop=True)
    df_all.to_csv(RAW_PATH, index=False)

    print()
    print("=" * 55)
    print("  CURRENT SEASON UPDATE")
    print("=" * 55)
    print(f"  Mua cap nhat     : {target_label}")
    print(f"  So tran cu       : {old_count}")
    print(f"  So tran moi      : {len(df_new)}")
    print(f"  Chenh lech       : {len(df_new) - old_count:+d}")
    print_dataset_stats(df_all, RAW_PATH)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Thu thap du lieu EPL tu football-data.co.uk"
    )
    parser.add_argument(
        "--current-only",
        action="store_true",
        help="Chi download lai mua hien tai va replace trong data/epl_raw.csv",
    )
    parser.add_argument(
        "--season",
        default=SEASONS[-1],
        choices=SEASONS,
        help="Mua can update khi dung --current-only, mac dinh la mua moi nhat",
    )
    return parser.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(SEASONS_DIR, exist_ok=True)

    print("=" * 55)
    print("  EPL Data Collector - football-data.co.uk")
    print("=" * 55)

    if args.current_only:
        update_current_season(args.season)
    else:
        collect_full()


if __name__ == "__main__":
    main()
