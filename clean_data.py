"""
BUOC 2: Lam sach du lieu va tao features (PHIEN BAN CAI THIEN)
================================================================
Input : data/epl_raw.csv
Output: data/epl_clean.csv

Chay: python clean_data.py
"""

import pandas as pd
import numpy as np
import os

INPUT_PATH  = "data/epl_raw.csv"
OUTPUT_PATH = "data/epl_clean.csv"

# So tran gan nhat de tinh form / averages
FORM_WINDOW = 5

# Elo config
ELO_K = 20        # He so cap nhat Elo
ELO_DEFAULT = 1500


# ── 1. Load du lieu ───────────────────────────────────────────────────────────

df = pd.read_csv(INPUT_PATH)
df["Date"] = pd.to_datetime(df["Date"], format="mixed", dayfirst=True, errors="coerce")
df = df.dropna(subset=["Date", "HomeTeam", "AwayTeam", "FTR"])
df = df.sort_values("Date").reset_index(drop=True)

print(f"Loaded: {len(df)} tran, {df['Season'].nunique()} mua")


# ── 2. Chuan hoa ten doi ──────────────────────────────────────────────────────

NAME_MAP = {
    "Man United"  : "Manchester United",
    "Man City"    : "Manchester City",
    "Newcastle"   : "Newcastle United",
    "Wolves"      : "Wolverhampton",
    "Sheffield United" : "Sheffield Utd",
    "Tottenham"   : "Tottenham",
    "Leeds"       : "Leeds United",
    "Leicester"   : "Leicester City",
    "Norwich"     : "Norwich City",
    "Stoke"       : "Stoke City",
    "Swansea"     : "Swansea City",
    "Hull"        : "Hull City",
    "West Brom"   : "West Bromwich",
    "QPR"         : "Queens Park Rangers",
}

df["HomeTeam"] = df["HomeTeam"].replace(NAME_MAP)
df["AwayTeam"] = df["AwayTeam"].replace(NAME_MAP)


# ── 3. Ham tinh form N tran gan nhat (RESET MOI MUA) ─────────────────────────
# Sua bug: Reset lich su moi dau mua giai moi

def compute_form(df: pd.DataFrame, n: int = FORM_WINDOW) -> tuple[list, list]:
    """
    Duyet tung tran theo thu tu thoi gian.
    Reset form khi bat dau mua moi.
    """
    history: dict[str, list] = {}
    current_season = None

    home_forms = []
    away_forms = []

    for _, row in df.iterrows():
        season = row["Season"]
        home = row["HomeTeam"]
        away = row["AwayTeam"]
        result = row["FTR"]

        # Reset form khi doi mua
        if season != current_season:
            history = {}
            current_season = season

        # Tinh form hien tai (truoc khi cap nhat tran nay)
        home_pts = history.get(home, [])
        away_pts = history.get(away, [])

        home_form = np.mean(home_pts[-n:]) if len(home_pts) >= 1 else 1.0
        away_form = np.mean(away_pts[-n:]) if len(away_pts) >= 1 else 1.0

        home_forms.append(home_form)
        away_forms.append(away_form)

        # Cap nhat lich su sau tran
        if result == "H":
            history.setdefault(home, []).append(3)
            history.setdefault(away, []).append(0)
        elif result == "D":
            history.setdefault(home, []).append(1)
            history.setdefault(away, []).append(1)
        else:
            history.setdefault(home, []).append(0)
            history.setdefault(away, []).append(3)

    return home_forms, away_forms


# ── 4. Ham tinh trung binh ban thang/thua gan nhat (RESET MOI MUA) ───────────

def compute_goal_averages(df: pd.DataFrame, n: int = FORM_WINDOW):
    scored_hist:   dict[str, list] = {}
    conceded_hist: dict[str, list] = {}
    current_season = None

    home_scored_avg    = []
    home_conceded_avg  = []
    away_scored_avg    = []
    away_conceded_avg  = []

    for _, row in df.iterrows():
        season = row["Season"]
        home = row["HomeTeam"]
        away = row["AwayTeam"]
        hg   = row["FTHG"]
        ag   = row["FTAG"]

        if season != current_season:
            scored_hist = {}
            conceded_hist = {}
            current_season = season

        hs = scored_hist.get(home, [])
        hc = conceded_hist.get(home, [])
        as_ = scored_hist.get(away, [])
        ac  = conceded_hist.get(away, [])

        home_scored_avg.append(np.mean(hs[-n:])  if hs  else 1.3)
        home_conceded_avg.append(np.mean(hc[-n:]) if hc  else 1.1)
        away_scored_avg.append(np.mean(as_[-n:]) if as_ else 1.1)
        away_conceded_avg.append(np.mean(ac[-n:]) if ac  else 1.3)

        scored_hist.setdefault(home, []).append(hg)
        conceded_hist.setdefault(home, []).append(ag)
        scored_hist.setdefault(away, []).append(ag)
        conceded_hist.setdefault(away, []).append(hg)

    return home_scored_avg, home_conceded_avg, away_scored_avg, away_conceded_avg


# ── 5. Ham tinh ti le thang lich su doi dau (H2H) ────────────────────────────

def compute_h2h(df: pd.DataFrame) -> list:
    h2h_hist: dict[tuple, list] = {}
    h2h_winrates = []

    for _, row in df.iterrows():
        home = row["HomeTeam"]
        away = row["AwayTeam"]
        key  = tuple(sorted([home, away]))

        past = h2h_hist.get(key, [])
        past_home = [r for (h, r) in past if h == home]

        if past_home:
            winrate = np.mean(past_home)
        else:
            winrate = 0.45

        h2h_winrates.append(winrate)

        if row["FTR"] == "H":
            val = 1.0
        elif row["FTR"] == "D":
            val = 0.5
        else:
            val = 0.0

        h2h_hist.setdefault(key, []).append((home, val))

    return h2h_winrates


# ── 6. Ham tinh ti le thang san nha tich luy ─────────────────────────────────

def compute_home_advantage(df: pd.DataFrame) -> list:
    home_record: dict[str, list] = {}
    advantages = []

    for _, row in df.iterrows():
        home = row["HomeTeam"]
        past = home_record.get(home, [])

        advantages.append(np.mean(past) if past else 0.45)

        if row["FTR"] == "H":
            home_record.setdefault(home, []).append(1.0)
        elif row["FTR"] == "D":
            home_record.setdefault(home, []).append(0.5)
        else:
            home_record.setdefault(home, []).append(0.0)

    return advantages


# ── 7. [MOI] Ham tinh Elo rating ──────────────────────────────────────────────
# Elo don gian: thang +K, thua -K, hoa +0. Gia tri tich luy qua tat ca mua.

def compute_elo(df: pd.DataFrame, k: float = ELO_K, default: float = ELO_DEFAULT):
    elo: dict[str, float] = {}
    home_elos = []
    away_elos = []

    for _, row in df.iterrows():
        home = row["HomeTeam"]
        away = row["AwayTeam"]

        h_elo = elo.get(home, default)
        a_elo = elo.get(away, default)

        home_elos.append(h_elo)
        away_elos.append(a_elo)

        # Expected score (logistic)
        exp_home = 1.0 / (1.0 + 10 ** ((a_elo - h_elo) / 400))
        exp_away = 1.0 - exp_home

        # Actual score
        if row["FTR"] == "H":
            s_home, s_away = 1.0, 0.0
        elif row["FTR"] == "D":
            s_home, s_away = 0.5, 0.5
        else:
            s_home, s_away = 0.0, 1.0

        elo[home] = h_elo + k * (s_home - exp_home)
        elo[away] = a_elo + k * (s_away - exp_away)

    return home_elos, away_elos


# ── 8. [MOI] Ham tinh shots/corners trung binh ───────────────────────────────

def compute_stat_averages(df: pd.DataFrame, home_col: str, away_col: str,
                          n: int = FORM_WINDOW, default: float = 5.0):
    """
    Tinh trung binh thong ke N tran gan nhat cho ca doi nha va doi khach.
    Tra ve: home_stat_avg, away_stat_avg
    """
    hist: dict[str, list] = {}
    current_season = None

    home_avgs = []
    away_avgs = []

    for _, row in df.iterrows():
        season = row["Season"]
        home = row["HomeTeam"]
        away = row["AwayTeam"]

        if season != current_season:
            hist = {}
            current_season = season

        h_hist = hist.get(home, [])
        a_hist = hist.get(away, [])

        home_avgs.append(np.mean(h_hist[-n:]) if h_hist else default)
        away_avgs.append(np.mean(a_hist[-n:]) if a_hist else default)

        # Cap nhat: moi doi nhan stat cua chinh minh
        hist.setdefault(home, []).append(row[home_col])
        hist.setdefault(away, []).append(row[away_col])

    return home_avgs, away_avgs


# ── 9. [MOI] Ham tinh rest days ──────────────────────────────────────────────

def compute_rest_days(df: pd.DataFrame):
    """So ngay nghi giua 2 tran lien tiep. Default = 7 neu chua co data."""
    last_match: dict[str, pd.Timestamp] = {}
    home_rest = []
    away_rest = []

    for _, row in df.iterrows():
        home = row["HomeTeam"]
        away = row["AwayTeam"]
        date = row["Date"]

        h_last = last_match.get(home)
        a_last = last_match.get(away)

        home_rest.append((date - h_last).days if h_last is not None else 7)
        away_rest.append((date - a_last).days if a_last is not None else 7)

        last_match[home] = date
        last_match[away] = date

    return home_rest, away_rest


# ── 10. [MOI] Ham tinh win/loss streak ────────────────────────────────────────

def compute_streak(df: pd.DataFrame):
    """
    Chuoi thang/thua hien tai.
    Duong = dang thang lien tiep, am = dang thua lien tiep, 0 = vua hoa.
    Reset moi mua.
    """
    streaks: dict[str, int] = {}
    current_season = None

    home_streaks = []
    away_streaks = []

    for _, row in df.iterrows():
        season = row["Season"]
        home = row["HomeTeam"]
        away = row["AwayTeam"]

        if season != current_season:
            streaks = {}
            current_season = season

        home_streaks.append(streaks.get(home, 0))
        away_streaks.append(streaks.get(away, 0))

        # Cap nhat streak sau tran
        if row["FTR"] == "H":
            streaks[home] = max(1, streaks.get(home, 0) + 1) if streaks.get(home, 0) >= 0 else 1
            streaks[away] = min(-1, streaks.get(away, 0) - 1) if streaks.get(away, 0) <= 0 else -1
        elif row["FTR"] == "A":
            streaks[home] = min(-1, streaks.get(home, 0) - 1) if streaks.get(home, 0) <= 0 else -1
            streaks[away] = max(1, streaks.get(away, 0) + 1) if streaks.get(away, 0) >= 0 else 1
        else:
            streaks[home] = 0
            streaks[away] = 0

    return home_streaks, away_streaks


# ── 11. [MOI] Ham tinh goal difference tich luy trong mua ────────────────────

def compute_season_gd(df: pd.DataFrame):
    """Hieu so ban thang - ban thua tich luy trong mua hien tai."""
    gd: dict[str, int] = {}
    current_season = None

    home_gds = []
    away_gds = []

    for _, row in df.iterrows():
        season = row["Season"]
        home = row["HomeTeam"]
        away = row["AwayTeam"]
        hg = int(row["FTHG"])
        ag = int(row["FTAG"])

        if season != current_season:
            gd = {}
            current_season = season

        home_gds.append(gd.get(home, 0))
        away_gds.append(gd.get(away, 0))

        gd[home] = gd.get(home, 0) + (hg - ag)
        gd[away] = gd.get(away, 0) + (ag - hg)

    return home_gds, away_gds


# ── 12. [MOI] Ham tinh season points tich luy ────────────────────────────────

def compute_season_points(df: pd.DataFrame):
    """Diem tich luy trong mua hien tai (3-1-0)."""
    pts: dict[str, int] = {}
    matches_played: dict[str, int] = {}
    current_season = None

    home_ppg = []  # points per game
    away_ppg = []

    for _, row in df.iterrows():
        season = row["Season"]
        home = row["HomeTeam"]
        away = row["AwayTeam"]

        if season != current_season:
            pts = {}
            matches_played = {}
            current_season = season

        h_pts = pts.get(home, 0)
        a_pts = pts.get(away, 0)
        h_mp = matches_played.get(home, 0)
        a_mp = matches_played.get(away, 0)

        home_ppg.append(h_pts / h_mp if h_mp > 0 else 1.5)
        away_ppg.append(a_pts / a_mp if a_mp > 0 else 1.5)

        if row["FTR"] == "H":
            pts[home] = h_pts + 3
            pts[away] = a_pts + 0
        elif row["FTR"] == "D":
            pts[home] = h_pts + 1
            pts[away] = a_pts + 1
        else:
            pts[home] = h_pts + 0
            pts[away] = a_pts + 3

        matches_played[home] = h_mp + 1
        matches_played[away] = a_mp + 1

    return home_ppg, away_ppg


# ── 13. [MOI] Ham tinh match number trong mua (season progress) ──────────────

def compute_season_progress(df: pd.DataFrame):
    """So tran da choi trong mua / 38 (tong so vong EPL)."""
    match_count: dict[str, int] = {}
    current_season = None

    home_progress = []
    away_progress = []

    for _, row in df.iterrows():
        season = row["Season"]
        home = row["HomeTeam"]
        away = row["AwayTeam"]

        if season != current_season:
            match_count = {}
            current_season = season

        home_progress.append(match_count.get(home, 0) / 38.0)
        away_progress.append(match_count.get(away, 0) / 38.0)

        match_count[home] = match_count.get(home, 0) + 1
        match_count[away] = match_count.get(away, 0) + 1

    return home_progress, away_progress


# ==============================================================================
# GAN FEATURES VAO DATAFRAME
# ==============================================================================

print("Dang tinh features...")

# --- Features cu (da sua bug reset mua) ---
df["home_form"],  df["away_form"]  = compute_form(df)

(df["home_scored_avg"],
 df["home_conceded_avg"],
 df["away_scored_avg"],
 df["away_conceded_avg"]) = compute_goal_averages(df)

df["h2h_home_winrate"]  = compute_h2h(df)
df["home_advantage"]    = compute_home_advantage(df)

# Chenh lech form
df["form_diff"]          = df["home_form"] - df["away_form"]
df["scored_diff"]        = df["home_scored_avg"] - df["away_scored_avg"]
df["conceded_diff"]      = df["home_conceded_avg"] - df["away_conceded_avg"]

# --- Features MOI ---

# Elo rating
df["home_elo"], df["away_elo"] = compute_elo(df)
df["elo_diff"] = df["home_elo"] - df["away_elo"]

# Shots trung binh
df["home_shots_avg"], df["away_shots_avg"] = compute_stat_averages(df, "HS", "AS", default=5.0)
df["home_sot_avg"], df["away_sot_avg"] = compute_stat_averages(df, "HST", "AST", default=2.0)
df["shots_diff"] = df["home_shots_avg"] - df["away_shots_avg"]
df["sot_diff"]   = df["home_sot_avg"] - df["away_sot_avg"]

# Corners trung binh
df["home_corners_avg"], df["away_corners_avg"] = compute_stat_averages(df, "HC", "AC", default=5.0)
df["corners_diff"] = df["home_corners_avg"] - df["away_corners_avg"]

# Rest days
df["home_rest"], df["away_rest"] = compute_rest_days(df)
df["rest_diff"] = df["home_rest"] - df["away_rest"]

# Streak (chuoi thang/thua)
df["home_streak"], df["away_streak"] = compute_streak(df)
df["streak_diff"] = df["home_streak"] - df["away_streak"]

# Season goal difference
df["home_season_gd"], df["away_season_gd"] = compute_season_gd(df)
df["season_gd_diff"] = df["home_season_gd"] - df["away_season_gd"]

# Season points per game
df["home_ppg"], df["away_ppg"] = compute_season_points(df)
df["ppg_diff"] = df["home_ppg"] - df["away_ppg"]

# Season progress
df["home_progress"], df["away_progress"] = compute_season_progress(df)


# ── Encode label ──────────────────────────────────────────────────────────────

LABEL_MAP = {"H": 0, "D": 1, "A": 2}
df["label"] = df["FTR"].map(LABEL_MAP)


# ── Chon cot cuoi cung de luu ─────────────────────────────────────────────────

FINAL_COLS = [
    # Thong tin tran dau
    "Season", "Date", "HomeTeam", "AwayTeam",
    # Ket qua thuc te
    "FTHG", "FTAG", "FTR", "label",
    # Features cu (da sua)
    "home_form", "away_form", "form_diff",
    "home_scored_avg", "home_conceded_avg",
    "away_scored_avg", "away_conceded_avg",
    "scored_diff", "conceded_diff",
    "h2h_home_winrate",
    "home_advantage",
    # Features MOI
    "home_elo", "away_elo", "elo_diff",
    "home_shots_avg", "away_shots_avg", "shots_diff",
    "home_sot_avg", "away_sot_avg", "sot_diff",
    "home_corners_avg", "away_corners_avg", "corners_diff",
    "home_rest", "away_rest", "rest_diff",
    "home_streak", "away_streak", "streak_diff",
    "home_season_gd", "away_season_gd", "season_gd_diff",
    "home_ppg", "away_ppg", "ppg_diff",
    "home_progress", "away_progress",
]

df_clean = df[FINAL_COLS].copy()
df_clean = df_clean.dropna()

os.makedirs("data", exist_ok=True)
os.makedirs("data/epl_seasons_clean", exist_ok=True)

df_clean.to_csv(OUTPUT_PATH, index=False)

for season, group in df_clean.groupby("Season"):
    label = season.replace("/", "_")
    path  = f"data/epl_seasons_clean/epl_{label}.csv"
    group.to_csv(path, index=False)


# ── In thong ke ───────────────────────────────────────────────────────────────

n_features = len(FINAL_COLS) - 8  # tru cac cot thong tin
print(f"\nKet qua sau khi lam sach:")
print(f"  Tong so tran  : {len(df_clean)}")
print(f"  So mua giai   : {df_clean['Season'].nunique()}")
print(f"  So features   : {n_features}")
print()
print("Phan bo ket qua:")
for label_val, name in [(0, "Home Win"), (1, "Draw"), (2, "Away Win")]:
    count = (df_clean["label"] == label_val).sum()
    pct   = count / len(df_clean) * 100
    print(f"  {name:10s}: {count:4d} ({pct:.1f}%)")
print()

nan_check = df_clean.isnull().sum()
if nan_check.sum() == 0:
    print("Khong co gia tri NaN trong features.")
else:
    print("Canh bao - cac cot bi NaN:")
    print(nan_check[nan_check > 0])

print(f"\nDa luu: {OUTPUT_PATH}")
print("Buoc tiep theo: chay train_model.py")