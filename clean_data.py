"""
BUOC 2: Lam sach du lieu va tao features
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

FORM_WINDOW = 5
ELO_K       = 20
ELO_DEFAULT = 1500


# ------------------------------------------------------------------------------
# 1. Load du lieu
# ------------------------------------------------------------------------------

df = pd.read_csv(INPUT_PATH)
df["Date"] = pd.to_datetime(df["Date"], format="%Y-%m-%d", errors="coerce")
df = df.dropna(subset=["Date", "HomeTeam", "AwayTeam", "FTR"])
df = df.sort_values("Date").reset_index(drop=True)

print(f"Loaded: {len(df)} tran, {df['Season'].nunique()} mua")


# ------------------------------------------------------------------------------
# 2. Chuan hoa ten doi
# ------------------------------------------------------------------------------

NAME_MAP = {
    "Man United"       : "Manchester United",
    "Man City"         : "Manchester City",
    "Newcastle"        : "Newcastle United",
    "Wolves"           : "Wolverhampton",
    "Sheffield United" : "Sheffield Utd",
    "Tottenham"        : "Tottenham",
    "Leeds"            : "Leeds United",
    "Leicester"        : "Leicester City",
    "Norwich"          : "Norwich City",
    "Stoke"            : "Stoke City",
    "Swansea"          : "Swansea City",
    "Hull"             : "Hull City",
    "West Brom"        : "West Bromwich",
    "QPR"              : "Queens Park Rangers",
}

df["HomeTeam"] = df["HomeTeam"].replace(NAME_MAP)
df["AwayTeam"]  = df["AwayTeam"].replace(NAME_MAP)


# ------------------------------------------------------------------------------
# 3. Form don gian (trung binh diem N tran gan nhat, reset moi mua)
# ------------------------------------------------------------------------------

def compute_form(df: pd.DataFrame, n: int = FORM_WINDOW):
    history: dict[str, list] = {}
    current_season = None
    home_forms, away_forms = [], []

    for _, row in df.iterrows():
        season, home, away, result = row["Season"], row["HomeTeam"], row["AwayTeam"], row["FTR"]

        if season != current_season:
            history = {}
            current_season = season

        home_pts = history.get(home, [])
        away_pts = history.get(away, [])

        home_forms.append(np.mean(home_pts[-n:]) if home_pts else 1.0)
        away_forms.append(np.mean(away_pts[-n:]) if away_pts else 1.0)

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


# ------------------------------------------------------------------------------
# 4. [MOI] Weighted form: tran gan hon co trong so cao hon (exponential decay)
#
# Vi du voi n=5, decay=0.75:
#   tran gan nhat  * 0.75^0 = 1.00
#   tran thu 2     * 0.75^1 = 0.75
#   tran thu 3     * 0.75^2 = 0.56
#   tran thu 4     * 0.75^3 = 0.42
#   tran thu 5     * 0.75^4 = 0.32
# => Phong do hien tai anh huong nhieu hon phong do cu
# ------------------------------------------------------------------------------

def compute_weighted_form(df: pd.DataFrame, n: int = FORM_WINDOW, decay: float = 0.75):
    """
    Tinh weighted form: moi tran gan nhat duoc nhan trong so cao hon.
    Tra ve form co trong so (da chuan hoa) cho doi nha va doi khach.
    Reset moi mua.
    """
    history: dict[str, list] = {}  # {team: [diem1, diem2, ...]} (cu nhat truoc)
    current_season = None
    home_wforms, away_wforms = [], []

    # Tao mang trong so: tran gan nhat co trong so cao nhat
    weights = np.array([decay ** i for i in range(n)])  # [1, 0.75, 0.56, ...]

    for _, row in df.iterrows():
        season, home, away, result = row["Season"], row["HomeTeam"], row["AwayTeam"], row["FTR"]

        if season != current_season:
            history = {}
            current_season = season

        home_pts = history.get(home, [])
        away_pts = history.get(away, [])

        # Lay N tran gan nhat (dao nguoc de tran moi nhat o dau)
        if home_pts:
            recent_h = np.array(home_pts[-n:][::-1])   # [gan nhat, ..., cu nhat]
            w_h      = weights[:len(recent_h)]
            home_wforms.append(np.dot(recent_h, w_h) / w_h.sum())
        else:
            home_wforms.append(1.0)

        if away_pts:
            recent_a = np.array(away_pts[-n:][::-1])
            w_a      = weights[:len(recent_a)]
            away_wforms.append(np.dot(recent_a, w_a) / w_a.sum())
        else:
            away_wforms.append(1.0)

        # Cap nhat lich su
        if result == "H":
            history.setdefault(home, []).append(3)
            history.setdefault(away, []).append(0)
        elif result == "D":
            history.setdefault(home, []).append(1)
            history.setdefault(away, []).append(1)
        else:
            history.setdefault(home, []).append(0)
            history.setdefault(away, []).append(3)

    return home_wforms, away_wforms


# ------------------------------------------------------------------------------
# 5. [MOI] Form co tinh do kho cua doi thu (opponent-adjusted form)
#
# Y tuong: thang Man City (ELO cao) quy gia hon thang Sheffield Utd (ELO thap)
# Diem thuc te se duoc nhan voi (ELO doi thu / 1500) de phan anh do kho
# ------------------------------------------------------------------------------

def compute_adjusted_form(df: pd.DataFrame, n: int = FORM_WINDOW):
    """
    Tinh form co dieu chinh theo do manh cua doi thu:
      adjusted_pts = pts * (opponent_elo / ELO_DEFAULT)
    Cao hon = thang nhung doi manh hon, thua truoc doi yeu hon.
    Reset moi mua.
    """
    history: dict[str, list] = {}   # {team: [adjusted_pts]}
    elo: dict[str, float]    = {}   # ELO tich luy (khong reset)
    current_season = None
    home_adj, away_adj = [], []

    for _, row in df.iterrows():
        season, home, away, result = row["Season"], row["HomeTeam"], row["AwayTeam"], row["FTR"]

        if season != current_season:
            history = {}
            current_season = season

        h_elo = elo.get(home, ELO_DEFAULT)
        a_elo = elo.get(away, ELO_DEFAULT)

        home_hist = history.get(home, [])
        away_hist = history.get(away, [])

        home_adj.append(np.mean(home_hist[-n:]) if home_hist else 1.5)
        away_adj.append(np.mean(away_hist[-n:]) if away_hist else 1.5)

        # Tinh diem co dieu chinh theo ELO doi thu
        opp_factor_h = a_elo / ELO_DEFAULT   # doi nha gap doi khach
        opp_factor_a = h_elo / ELO_DEFAULT   # doi khach gap doi nha

        if result == "H":
            history.setdefault(home, []).append(3 * opp_factor_h)
            history.setdefault(away, []).append(0)
        elif result == "D":
            history.setdefault(home, []).append(1 * opp_factor_h)
            history.setdefault(away, []).append(1 * opp_factor_a)
        else:
            history.setdefault(home, []).append(0)
            history.setdefault(away, []).append(3 * opp_factor_a)

        # Cap nhat ELO
        exp_home = 1.0 / (1.0 + 10 ** ((a_elo - h_elo) / 400))
        s_home   = 1.0 if result == "H" else (0.5 if result == "D" else 0.0)
        elo[home] = h_elo + ELO_K * (s_home - exp_home)
        elo[away] = a_elo + ELO_K * ((1 - s_home) - (1 - exp_home))

    return home_adj, away_adj


# ------------------------------------------------------------------------------
# 6. Trung binh ban thang/thua (reset moi mua)
# ------------------------------------------------------------------------------

def compute_goal_averages(df: pd.DataFrame, n: int = FORM_WINDOW):
    scored_hist: dict[str, list]   = {}
    conceded_hist: dict[str, list] = {}
    current_season = None
    home_scored_avg, home_conceded_avg = [], []
    away_scored_avg, away_conceded_avg = [], []

    for _, row in df.iterrows():
        season = row["Season"]
        home, away = row["HomeTeam"], row["AwayTeam"]
        hg, ag = row["FTHG"], row["FTAG"]

        if season != current_season:
            scored_hist, conceded_hist = {}, {}
            current_season = season

        hs  = scored_hist.get(home, [])
        hc  = conceded_hist.get(home, [])
        as_ = scored_hist.get(away, [])
        ac  = conceded_hist.get(away, [])

        home_scored_avg.append(np.mean(hs[-n:])   if hs  else 1.3)
        home_conceded_avg.append(np.mean(hc[-n:]) if hc  else 1.1)
        away_scored_avg.append(np.mean(as_[-n:])  if as_ else 1.1)
        away_conceded_avg.append(np.mean(ac[-n:]) if ac  else 1.3)

        scored_hist.setdefault(home, []).append(hg)
        conceded_hist.setdefault(home, []).append(ag)
        scored_hist.setdefault(away, []).append(ag)
        conceded_hist.setdefault(away, []).append(hg)

    return home_scored_avg, home_conceded_avg, away_scored_avg, away_conceded_avg


# ------------------------------------------------------------------------------
# 7. Lich su doi dau truc tiep (H2H) — doi xung (khong thien vi san nha)
#
# h2h_dominance = (so tran home_team thang - so tran away_team thang) / tong tran
# Gia tri tu -1.0 (away team thong tri) den +1.0 (home team thong tri)
# Default: 0.0 (trung lap — khong doi nao co loi the)
# ------------------------------------------------------------------------------

def compute_h2h(df: pd.DataFrame) -> list:
    h2h_hist: dict[tuple, list] = {}  # {sorted_key: [(home_team_of_that_row, result_for_home)]}
    h2h_dominance = []

    for _, row in df.iterrows():
        home, away = row["HomeTeam"], row["AwayTeam"]
        key = tuple(sorted([home, away]))

        past = h2h_hist.get(key, [])
        if past:
            # Tinh net dominance cua home team hien tai
            home_wins = sum(1 for (h, r) in past if (h == home and r == 1.0) or (h != home and r == 0.0))
            away_wins = sum(1 for (h, r) in past if (h == home and r == 0.0) or (h != home and r == 1.0))
            h2h_dominance.append((home_wins - away_wins) / len(past))
        else:
            h2h_dominance.append(0.0)  # Trung lap — khong thien vi san nha

        val = 1.0 if row["FTR"] == "H" else (0.5 if row["FTR"] == "D" else 0.0)
        h2h_hist.setdefault(key, []).append((home, val))

    return h2h_dominance


# ------------------------------------------------------------------------------
# 8. [DA XOA] Ti le thang san nha tich luy
#    Da loai bo home_advantage vi gay thien vi san nha.
#    ELO va venue_form da du de capture do manh cua doi.
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
# 9. ELO rating (tich luy qua tat ca mua)
# ------------------------------------------------------------------------------

def compute_elo(df: pd.DataFrame, k: float = ELO_K, default: float = ELO_DEFAULT):
    elo: dict[str, float] = {}
    home_elos, away_elos = [], []

    for _, row in df.iterrows():
        home, away = row["HomeTeam"], row["AwayTeam"]
        h_elo = elo.get(home, default)
        a_elo = elo.get(away, default)

        home_elos.append(h_elo)
        away_elos.append(a_elo)

        exp_home = 1.0 / (1.0 + 10 ** ((a_elo - h_elo) / 400))
        s_home   = 1.0 if row["FTR"] == "H" else (0.5 if row["FTR"] == "D" else 0.0)

        elo[home] = h_elo + k * (s_home - exp_home)
        elo[away] = a_elo + k * ((1 - s_home) - (1 - exp_home))

    return home_elos, away_elos


# ------------------------------------------------------------------------------
# 10. Shots on target trung binh (giu lai, bo shots tong so)
#     Shots on target phan anh chat luong cu sut tot hon shots tong so
# ------------------------------------------------------------------------------

def compute_stat_averages(df: pd.DataFrame, home_col: str, away_col: str,
                          n: int = FORM_WINDOW, default: float = 5.0):
    hist: dict[str, list] = {}
    current_season = None
    home_avgs, away_avgs = [], []

    for _, row in df.iterrows():
        season = row["Season"]
        home, away = row["HomeTeam"], row["AwayTeam"]

        if season != current_season:
            hist = {}
            current_season = season

        h_hist = hist.get(home, [])
        a_hist = hist.get(away, [])

        home_avgs.append(np.mean(h_hist[-n:]) if h_hist else default)
        away_avgs.append(np.mean(a_hist[-n:]) if a_hist else default)

        hist.setdefault(home, []).append(row[home_col])
        hist.setdefault(away, []).append(row[away_col])

    return home_avgs, away_avgs


# ------------------------------------------------------------------------------
# 11. Rest days (so ngay nghi giua 2 tran lien tiep)
# ------------------------------------------------------------------------------

def compute_rest_days(df: pd.DataFrame):
    last_match: dict[str, pd.Timestamp] = {}
    home_rest, away_rest = [], []

    for _, row in df.iterrows():
        home, away, date = row["HomeTeam"], row["AwayTeam"], row["Date"]
        h_last = last_match.get(home)
        a_last = last_match.get(away)

        home_rest.append((date - h_last).days if h_last is not None else 7)
        away_rest.append((date - a_last).days if a_last is not None else 7)

        last_match[home] = date
        last_match[away] = date

    return home_rest, away_rest


# ------------------------------------------------------------------------------
# 12. Season goal difference va points per game
# ------------------------------------------------------------------------------

def compute_season_gd(df: pd.DataFrame):
    gd: dict[str, int] = {}
    current_season = None
    home_gds, away_gds = [], []

    for _, row in df.iterrows():
        season = row["Season"]
        home, away = row["HomeTeam"], row["AwayTeam"]
        hg, ag = int(row["FTHG"]), int(row["FTAG"])

        if season != current_season:
            gd = {}
            current_season = season

        home_gds.append(gd.get(home, 0))
        away_gds.append(gd.get(away, 0))

        gd[home] = gd.get(home, 0) + (hg - ag)
        gd[away] = gd.get(away, 0) + (ag - hg)

    return home_gds, away_gds


def compute_season_points(df: pd.DataFrame):
    pts: dict[str, int] = {}
    mp:  dict[str, int] = {}
    current_season = None
    home_ppg, away_ppg = [], []

    for _, row in df.iterrows():
        season = row["Season"]
        home, away = row["HomeTeam"], row["AwayTeam"]

        if season != current_season:
            pts, mp = {}, {}
            current_season = season

        h_pts, a_pts = pts.get(home, 0), pts.get(away, 0)
        h_mp,  a_mp  = mp.get(home, 0),  mp.get(away, 0)

        home_ppg.append(h_pts / h_mp if h_mp > 0 else 1.5)
        away_ppg.append(a_pts / a_mp if a_mp > 0 else 1.5)

        if row["FTR"] == "H":
            pts[home] = h_pts + 3
        elif row["FTR"] == "D":
            pts[home] = h_pts + 1
            pts[away] = a_pts + 1
        else:
            pts[away] = a_pts + 3

        mp[home] = h_mp + 1
        mp[away] = a_mp + 1

    return home_ppg, away_ppg


# ------------------------------------------------------------------------------
# 13. [MOI] Form chi tinh o san nha / san khach
#     VD: doi nha manh o san nha nhung yeu o san khach
# ------------------------------------------------------------------------------

def compute_venue_form(df: pd.DataFrame, n: int = FORM_WINDOW):
    """
    Tinh form CHI tren san nha (cho doi nha) va CHI tren san khach (cho doi khach).
    Khac voi form thuong (tinh toan bo tran), day chi tinh khi doi da dung vai tro do.
    Reset moi mua.
    """
    home_at_home: dict[str, list] = {}  # lich su diem khi da san nha
    away_at_away: dict[str, list] = {}  # lich su diem khi da san khach
    current_season = None
    home_venue_forms, away_venue_forms = [], []

    for _, row in df.iterrows():
        season, home, away, result = row["Season"], row["HomeTeam"], row["AwayTeam"], row["FTR"]

        if season != current_season:
            home_at_home, away_at_away = {}, {}
            current_season = season

        hh = home_at_home.get(home, [])
        aa = away_at_away.get(away, [])

        home_venue_forms.append(np.mean(hh[-n:]) if hh else 1.0)  # Trung lap (khong gia dinh home tot hon)
        away_venue_forms.append(np.mean(aa[-n:]) if aa else 1.0)  # Trung lap (khong gia dinh away yeu hon)

        # Cap nhat
        if result == "H":
            home_at_home.setdefault(home, []).append(3)
            away_at_away.setdefault(away, []).append(0)
        elif result == "D":
            home_at_home.setdefault(home, []).append(1)
            away_at_away.setdefault(away, []).append(1)
        else:
            home_at_home.setdefault(home, []).append(0)
            away_at_away.setdefault(away, []).append(3)

    return home_venue_forms, away_venue_forms


# ------------------------------------------------------------------------------
# 14. [MOI] Clean sheet rate (ti le giu sach luoi)
# ------------------------------------------------------------------------------

def compute_clean_sheet(df: pd.DataFrame, n: int = FORM_WINDOW):
    """
    Ti le giu sach luoi (khong bi ban thang) trong n tran gan nhat.
    Clean sheet cao = hang phong ngu tot.
    Reset moi mua.
    """
    cs_hist: dict[str, list] = {}
    current_season = None
    home_cs, away_cs = [], []

    for _, row in df.iterrows():
        season, home, away = row["Season"], row["HomeTeam"], row["AwayTeam"]
        hg, ag = row["FTHG"], row["FTAG"]

        if season != current_season:
            cs_hist = {}
            current_season = season

        h_hist = cs_hist.get(home, [])
        a_hist = cs_hist.get(away, [])

        home_cs.append(np.mean(h_hist[-n:]) if h_hist else 0.3)
        away_cs.append(np.mean(a_hist[-n:]) if a_hist else 0.3)

        # 1 = clean sheet, 0 = bi ban thang
        cs_hist.setdefault(home, []).append(1.0 if ag == 0 else 0.0)
        cs_hist.setdefault(away, []).append(1.0 if hg == 0 else 0.0)

    return home_cs, away_cs


# ------------------------------------------------------------------------------
# 15. [MOI] Win streak / Loss streak (chuoi thang/thua lien tiep)
# ------------------------------------------------------------------------------

def compute_streaks(df: pd.DataFrame):
    """
    Tinh chuoi thang lien tiep va chuoi thua lien tiep hien tai.
    VD: Chelsea thua 5 tran lien tiep -> loss_streak = 5
    Reset moi mua.
    """
    win_streak:  dict[str, int] = {}
    loss_streak: dict[str, int] = {}
    current_season = None
    home_ws, away_ws = [], []
    home_ls, away_ls = [], []

    for _, row in df.iterrows():
        season, home, away, result = row["Season"], row["HomeTeam"], row["AwayTeam"], row["FTR"]

        if season != current_season:
            win_streak, loss_streak = {}, {}
            current_season = season

        home_ws.append(win_streak.get(home, 0))
        away_ws.append(win_streak.get(away, 0))
        home_ls.append(loss_streak.get(home, 0))
        away_ls.append(loss_streak.get(away, 0))

        # Cap nhat streaks
        if result == "H":
            win_streak[home]  = win_streak.get(home, 0) + 1
            loss_streak[home] = 0
            win_streak[away]  = 0
            loss_streak[away] = loss_streak.get(away, 0) + 1
        elif result == "A":
            win_streak[away]  = win_streak.get(away, 0) + 1
            loss_streak[away] = 0
            win_streak[home]  = 0
            loss_streak[home] = loss_streak.get(home, 0) + 1
        else:  # Draw
            win_streak[home]  = 0
            win_streak[away]  = 0
            loss_streak[home] = 0
            loss_streak[away] = 0

    return home_ws, away_ws, home_ls, away_ls


# ==============================================================================
# GAN FEATURES VAO DATAFRAME
# ==============================================================================

print("Dang tinh features...")

# Form don gian (giu lai de so sanh)
df["home_form"], df["away_form"] = compute_form(df)
df["form_diff"] = df["home_form"] - df["away_form"]

# [MOI] Weighted form (tran gan hon quan trong hon)
df["home_wform"], df["away_wform"] = compute_weighted_form(df, decay=0.75)
df["wform_diff"] = df["home_wform"] - df["away_wform"]

# [MOI] Adjusted form (co tinh do kho cua doi thu)
df["home_adj_form"], df["away_adj_form"] = compute_adjusted_form(df)
df["adj_form_diff"] = df["home_adj_form"] - df["away_adj_form"]

# Trung binh ban thang/thua
(df["home_scored_avg"], df["home_conceded_avg"],
 df["away_scored_avg"], df["away_conceded_avg"]) = compute_goal_averages(df)
df["scored_diff"]   = df["home_scored_avg"] - df["away_scored_avg"]
df["conceded_diff"] = df["home_conceded_avg"] - df["away_conceded_avg"]

# H2H — doi xung (khong con thien vi san nha)
df["h2h_dominance"] = compute_h2h(df)

# ELO
df["home_elo"], df["away_elo"] = compute_elo(df)
df["elo_diff"] = df["home_elo"] - df["away_elo"]

# [GIU] Chi giu shots on target, bo shots tong so (chat luong hon so luong)
df["home_sot_avg"], df["away_sot_avg"] = compute_stat_averages(
    df, "HST", "AST", default=2.0)
df["sot_diff"] = df["home_sot_avg"] - df["away_sot_avg"]

# Rest days
df["home_rest"], df["away_rest"] = compute_rest_days(df)
df["rest_diff"] = df["home_rest"] - df["away_rest"]

# Season stats
df["home_season_gd"], df["away_season_gd"] = compute_season_gd(df)
df["season_gd_diff"] = df["home_season_gd"] - df["away_season_gd"]

df["home_ppg"], df["away_ppg"] = compute_season_points(df)
df["ppg_diff"] = df["home_ppg"] - df["away_ppg"]

# [MOI] Form san nha/khach rieng
df["home_venue_form"], df["away_venue_form"] = compute_venue_form(df)
df["venue_form_diff"] = df["home_venue_form"] - df["away_venue_form"]

# [MOI] Clean sheet rate
df["home_cs_rate"], df["away_cs_rate"] = compute_clean_sheet(df)
df["cs_diff"] = df["home_cs_rate"] - df["away_cs_rate"]

# [MOI] Win/Loss streaks
(df["home_win_streak"], df["away_win_streak"],
 df["home_loss_streak"], df["away_loss_streak"]) = compute_streaks(df)
df["win_streak_diff"]  = df["home_win_streak"]  - df["away_win_streak"]
df["loss_streak_diff"] = df["home_loss_streak"] - df["away_loss_streak"]

# Label encode
LABEL_MAP = {"H": 0, "D": 1, "A": 2}
df["label"] = df["FTR"].map(LABEL_MAP)


# ------------------------------------------------------------------------------
# Chon cot cuoi cung
# ------------------------------------------------------------------------------

FINAL_COLS = [
    # Thong tin tran dau
    "Season", "Date", "HomeTeam", "AwayTeam",
    # Ket qua thuc te
    "FTHG", "FTAG", "FTR", "label",
    # Form don gian (giu de tham khao)
    "home_form", "away_form", "form_diff",
    # Weighted form
    "home_wform", "away_wform", "wform_diff",
    # Adjusted form (co tinh do kho doi thu)
    "home_adj_form", "away_adj_form", "adj_form_diff",
    # Ban thang/thua
    "home_scored_avg", "home_conceded_avg",
    "away_scored_avg", "away_conceded_avg",
    "scored_diff", "conceded_diff",
    # H2H — doi xung (da sua de khong thien vi san nha)
    "h2h_dominance",
    # ELO
    "home_elo", "away_elo", "elo_diff",
    # Shots on target
    "home_sot_avg", "away_sot_avg", "sot_diff",
    # Rest days
    "home_rest", "away_rest", "rest_diff",
    # Season stats
    "home_season_gd", "away_season_gd", "season_gd_diff",
    "home_ppg", "away_ppg", "ppg_diff",
    # Form san nha/khach rieng (defaults trung lap 1.0/1.0)
    "home_venue_form", "away_venue_form", "venue_form_diff",
    # Clean sheet rate
    "home_cs_rate", "away_cs_rate", "cs_diff",
    # Win/Loss streaks
    "home_win_streak", "away_win_streak", "win_streak_diff",
    "home_loss_streak", "away_loss_streak", "loss_streak_diff",
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


# ------------------------------------------------------------------------------
# Thong ke
# ------------------------------------------------------------------------------

n_features = len(FINAL_COLS) - 8
print(f"\nKet qua sau khi lam sach:")
print(f"  Tong so tran  : {len(df_clean)}")
print(f"  So mua giai   : {df_clean['Season'].nunique()}")
print(f"  So features   : {n_features}")
print()
print("Phan bo ket qua:")
for lv, name in [(0, "Home Win"), (1, "Draw"), (2, "Away Win")]:
    count = (df_clean["label"] == lv).sum()
    pct   = count / len(df_clean) * 100
    print(f"  {name:10s}: {count:4d} ({pct:.1f}%)")

nan_check = df_clean.isnull().sum()
if nan_check.sum() == 0:
    print("\nKhong co gia tri NaN.")
else:
    print("\nCac cot bi NaN:")
    print(nan_check[nan_check > 0])

print(f"\nDa luu: {OUTPUT_PATH}")
print("Tiep theo: chay python train_model.py")