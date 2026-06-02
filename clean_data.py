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
# 9. ELO rating (tich luy qua tat ca mua, regression to mean giua cac mua)
# ------------------------------------------------------------------------------

ELO_REGRESS = 0.7   # Dau mua moi: elo = 0.7 * elo_cuoi_mua_truoc + 0.3 * default

def compute_elo(df: pd.DataFrame, k: float = ELO_K, default: float = ELO_DEFAULT):
    elo: dict[str, float] = {}
    current_season = None
    home_elos, away_elos = [], []

    for _, row in df.iterrows():
        season = row["Season"]
        home, away = row["HomeTeam"], row["AwayTeam"]

        # Regression to mean dau moi mua
        if season != current_season:
            if current_season is not None:  # Khong regress mua dau tien
                for team in list(elo.keys()):
                    elo[team] = ELO_REGRESS * elo[team] + (1 - ELO_REGRESS) * default
            current_season = season

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


# ------------------------------------------------------------------------------
# 16. [MOI] Season progress — vong dau hien tai / 38
# ------------------------------------------------------------------------------

def compute_season_progress(df: pd.DataFrame):
    """
    Tinh tien trinh mua giai cho moi tran dau.
    season_progress = so tran doi nha da da trong mua / 38
    Gia tri tu 0.0 (dau mua) den ~1.0 (cuoi mua).
    """
    match_count: dict[str, int] = {}  # {team: so tran da da trong mua}
    current_season = None
    home_progress, away_progress = [], []

    for _, row in df.iterrows():
        season, home, away = row["Season"], row["HomeTeam"], row["AwayTeam"]

        if season != current_season:
            match_count = {}
            current_season = season

        h_count = match_count.get(home, 0)
        a_count = match_count.get(away, 0)

        home_progress.append(h_count / 38.0)
        away_progress.append(a_count / 38.0)

        match_count[home] = h_count + 1
        match_count[away] = a_count + 1

    return home_progress, away_progress


# ------------------------------------------------------------------------------
# 17. [MOI] Standings context — khoang cach diem den dinh/day bang
#     Va motivation proxy
# ------------------------------------------------------------------------------

def compute_standings_context(df: pd.DataFrame):
    """
    Tinh:
    - pts_gap_to_leader: khoang cach diem den doi dung dau (luon <= 0)
    - pts_gap_to_relegation: khoang cach diem den vi tri xuong hang (thu 18)
      Duong = an toan, am = dang trong vung nguy hiem
    - motivation: 0 = thieu dong luc (da an toan va khong canh tranh gi),
                  1 = co dong luc (dang canh tranh dinh/day)
    Reset moi mua.
    """
    pts: dict[str, int] = {}
    mp:  dict[str, int] = {}
    current_season = None

    home_gap_top, away_gap_top = [], []
    home_gap_rel, away_gap_rel = [], []
    home_motivation, away_motivation = [], []

    for _, row in df.iterrows():
        season, home, away = row["Season"], row["HomeTeam"], row["AwayTeam"]

        if season != current_season:
            pts, mp = {}, {}
            current_season = season

        h_pts = pts.get(home, 0)
        a_pts = pts.get(away, 0)
        h_mp  = mp.get(home, 0)
        a_mp  = mp.get(away, 0)

        # Tinh standings hien tai
        all_pts = sorted(pts.values(), reverse=True) if pts else [0]
        leader_pts = all_pts[0] if all_pts else 0
        # Vi tri xuong hang: thu 18 (index 17) hoac cuoi cung neu chua du 20 doi
        rel_idx = min(17, len(all_pts) - 1)
        rel_pts = all_pts[rel_idx] if all_pts else 0

        # Gap to leader (luon <= 0)
        home_gap_top.append(h_pts - leader_pts)
        away_gap_top.append(a_pts - leader_pts)

        # Gap to relegation (duong = an toan)
        home_gap_rel.append(h_pts - rel_pts)
        away_gap_rel.append(a_pts - rel_pts)

        # Motivation: 1.0 neu dang canh tranh (top 6 hoac bot 5), else giam dan
        remaining_h = max(1, 38 - h_mp)
        remaining_a = max(1, 38 - a_mp)
        max_possible_h = h_pts + remaining_h * 3
        max_possible_a = a_pts + remaining_a * 3

        # Doi mat dong luc neu: khong the len top 4 VA da an toan khoi xuong hang
        h_can_top = max_possible_h >= leader_pts  # con kha nang vo dich
        h_safe    = h_pts - rel_pts > remaining_h * 1.5  # khoang cach lon
        a_can_top = max_possible_a >= leader_pts
        a_safe    = a_pts - rel_pts > remaining_a * 1.5

        h_motiv = 1.0 if (h_can_top or not h_safe) else 0.5
        a_motiv = 1.0 if (a_can_top or not a_safe) else 0.5

        home_motivation.append(h_motiv)
        away_motivation.append(a_motiv)

        # Cap nhat diem
        if row["FTR"] == "H":
            pts[home] = h_pts + 3
        elif row["FTR"] == "D":
            pts[home] = h_pts + 1
            pts[away] = a_pts + 1
        else:
            pts[away] = a_pts + 3

        mp[home] = h_mp + 1
        mp[away] = a_mp + 1

    return (home_gap_top, away_gap_top,
            home_gap_rel, away_gap_rel,
            home_motivation, away_motivation)


# ------------------------------------------------------------------------------
# 18. [MOI v3] Race context — title / Europe / relegation pressure
# ------------------------------------------------------------------------------

def _clip01(value: float) -> float:
    return float(max(0.0, min(1.0, value)))


def _race_pressure_from_gap(gap: float, window: float = 9.0) -> float:
    """Cao khi doi nam gan moc canh tranh diem, thap khi da qua xa."""
    return _clip01(1.0 - abs(gap) / window)


def compute_race_context(df: pd.DataFrame):
    """
    Tao feature ve muc do canh tranh truoc tran:
    - position/points hien tai
    - khoang cach den top 4/top 6
    - title/top4/europe/relegation pressure
    - importance va dead-rubber proxy

    Tat ca duoc tinh tu bang xep hang truoc tran, khong dung ket qua tran hien tai.
    """
    season_teams = {
        season: sorted(set(group["HomeTeam"]).union(group["AwayTeam"]))
        for season, group in df.groupby("Season")
    }

    stats = {}
    current_season = None

    home_points, away_points = [], []
    home_position, away_position = [], []
    home_gap_top4, away_gap_top4 = [], []
    home_gap_europe, away_gap_europe = [], []
    home_title_pressure, away_title_pressure = [], []
    home_top4_pressure, away_top4_pressure = [], []
    home_europe_pressure, away_europe_pressure = [], []
    home_relegation_pressure, away_relegation_pressure = [], []
    home_importance, away_importance = [], []
    home_dead_rubber, away_dead_rubber = [], []

    def init_stats(season):
        return {
            team: {"pts": 0, "mp": 0, "gf": 0, "ga": 0}
            for team in season_teams[season]
        }

    def table_rows():
        return sorted(
            stats,
            key=lambda t: (
                -stats[t]["pts"],
                -(stats[t]["gf"] - stats[t]["ga"]),
                -stats[t]["gf"],
                t,
            ),
        )

    def team_context(team):
        table = table_rows()
        n_teams = len(table)
        pos_map = {team_name: idx + 1 for idx, team_name in enumerate(table)}
        pos = pos_map[team]

        pts = stats[team]["pts"]
        mp = stats[team]["mp"]
        remaining = max(0, 38 - mp)
        progress = mp / 38.0
        late_weight = _clip01((progress - 0.55) / 0.35)

        leader_pts = stats[table[0]]["pts"]
        top4_pts = stats[table[min(3, n_teams - 1)]]["pts"]
        europe_pts = stats[table[min(5, n_teams - 1)]]["pts"]
        rel_pts = stats[table[min(17, n_teams - 1)]]["pts"]

        gap_leader = pts - leader_pts
        gap_top4 = pts - top4_pts
        gap_europe = pts - europe_pts
        gap_relegation = pts - rel_pts

        title_alive = pos <= 4 and (pts + remaining * 3 >= leader_pts) and gap_leader >= -9
        title_pressure = late_weight * (_race_pressure_from_gap(gap_leader, 9.0) if title_alive else 0.0)

        top4_alive = pts + remaining * 3 >= top4_pts
        top4_pressure = late_weight * (_race_pressure_from_gap(gap_top4, 9.0) if top4_alive else 0.0)

        europe_alive = pts + remaining * 3 >= europe_pts
        europe_pressure = late_weight * (_race_pressure_from_gap(gap_europe, 9.0) if europe_alive else 0.0)

        if pos >= 18:
            relegation_pressure = late_weight
        elif pos >= 15 or gap_relegation <= 9:
            relegation_pressure = late_weight * _clip01(1.0 - max(gap_relegation, 0) / 9.0)
        else:
            relegation_pressure = 0.0

        race_importance = max(
            title_pressure,
            top4_pressure,
            europe_pressure,
            relegation_pressure,
        )
        baseline_importance = 0.35 * (1.0 - late_weight) + 0.10 * late_weight
        importance = max(baseline_importance, race_importance)
        dead_rubber = late_weight if importance <= 0.25 else 0.0

        return {
            "points": pts,
            "position": pos,
            "gap_top4": gap_top4,
            "gap_europe": gap_europe,
            "title_pressure": title_pressure,
            "top4_pressure": top4_pressure,
            "europe_pressure": europe_pressure,
            "relegation_pressure": relegation_pressure,
            "importance": importance,
            "dead_rubber": dead_rubber,
        }

    for _, row in df.iterrows():
        season, home, away = row["Season"], row["HomeTeam"], row["AwayTeam"]

        if season != current_season:
            stats = init_stats(season)
            current_season = season

        h_ctx = team_context(home)
        a_ctx = team_context(away)

        home_points.append(h_ctx["points"])
        away_points.append(a_ctx["points"])
        home_position.append(h_ctx["position"])
        away_position.append(a_ctx["position"])
        home_gap_top4.append(h_ctx["gap_top4"])
        away_gap_top4.append(a_ctx["gap_top4"])
        home_gap_europe.append(h_ctx["gap_europe"])
        away_gap_europe.append(a_ctx["gap_europe"])
        home_title_pressure.append(h_ctx["title_pressure"])
        away_title_pressure.append(a_ctx["title_pressure"])
        home_top4_pressure.append(h_ctx["top4_pressure"])
        away_top4_pressure.append(a_ctx["top4_pressure"])
        home_europe_pressure.append(h_ctx["europe_pressure"])
        away_europe_pressure.append(a_ctx["europe_pressure"])
        home_relegation_pressure.append(h_ctx["relegation_pressure"])
        away_relegation_pressure.append(a_ctx["relegation_pressure"])
        home_importance.append(h_ctx["importance"])
        away_importance.append(a_ctx["importance"])
        home_dead_rubber.append(h_ctx["dead_rubber"])
        away_dead_rubber.append(a_ctx["dead_rubber"])

        hg, ag = int(row["FTHG"]), int(row["FTAG"])
        if row["FTR"] == "H":
            stats[home]["pts"] += 3
        elif row["FTR"] == "D":
            stats[home]["pts"] += 1
            stats[away]["pts"] += 1
        else:
            stats[away]["pts"] += 3

        stats[home]["mp"] += 1
        stats[away]["mp"] += 1
        stats[home]["gf"] += hg
        stats[home]["ga"] += ag
        stats[away]["gf"] += ag
        stats[away]["ga"] += hg

    return {
        "home_points": home_points,
        "away_points": away_points,
        "home_position": home_position,
        "away_position": away_position,
        "home_gap_top4": home_gap_top4,
        "away_gap_top4": away_gap_top4,
        "home_gap_europe": home_gap_europe,
        "away_gap_europe": away_gap_europe,
        "home_title_pressure": home_title_pressure,
        "away_title_pressure": away_title_pressure,
        "home_top4_pressure": home_top4_pressure,
        "away_top4_pressure": away_top4_pressure,
        "home_europe_pressure": home_europe_pressure,
        "away_europe_pressure": away_europe_pressure,
        "home_relegation_pressure": home_relegation_pressure,
        "away_relegation_pressure": away_relegation_pressure,
        "home_importance": home_importance,
        "away_importance": away_importance,
        "home_dead_rubber": home_dead_rubber,
        "away_dead_rubber": away_dead_rubber,
    }


# ------------------------------------------------------------------------------
# 19. [MOI] Draw rate gan day (ti le hoa trong n tran gan nhat)
# ------------------------------------------------------------------------------

def compute_draw_rate(df: pd.DataFrame, n: int = FORM_WINDOW):
    """
    Ti le tran hoa trong n tran gan nhat cua moi doi.
    Doi hay hoa gan day -> co xu huong tiep tuc hoa.
    Reset moi mua.
    """
    draw_hist: dict[str, list] = {}  # {team: [1 if draw else 0]}
    current_season = None
    home_dr, away_dr = [], []

    for _, row in df.iterrows():
        season, home, away, result = row["Season"], row["HomeTeam"], row["AwayTeam"], row["FTR"]

        if season != current_season:
            draw_hist = {}
            current_season = season

        h_hist = draw_hist.get(home, [])
        a_hist = draw_hist.get(away, [])

        home_dr.append(np.mean(h_hist[-n:]) if h_hist else 0.25)
        away_dr.append(np.mean(a_hist[-n:]) if a_hist else 0.25)

        is_draw = 1.0 if result == "D" else 0.0
        draw_hist.setdefault(home, []).append(is_draw)
        draw_hist.setdefault(away, []).append(is_draw)

    return home_dr, away_dr


# ------------------------------------------------------------------------------
# 19. [MOI] Form momentum — form dang tang hay giam
#     form_momentum = weighted_form_5 - simple_form_10
#     Duong = dang len phong do, am = dang xuong
# ------------------------------------------------------------------------------

def compute_form_momentum(df: pd.DataFrame, n_short: int = 3, n_long: int = 8):
    """
    Tinh momentum phong do: form ngan han (3 tran) - form dai han (8 tran).
    Duong = doi dang tien bo, am = doi dang tut.
    Reset moi mua.
    """
    history: dict[str, list] = {}
    current_season = None
    home_mom, away_mom = [], []

    for _, row in df.iterrows():
        season, home, away, result = row["Season"], row["HomeTeam"], row["AwayTeam"], row["FTR"]

        if season != current_season:
            history = {}
            current_season = season

        h_hist = history.get(home, [])
        a_hist = history.get(away, [])

        if len(h_hist) >= n_short:
            short_h = np.mean(h_hist[-n_short:])
            long_h  = np.mean(h_hist[-n_long:]) if len(h_hist) >= n_long else np.mean(h_hist)
            home_mom.append(short_h - long_h)
        else:
            home_mom.append(0.0)

        if len(a_hist) >= n_short:
            short_a = np.mean(a_hist[-n_short:])
            long_a  = np.mean(a_hist[-n_long:]) if len(a_hist) >= n_long else np.mean(a_hist)
            away_mom.append(short_a - long_a)
        else:
            away_mom.append(0.0)

        # Cap nhat lich su diem
        if result == "H":
            history.setdefault(home, []).append(3)
            history.setdefault(away, []).append(0)
        elif result == "D":
            history.setdefault(home, []).append(1)
            history.setdefault(away, []).append(1)
        else:
            history.setdefault(home, []).append(0)
            history.setdefault(away, []).append(3)

    return home_mom, away_mom


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

# ELO (voi regression to mean giua cac mua)
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

# [MOI v2] Season progress
df["home_progress"], df["away_progress"] = compute_season_progress(df)
df["season_progress"] = (df["home_progress"] + df["away_progress"]) / 2

# [MOI v2] Standings context & motivation
(df["home_gap_top"], df["away_gap_top"],
 df["home_gap_rel"], df["away_gap_rel"],
 df["home_motivation"], df["away_motivation"]) = compute_standings_context(df)
df["gap_top_diff"] = df["home_gap_top"] - df["away_gap_top"]
df["gap_rel_diff"] = df["home_gap_rel"] - df["away_gap_rel"]
df["motivation_diff"] = df["home_motivation"] - df["away_motivation"]
# Khi ca 2 doi deu thieu dong luc -> xac suat Draw tang
df["low_motivation"] = ((df["home_motivation"] + df["away_motivation"]) / 2)

# [MOI v3] Race context: title / Europe / relegation / dead rubber
race_context = compute_race_context(df)
for col, values in race_context.items():
    df[col] = values

df["points_diff"] = df["home_points"] - df["away_points"]
# Duong = doi nha dang xep hang cao hon (vi tri nho hon)
df["position_diff"] = df["away_position"] - df["home_position"]
df["gap_top4_diff"] = df["home_gap_top4"] - df["away_gap_top4"]
df["gap_europe_diff"] = df["home_gap_europe"] - df["away_gap_europe"]
df["title_pressure_diff"] = df["home_title_pressure"] - df["away_title_pressure"]
df["top4_pressure_diff"] = df["home_top4_pressure"] - df["away_top4_pressure"]
df["europe_pressure_diff"] = df["home_europe_pressure"] - df["away_europe_pressure"]
df["relegation_pressure_diff"] = df["home_relegation_pressure"] - df["away_relegation_pressure"]
df["importance_diff"] = df["home_importance"] - df["away_importance"]
df["importance_avg"] = (df["home_importance"] + df["away_importance"]) / 2
df["dead_rubber_avg"] = (df["home_dead_rubber"] + df["away_dead_rubber"]) / 2
df["late_importance_diff"] = df["season_progress"] * df["importance_diff"]
df["late_dead_rubber"] = df["season_progress"] * df["dead_rubber_avg"]

# [MOI v2] Draw rate gan day
df["home_draw_rate"], df["away_draw_rate"] = compute_draw_rate(df)
df["draw_rate_avg"] = (df["home_draw_rate"] + df["away_draw_rate"]) / 2

# [MOI v2] Form momentum
df["home_momentum"], df["away_momentum"] = compute_form_momentum(df)
df["momentum_diff"] = df["home_momentum"] - df["away_momentum"]

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
    # ELO (voi regression to mean)
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
    # [MOI v2] Season progress & late-season features
    "home_progress", "away_progress", "season_progress",
    # Standings context
    "home_gap_top", "away_gap_top", "gap_top_diff",
    "home_gap_rel", "away_gap_rel", "gap_rel_diff",
    "home_motivation", "away_motivation", "motivation_diff", "low_motivation",
    # Race context / standings pressure
    "home_points", "away_points", "points_diff",
    "home_position", "away_position", "position_diff",
    "home_gap_top4", "away_gap_top4", "gap_top4_diff",
    "home_gap_europe", "away_gap_europe", "gap_europe_diff",
    "home_title_pressure", "away_title_pressure", "title_pressure_diff",
    "home_top4_pressure", "away_top4_pressure", "top4_pressure_diff",
    "home_europe_pressure", "away_europe_pressure", "europe_pressure_diff",
    "home_relegation_pressure", "away_relegation_pressure", "relegation_pressure_diff",
    "home_importance", "away_importance", "importance_diff", "importance_avg",
    "home_dead_rubber", "away_dead_rubber", "dead_rubber_avg",
    "late_importance_diff", "late_dead_rubber",
    # Draw rate
    "home_draw_rate", "away_draw_rate", "draw_rate_avg",
    # Form momentum
    "home_momentum", "away_momentum", "momentum_diff",
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
