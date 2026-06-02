"""
BUOC 4a: Du doan ket qua tran dau EPL theo gameweek (v2)
=====================================================
Input:  models/model_best.pkl
        models/scaler.pkl
        data/epl_clean.csv        <- lich su de tinh features
Output: In ket qua du doan ra terminal
        predictions/gw{N}_predictions.csv

Thay doi v2:
  - Bo prior correction (bug logic)
  - Them draw-aware prediction voi draw_boost tu model
  - Them features: season_progress, motivation, draw_rate, momentum

Chay: python predict.py
"""

import pandas as pd
import numpy as np
import pickle
import os
from datetime import datetime

os.makedirs("predictions", exist_ok=True)

ELO_DEFAULT = 1500
ELO_K       = 20
ELO_REGRESS = 0.7    # Dong bo voi clean_data.py
FORM_N      = 5
DECAY       = 0.75


# ------------------------------------------------------------------------------
# 1. Load model va du lieu lich su
# ------------------------------------------------------------------------------

with open("models/model_best.pkl", "rb") as f:
    model_data = pickle.load(f)

model       = model_data["model"]
scaler      = model_data.get("scaler")
feat_cols   = model_data["feature_cols"]
model_name  = model_data["model_name"]
draw_boost  = model_data.get("draw_boost", 0.0)   # Draw boost tu train

version = str(model_data.get("version", "?"))
version_label = version if version.startswith("v") else f"v{version}"
print(f"Model: {model_name} ({version_label})")
print(f"Draw Boost: {draw_boost:.2f}")
print(f"Features: {feat_cols}")
print()

df_hist = pd.read_csv("data/epl_clean.csv")
df_hist["Date"] = pd.to_datetime(df_hist["Date"])
df_hist = df_hist.sort_values("Date").reset_index(drop=True)


# ------------------------------------------------------------------------------
# 2. Ham tinh features tu lich su cho 1 cap doi
#
# Tat ca cac ham nay deu nhan du lieu lich su tinh den thoi diem truoc tran
# -> khong dung thong tin tuong lai
# ------------------------------------------------------------------------------

def get_elo(df, team):
    """Lay ELO hien tai cua mot doi tu cot elo da tinh san trong df."""
    games = df[(df["HomeTeam"] == team) | (df["AwayTeam"] == team)]
    if len(games) == 0:
        return ELO_DEFAULT
    last = games.iloc[-1]
    return last["home_elo"] if last["HomeTeam"] == team else last["away_elo"]


def get_weighted_form(df, team, n=FORM_N, decay=DECAY):
    """
    Tinh weighted form cua doi trong n tran gan nhat.
    Tran gan hon co trong so cao hon (exponential decay).
    """
    games = df[(df["HomeTeam"] == team) | (df["AwayTeam"] == team)].tail(n)
    if len(games) == 0:
        return 1.0

    weights = np.array([decay ** i for i in range(len(games))])
    points  = []

    for _, row in games.iterrows():
        if row["HomeTeam"] == team:
            pts = 3 if row["FTR"] == "H" else (1 if row["FTR"] == "D" else 0)
        else:
            pts = 3 if row["FTR"] == "A" else (1 if row["FTR"] == "D" else 0)
        points.append(pts)

    # Dao nguoc de tran gan nhat co trong so cao nhat
    points  = np.array(points[::-1])
    weights = weights[:len(points)]
    return float(np.dot(points, weights) / weights.sum())


def get_adjusted_form(df, team, n=FORM_N):
    """
    Tinh form co dieu chinh theo ELO doi thu.
    Thang doi manh hon duoc tinh cao hon.
    """
    games = df[(df["HomeTeam"] == team) | (df["AwayTeam"] == team)].tail(n)
    if len(games) == 0:
        return 1.5

    adj_pts = []
    for _, row in games.iterrows():
        if row["HomeTeam"] == team:
            opp_elo = row["away_elo"]
            pts     = 3 if row["FTR"] == "H" else (1 if row["FTR"] == "D" else 0)
        else:
            opp_elo = row["home_elo"]
            pts     = 3 if row["FTR"] == "A" else (1 if row["FTR"] == "D" else 0)
        adj_pts.append(pts * (opp_elo / ELO_DEFAULT))

    return float(np.mean(adj_pts))


def get_goal_avgs(df, team, n=FORM_N):
    """Tinh trung binh ban thang va ban thua trong n tran gan nhat."""
    games = df[(df["HomeTeam"] == team) | (df["AwayTeam"] == team)].tail(n)
    if len(games) == 0:
        return 1.3, 1.1

    scored, conceded = [], []
    for _, row in games.iterrows():
        if row["HomeTeam"] == team:
            scored.append(row["FTHG"])
            conceded.append(row["FTAG"])
        else:
            scored.append(row["FTAG"])
            conceded.append(row["FTHG"])

    return float(np.mean(scored)), float(np.mean(conceded))


def get_sot_avg(df, team, n=FORM_N):
    """Tinh trung binh shots on target trong n tran gan nhat."""
    games = df[(df["HomeTeam"] == team) | (df["AwayTeam"] == team)].tail(n)
    if len(games) == 0:
        return 4.0

    sots = []
    for _, row in games.iterrows():
        sot_col = "HST" if row["HomeTeam"] == team else "AST"
        if sot_col in row and not pd.isna(row[sot_col]):
            sots.append(row[sot_col])

    return float(np.mean(sots)) if sots else 4.0


def get_h2h_dominance(df, home, away, n=6):
    """Tinh net dominance cua doi nha trong lich su doi dau (doi xung).
    Gia tri tu -1.0 (away thong tri) den +1.0 (home thong tri)."""
    h2h = df[
        ((df["HomeTeam"] == home) & (df["AwayTeam"] == away)) |
        ((df["HomeTeam"] == away) & (df["AwayTeam"] == home))
    ].tail(n)

    if len(h2h) == 0:
        return 0.0  # Trung lap — khong thien vi san nha

    home_wins = 0
    away_wins = 0
    for _, row in h2h.iterrows():
        if row["HomeTeam"] == home and row["FTR"] == "H":
            home_wins += 1
        elif row["AwayTeam"] == home and row["FTR"] == "A":
            home_wins += 1
        elif row["FTR"] != "D":
            away_wins += 1

    return round((home_wins - away_wins) / len(h2h), 4)


def get_recent_form_string(df, team, n=5):
    """Tra ve chuoi dang W D L W W cho n tran gan nhat."""
    games = df[(df["HomeTeam"] == team) | (df["AwayTeam"] == team)].tail(n)
    if len(games) == 0:
        return "N/A"
    
    res = []
    for _, row in games.iterrows():
        if row["HomeTeam"] == team:
            if row["FTR"] == "H": res.append("W")
            elif row["FTR"] == "D": res.append("D")
            else: res.append("L")
        else:
            if row["FTR"] == "A": res.append("W")
            elif row["FTR"] == "D": res.append("D")
            else: res.append("L")
    return " ".join(res)


def get_h2h_stats(df, home, away, n=6):
    """Tra ve chuoi Thang-Hoa-Thua cua doi nha khi gap doi khach."""
    h2h = df[
        ((df["HomeTeam"] == home) & (df["AwayTeam"] == away)) |
        ((df["HomeTeam"] == away) & (df["AwayTeam"] == home))
    ].tail(n)

    if len(h2h) == 0:
        return "N/A"

    home_wins = 0
    draws = 0
    away_wins = 0
    for _, row in h2h.iterrows():
        if row["FTR"] == "D":
            draws += 1
        elif row["HomeTeam"] == home and row["FTR"] == "H":
            home_wins += 1
        elif row["AwayTeam"] == home and row["FTR"] == "A":
            home_wins += 1
        else:
            away_wins += 1

    return f"Thắng {home_wins} - Hòa {draws} - Thua {away_wins}"


def get_season_gd(df, team, season):
    """Goal difference tinh den hien tai trong mua."""
    season_games = df[
        (df["Season"] == season) &
        ((df["HomeTeam"] == team) | (df["AwayTeam"] == team))
    ]
    if len(season_games) == 0:
        return 0

    gd = 0
    for _, row in season_games.iterrows():
        if row["HomeTeam"] == team:
            gd += row["FTHG"] - row["FTAG"]
        else:
            gd += row["FTAG"] - row["FTHG"]
    return int(gd)


def get_ppg(df, team, season):
    """Points per game trong mua hien tai."""
    season_games = df[
        (df["Season"] == season) &
        ((df["HomeTeam"] == team) | (df["AwayTeam"] == team))
    ]
    if len(season_games) == 0:
        return 1.5

    pts = 0
    for _, row in season_games.iterrows():
        if row["HomeTeam"] == team:
            pts += 3 if row["FTR"] == "H" else (1 if row["FTR"] == "D" else 0)
        else:
            pts += 3 if row["FTR"] == "A" else (1 if row["FTR"] == "D" else 0)

    return round(pts / len(season_games), 4)


def get_venue_form(df, team, role="home", n=FORM_N):
    """
    Tinh form CHI tren san nha (role='home') hoac CHI tren san khach (role='away').
    """
    if role == "home":
        games = df[df["HomeTeam"] == team].tail(n)
        if len(games) == 0:
            return 1.0  # Trung lap (khong gia dinh home tot hon)
        pts = []
        for _, row in games.iterrows():
            pts.append(3 if row["FTR"] == "H" else (1 if row["FTR"] == "D" else 0))
    else:
        games = df[df["AwayTeam"] == team].tail(n)
        if len(games) == 0:
            return 1.0  # Trung lap (khong gia dinh away yeu hon)
        pts = []
        for _, row in games.iterrows():
            pts.append(3 if row["FTR"] == "A" else (1 if row["FTR"] == "D" else 0))

    return float(np.mean(pts))


def get_clean_sheet_rate(df, team, n=FORM_N):
    """Ti le giu sach luoi trong n tran gan nhat."""
    games = df[(df["HomeTeam"] == team) | (df["AwayTeam"] == team)].tail(n)
    if len(games) == 0:
        return 0.3

    cs = []
    for _, row in games.iterrows():
        if row["HomeTeam"] == team:
            cs.append(1.0 if row["FTAG"] == 0 else 0.0)
        else:
            cs.append(1.0 if row["FTHG"] == 0 else 0.0)

    return float(np.mean(cs))


def get_streak(df, team, streak_type="win"):
    """
    Tinh chuoi thang (streak_type='win') hoac thua ('loss') lien tiep hien tai.
    """
    games = df[(df["HomeTeam"] == team) | (df["AwayTeam"] == team)]
    if len(games) == 0:
        return 0

    streak = 0
    for _, row in games.iloc[::-1].iterrows():
        if row["HomeTeam"] == team:
            is_win  = row["FTR"] == "H"
            is_loss = row["FTR"] == "A"
        else:
            is_win  = row["FTR"] == "A"
            is_loss = row["FTR"] == "H"

        if streak_type == "win":
            if is_win:
                streak += 1
            else:
                break
        else:  # loss
            if is_loss:
                streak += 1
            else:
                break

    return streak


def get_rest_days(df, team, match_date):
    """So ngay nghi tu tran gan nhat den ngay thi dau hien tai."""
    games = df[(df["HomeTeam"] == team) | (df["AwayTeam"] == team)]
    if len(games) == 0:
        return 7
    last_date = games.iloc[-1]["Date"]
    return int((match_date - last_date).days)


# --- [MOI v2] Features moi ---

def get_season_progress(df, team, season):
    """So tran da da / 38."""
    season_games = df[
        (df["Season"] == season) &
        ((df["HomeTeam"] == team) | (df["AwayTeam"] == team))
    ]
    return len(season_games) / 38.0


def get_standings_context(df, team, season):
    """
    Tra ve (gap_to_top, gap_to_rel, motivation) cho doi trong mua.
    """
    season_df = df[df["Season"] == season]
    if len(season_df) == 0:
        return 0, 0, 1.0

    # Tinh bang diem
    teams_pts = {}
    teams_mp  = {}
    for _, row in season_df.iterrows():
        h, a = row["HomeTeam"], row["AwayTeam"]
        teams_pts.setdefault(h, 0)
        teams_pts.setdefault(a, 0)
        teams_mp.setdefault(h, 0)
        teams_mp.setdefault(a, 0)

        if row["FTR"] == "H":
            teams_pts[h] += 3
        elif row["FTR"] == "D":
            teams_pts[h] += 1
            teams_pts[a] += 1
        else:
            teams_pts[a] += 3

        teams_mp[h] += 1
        teams_mp[a] += 1

    if team not in teams_pts:
        return 0, 0, 1.0

    my_pts = teams_pts[team]
    my_mp  = teams_mp.get(team, 0)
    all_pts = sorted(teams_pts.values(), reverse=True)

    leader_pts = all_pts[0]
    rel_idx = min(17, len(all_pts) - 1)
    rel_pts = all_pts[rel_idx]

    gap_top = my_pts - leader_pts
    gap_rel = my_pts - rel_pts

    remaining = max(1, 38 - my_mp)
    max_possible = my_pts + remaining * 3
    can_top = max_possible >= leader_pts
    safe    = my_pts - rel_pts > remaining * 1.5
    motivation = 1.0 if (can_top or not safe) else 0.5

    return gap_top, gap_rel, motivation


def get_draw_rate(df, team, n=FORM_N):
    """Ti le hoa trong n tran gan nhat."""
    games = df[(df["HomeTeam"] == team) | (df["AwayTeam"] == team)].tail(n)
    if len(games) == 0:
        return 0.25
    draws = (games["FTR"] == "D").sum()
    return draws / len(games)


def get_form_momentum(df, team, n_short=3, n_long=8):
    """Form ngan han - form dai han. Duong = dang len form."""
    games = df[(df["HomeTeam"] == team) | (df["AwayTeam"] == team)]
    if len(games) < n_short:
        return 0.0

    points = []
    for _, row in games.iterrows():
        if row["HomeTeam"] == team:
            pts = 3 if row["FTR"] == "H" else (1 if row["FTR"] == "D" else 0)
        else:
            pts = 3 if row["FTR"] == "A" else (1 if row["FTR"] == "D" else 0)
        points.append(pts)

    short = np.mean(points[-n_short:])
    long  = np.mean(points[-n_long:]) if len(points) >= n_long else np.mean(points)
    return float(short - long)


def _clip01(value):
    return float(max(0.0, min(1.0, value)))


def _race_pressure_from_gap(gap, window=9.0):
    return _clip01(1.0 - abs(gap) / window)


def get_race_context(df, team, season):
    """
    Standings/race context truoc tran: title, top 4, Europe, relegation,
    va dead-rubber proxy. Dong bo voi clean_data.py.
    """
    season_df = df[df["Season"] == season]
    teams = sorted(set(season_df["HomeTeam"]).union(season_df["AwayTeam"]))
    if team not in teams:
        # Doi chua da tran nao trong mua: lay danh sach doi tu lich su gan nhat neu co.
        all_teams = sorted(set(df["HomeTeam"]).union(df["AwayTeam"]))
        teams = all_teams if team in all_teams else [team]

    stats = {
        t: {"pts": 0, "mp": 0, "gf": 0, "ga": 0}
        for t in teams
    }

    for _, row in season_df.iterrows():
        h, a = row["HomeTeam"], row["AwayTeam"]
        stats.setdefault(h, {"pts": 0, "mp": 0, "gf": 0, "ga": 0})
        stats.setdefault(a, {"pts": 0, "mp": 0, "gf": 0, "ga": 0})

        hg, ag = int(row["FTHG"]), int(row["FTAG"])
        if row["FTR"] == "H":
            stats[h]["pts"] += 3
        elif row["FTR"] == "D":
            stats[h]["pts"] += 1
            stats[a]["pts"] += 1
        else:
            stats[a]["pts"] += 3

        stats[h]["mp"] += 1
        stats[a]["mp"] += 1
        stats[h]["gf"] += hg
        stats[h]["ga"] += ag
        stats[a]["gf"] += ag
        stats[a]["ga"] += hg

    stats.setdefault(team, {"pts": 0, "mp": 0, "gf": 0, "ga": 0})

    table = sorted(
        stats,
        key=lambda t: (
            -stats[t]["pts"],
            -(stats[t]["gf"] - stats[t]["ga"]),
            -stats[t]["gf"],
            t,
        ),
    )
    n_teams = len(table)
    position = {team_name: idx + 1 for idx, team_name in enumerate(table)}[team]

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

    title_alive = position <= 4 and (pts + remaining * 3 >= leader_pts) and gap_leader >= -9
    title_pressure = late_weight * (_race_pressure_from_gap(gap_leader, 9.0) if title_alive else 0.0)

    top4_alive = pts + remaining * 3 >= top4_pts
    top4_pressure = late_weight * (_race_pressure_from_gap(gap_top4, 9.0) if top4_alive else 0.0)

    europe_alive = pts + remaining * 3 >= europe_pts
    europe_pressure = late_weight * (_race_pressure_from_gap(gap_europe, 9.0) if europe_alive else 0.0)

    if position >= 18:
        relegation_pressure = late_weight
    elif position >= 15 or gap_relegation <= 9:
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
        "position": position,
        "gap_top4": gap_top4,
        "gap_europe": gap_europe,
        "title_pressure": title_pressure,
        "top4_pressure": top4_pressure,
        "europe_pressure": europe_pressure,
        "relegation_pressure": relegation_pressure,
        "importance": importance,
        "dead_rubber": dead_rubber,
    }


# ------------------------------------------------------------------------------
# 3. Ham tong hop: tinh tat ca features cho 1 tran dau
# ------------------------------------------------------------------------------

def build_match_features(home, away, match_date, season, df):
    """
    Tinh tat ca features cho mot tran dau.
    Tra ve dict {feature_name: value}.
    """
    # Lay du lieu truoc tran nay
    df_before = df[df["Date"] < match_date].copy()
    df_season = df_before[df_before["Season"] == season].copy()

    home_elo  = get_elo(df_before, home)
    away_elo  = get_elo(df_before, away)
    elo_diff  = home_elo - away_elo

    home_wform = get_weighted_form(df_season, home)
    away_wform = get_weighted_form(df_season, away)

    home_adj = get_adjusted_form(df_season, home)
    away_adj = get_adjusted_form(df_season, away)

    home_gf, home_ga = get_goal_avgs(df_season, home)
    away_gf, away_ga = get_goal_avgs(df_season, away)

    home_sot = get_sot_avg(df_season, home)
    away_sot = get_sot_avg(df_season, away)

    h2h_dom  = get_h2h_dominance(df_before, home, away)

    home_gd  = get_season_gd(df_before, home, season)
    away_gd  = get_season_gd(df_before, away, season)

    home_ppg = get_ppg(df_before, home, season)
    away_ppg = get_ppg(df_before, away, season)

    # Venue form (defaults trung lap 1.0/1.0)
    home_venue = get_venue_form(df_season, home, role="home")
    away_venue = get_venue_form(df_season, away, role="away")

    # Clean sheet
    home_cs = get_clean_sheet_rate(df_season, home)
    away_cs = get_clean_sheet_rate(df_season, away)

    # Streaks
    home_ws = get_streak(df_season, home, "win")
    away_ws = get_streak(df_season, away, "win")
    home_ls = get_streak(df_season, home, "loss")
    away_ls = get_streak(df_season, away, "loss")

    home_rest = get_rest_days(df_before, home, match_date)
    away_rest = get_rest_days(df_before, away, match_date)

    # [MOI v2] Season progress
    home_prog = get_season_progress(df_before, home, season)
    away_prog = get_season_progress(df_before, away, season)
    season_progress = (home_prog + away_prog) / 2

    # [MOI v2] Standings context
    h_gap_top, h_gap_rel, h_motiv = get_standings_context(df_before, home, season)
    a_gap_top, a_gap_rel, a_motiv = get_standings_context(df_before, away, season)

    # [MOI v2] Draw rate
    home_dr = get_draw_rate(df_season, home)
    away_dr = get_draw_rate(df_season, away)

    # [MOI v2] Form momentum
    home_mom = get_form_momentum(df_season, home)
    away_mom = get_form_momentum(df_season, away)

    # [MOI v3] Race context
    h_race = get_race_context(df_before, home, season)
    a_race = get_race_context(df_before, away, season)

    return {
        "wform_diff":       home_wform - away_wform,
        "adj_form_diff":    home_adj   - away_adj,
        "scored_diff":      home_gf    - away_gf,
        "conceded_diff":    home_ga    - away_ga,
        "h2h_dominance":    h2h_dom,
        "elo_diff":         elo_diff,
        "sot_diff":         home_sot   - away_sot,
        "season_gd_diff":   home_gd    - away_gd,
        "ppg_diff":         home_ppg   - away_ppg,
        "venue_form_diff":  home_venue - away_venue,
        "cs_diff":          home_cs    - away_cs,
        "win_streak_diff":  home_ws    - away_ws,
        "loss_streak_diff": home_ls    - away_ls,
        "rest_diff":        home_rest  - away_rest,
        # [MOI v2]
        "season_progress":  season_progress,
        "gap_top_diff":     h_gap_top  - a_gap_top,
        "gap_rel_diff":     h_gap_rel  - a_gap_rel,
        "motivation_diff":  h_motiv    - a_motiv,
        "low_motivation":   (h_motiv + a_motiv) / 2,
        # [MOI v3]
        "points_diff":      h_race["points"] - a_race["points"],
        "position_diff":    a_race["position"] - h_race["position"],
        "gap_top4_diff":    h_race["gap_top4"] - a_race["gap_top4"],
        "gap_europe_diff":  h_race["gap_europe"] - a_race["gap_europe"],
        "title_pressure_diff": h_race["title_pressure"] - a_race["title_pressure"],
        "top4_pressure_diff": h_race["top4_pressure"] - a_race["top4_pressure"],
        "europe_pressure_diff": h_race["europe_pressure"] - a_race["europe_pressure"],
        "relegation_pressure_diff": h_race["relegation_pressure"] - a_race["relegation_pressure"],
        "importance_diff":  h_race["importance"] - a_race["importance"],
        "importance_avg":   (h_race["importance"] + a_race["importance"]) / 2,
        "dead_rubber_avg":  (h_race["dead_rubber"] + a_race["dead_rubber"]) / 2,
        "late_importance_diff": season_progress * (h_race["importance"] - a_race["importance"]),
        "late_dead_rubber": season_progress * ((h_race["dead_rubber"] + a_race["dead_rubber"]) / 2),
        "draw_rate_avg":    (home_dr + away_dr) / 2,
        "momentum_diff":    home_mom   - away_mom,
        
        # Raw stats for frontend UI details
        "home_elo":         round(home_elo, 1),
        "away_elo":         round(away_elo, 1),
        "home_form_str":    get_recent_form_string(df_season, home),
        "away_form_str":    get_recent_form_string(df_season, away),
        "home_gf":          round(home_gf, 2),
        "away_gf":          round(away_gf, 2),
        "home_ga":          round(home_ga, 2),
        "away_ga":          round(away_ga, 2),
        "home_cs":          round(home_cs, 2),
        "away_cs":          round(away_cs, 2),
        "h2h_str":          get_h2h_stats(df_before, home, away),
    }


# ------------------------------------------------------------------------------
# 4. Ham du doan cho 1 danh sach tran dau (1 gameweek)
# ------------------------------------------------------------------------------

def predict_gameweek(fixtures, gameweek, season="2025/26"):
    """
    fixtures: list of dict, moi dict co:
        {"home": "Arsenal", "away": "Liverpool", "date": "2026-04-12"}

    Tra ve DataFrame chua ket qua du doan.
    """
    print(f"Du doan Gameweek {gameweek} - Mua {season}")
    print("=" * 65)

    results = []

    for match in fixtures:
        home       = match["home"]
        away       = match["away"]
        match_date = pd.Timestamp(match["date"])

        # Tinh features
        feats = build_match_features(home, away, match_date, season, df_hist)

        # Lay dung thu tu features ma model can
        feat_vector = np.array([[feats.get(c, 0.0) for c in feat_cols]])

        # Scale neu can (Logistic Regression can scale)
        if scaler is not None:
            feat_vector = scaler.transform(feat_vector)

        # Du doan xac suat
        proba = model.predict_proba(feat_vector)[0]

        # [v2] Draw-aware prediction: boost Draw probability truoc khi chon
        adjusted_proba = proba.copy()
        adjusted_proba[1] += draw_boost   # Cong draw_boost vao Draw prob
        pred = np.argmax(adjusted_proba)

        label_map = {0: "Home Win", 1: "Draw", 2: "Away Win"}
        pred_label = label_map[pred]

        result = {
            "gameweek":  gameweek,
            "home":      home,
            "away":      away,
            "date":      match["date"],
            "home_win%": round(proba[0] * 100, 1),
            "draw%":     round(proba[1] * 100, 1),
            "away_win%": round(proba[2] * 100, 1),
            "prediction": pred_label,
            # Additional fields for frontend details modal
            "home_elo":      feats.get("home_elo", 0),
            "away_elo":      feats.get("away_elo", 0),
            "home_form_str": feats.get("home_form_str", ""),
            "away_form_str": feats.get("away_form_str", ""),
            "home_gf":       feats.get("home_gf", 0),
            "away_gf":       feats.get("away_gf", 0),
            "home_ga":       feats.get("home_ga", 0),
            "away_ga":       feats.get("away_ga", 0),
            "home_cs":       feats.get("home_cs", 0),
            "away_cs":       feats.get("away_cs", 0),
            "h2h_str":       feats.get("h2h_str", ""),
        }
        results.append(result)

        # In ket qua
        print(f"  {home:22s} vs {away:22s}")
        print(f"    H: {proba[0]*100:5.1f}%  D: {proba[1]*100:5.1f}%  A: {proba[2]*100:5.1f}%  => {pred_label}")
        print()

    df_pred = pd.DataFrame(results)

    # Luu ra file
    out_path = f"predictions/gw{gameweek}_predictions.csv"
    df_pred.to_csv(out_path, index=False)
    print(f"Da luu: {out_path}")

    return df_pred


# ------------------------------------------------------------------------------
# 5. Doc fixtures tu file fetch_fixtures.py da luu
#    Khong can nhap tay, chay fetch_fixtures.py truoc de co file nay
# ------------------------------------------------------------------------------

def load_latest_fixtures():
    """
    Tim file fixtures moi nhat trong thu muc predictions/.
    Uu tien file duoc tao/cap nhat gan nhat (theo thoi gian sua file),
    khong dung so gameweek vi tran da bu co the co so GW nho hon.

    Tra ve (fixtures_list, gameweek) hoac (None, None) neu khong tim thay.
    """
    import glob

    fix_files = glob.glob("predictions/gw*_fixtures.csv")
    if not fix_files:
        return None, None

    def extract_gw(path):
        try:
            return int(os.path.basename(path).replace("gw", "").replace("_fixtures.csv", ""))
        except:
            return 0

    # Lay file fixtures duoc sua GAN NHAT (file vua duoc fetch_fixtures.py tao)
    latest   = max(fix_files, key=os.path.getmtime)
    gw       = extract_gw(latest)
    df_fix   = pd.read_csv(latest)
    fixtures = df_fix[["home", "away", "date"]].to_dict(orient="records")

    print(f"[INFO] Dang dung file: {latest} (GW{gw}, {len(fixtures)} tran)")

    return fixtures, gw


if __name__ == "__main__":

    fixtures, gameweek = load_latest_fixtures()

    if fixtures is None:
        print("[LOI] Khong tim thay file fixtures trong predictions/")
        print("  Chay fetch_fixtures.py truoc de lay lich thi dau tu API.")
        exit(1)

    df_predictions = predict_gameweek(fixtures, gameweek=gameweek)

    print()
    print("Tom tat du doan:")
    print(f"  Home Win: {(df_predictions['prediction'] == 'Home Win').sum()} tran")
    print(f"  Draw    : {(df_predictions['prediction'] == 'Draw').sum()} tran")
    print(f"  Away Win: {(df_predictions['prediction'] == 'Away Win').sum()} tran")
