"""
Backend API cho EPL Prediction Dashboard
=========================================
Chay: uvicorn main:app --reload --port 8000

Cai thu vien:
  pip install fastapi uvicorn
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import pandas as pd
import numpy as np
import pickle
import glob
import os
import requests

app = FastAPI(title="EPL Predictor API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:5174",
        "http://localhost:3000",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:5174",
        "http://127.0.0.1:3000",
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Duong dan tuong doi tu thu muc backend/
BASE_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH   = os.path.join(BASE_DIR, "models", "model_best.pkl")
CLEAN_PATH   = os.path.join(BASE_DIR, "data",   "epl_clean.csv")
PRED_DIR     = os.path.join(BASE_DIR, "predictions")
ACC_LOG      = os.path.join(PRED_DIR, "accuracy_log.csv")
ACC_COMPARE  = os.path.join(PRED_DIR, "accuracy_comparison.csv")
REPORT_DIR   = os.path.join(BASE_DIR, "reports")

if os.path.exists(REPORT_DIR):
    app.mount("/reports", StaticFiles(directory=REPORT_DIR), name="reports")

FOOTBALL_API_KEY = "YOUR_API_KEY_HERE"   # Dung chung key voi fetch_fixtures.py


# ------------------------------------------------------------------------------
# Helper: load model
# ------------------------------------------------------------------------------

def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)

def load_history():
    df = pd.read_csv(CLEAN_PATH)
    df["Date"] = pd.to_datetime(df["Date"])
    return df.sort_values("Date").reset_index(drop=True)


def _safe_number(value, default=None, digits=None):
    if pd.isna(value):
        return default
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    if digits is not None:
        value = round(value, digits)
    return int(value) if float(value).is_integer() else value


def _team_match_result(row, team):
    is_home = row["HomeTeam"] == team
    gf = int(row["FTHG"] if is_home else row["FTAG"])
    ga = int(row["FTAG"] if is_home else row["FTHG"])

    if gf > ga:
        result = "W"
        points = 3
    elif gf == ga:
        result = "D"
        points = 1
    else:
        result = "L"
        points = 0

    return {
        "date": row["Date"].strftime("%Y-%m-%d"),
        "opponent": row["AwayTeam"] if is_home else row["HomeTeam"],
        "venue": "H" if is_home else "A",
        "result": result,
        "score": f"{gf}-{ga}",
        "gf": gf,
        "ga": ga,
        "points": points,
    }


def _summarize_team_matches(matches, team):
    rows = [_team_match_result(row, team) for _, row in matches.iterrows()]
    wins = sum(1 for row in rows if row["result"] == "W")
    draws = sum(1 for row in rows if row["result"] == "D")
    losses = sum(1 for row in rows if row["result"] == "L")
    gf = sum(row["gf"] for row in rows)
    ga = sum(row["ga"] for row in rows)
    points = sum(row["points"] for row in rows)

    played = len(rows)
    return {
        "played": played,
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "points": points,
        "ppg": round(points / played, 2) if played else 0,
        "gf": gf,
        "ga": ga,
        "gf_avg": round(gf / played, 2) if played else 0,
        "ga_avg": round(ga / played, 2) if played else 0,
        "clean_sheets": sum(1 for row in rows if row["ga"] == 0),
        "form": " ".join(row["result"] for row in rows) if rows else "N/A",
        "matches": rows,
    }


def _h2h_matches(matches, home_team):
    rows = []
    for _, row in matches.iterrows():
        home_goals = int(row["FTHG"])
        away_goals = int(row["FTAG"])

        if row["HomeTeam"] == home_team:
            gf, ga = home_goals, away_goals
            opponent = row["AwayTeam"]
            venue = "H"
        else:
            gf, ga = away_goals, home_goals
            opponent = row["HomeTeam"]
            venue = "A"

        result = "W" if gf > ga else "D" if gf == ga else "L"
        rows.append({
            "date": row["Date"].strftime("%Y-%m-%d"),
            "opponent": opponent,
            "venue": venue,
            "result": result,
            "score": f"{gf}-{ga}",
        })
    return rows


def _fixture_feature_row(history, fixture):
    date = pd.to_datetime(fixture.get("date"), errors="coerce")
    if pd.isna(date):
        return None

    mask = (
        (history["Date"] == date) &
        (history["HomeTeam"] == fixture.get("home")) &
        (history["AwayTeam"] == fixture.get("away"))
    )
    if not mask.any():
        return None
    return history[mask].iloc[0]


def enrich_predictions(df):
    history = load_history()
    rows = []

    for row in df.to_dict(orient="records"):
        match_date = pd.to_datetime(row.get("date"), errors="coerce")
        home_team = row.get("home")
        away_team = row.get("away")

        if pd.isna(match_date) or not home_team or not away_team:
            rows.append(row)
            continue

        current_season = history["Season"].max()
        before_all = history[history["Date"] < match_date]
        before = before_all[before_all["Season"] == current_season]

        home_all = before[(before["HomeTeam"] == home_team) | (before["AwayTeam"] == home_team)].tail(5)
        away_all = before[(before["HomeTeam"] == away_team) | (before["AwayTeam"] == away_team)].tail(5)
        home_home = before[before["HomeTeam"] == home_team].tail(5)
        away_away = before[before["AwayTeam"] == away_team].tail(5)
        h2h = before_all[
            ((before_all["HomeTeam"] == home_team) & (before_all["AwayTeam"] == away_team)) |
            ((before_all["HomeTeam"] == away_team) & (before_all["AwayTeam"] == home_team))
        ].tail(6)

        feature_row = _fixture_feature_row(history, row)
        if feature_row is not None:
            row.update({
                "home_position": _safe_number(feature_row.get("home_position")),
                "away_position": _safe_number(feature_row.get("away_position")),
                "home_points": _safe_number(feature_row.get("home_points")),
                "away_points": _safe_number(feature_row.get("away_points")),
                "home_motivation": _safe_number(feature_row.get("home_motivation"), digits=2),
                "away_motivation": _safe_number(feature_row.get("away_motivation"), digits=2),
                "home_venue_form_score": _safe_number(feature_row.get("home_venue_form"), digits=2),
                "away_venue_form_score": _safe_number(feature_row.get("away_venue_form"), digits=2),
            })

        row.update({
            "home_recent": _summarize_team_matches(home_all, home_team),
            "away_recent": _summarize_team_matches(away_all, away_team),
            "home_home_record": _summarize_team_matches(home_home, home_team),
            "away_away_record": _summarize_team_matches(away_away, away_team),
            "h2h_matches": _h2h_matches(h2h, home_team),
        })
        rows.append(row)

    return rows


# ------------------------------------------------------------------------------
# GET /api/predictions/latest   <- PHAI DAT TRUOC route co {gameweek}
# Tra ve du doan cua gameweek gan nhat
# ------------------------------------------------------------------------------

@app.get("/api/predictions/latest")
def get_latest_predictions():
    files = glob.glob(os.path.join(PRED_DIR, "gw*_predictions.csv"))
    if not files:
        raise HTTPException(status_code=404, detail="Chua co du doan nao")

    def extract_gw(p):
        try:
            return int(os.path.basename(p).replace("gw","").replace("_predictions.csv",""))
        except:
            return 0

    # Lay file predictions moi nhat theo thoi gian sua (khong dung so GW)
    latest = max(files, key=os.path.getmtime)
    gw     = extract_gw(latest)
    df     = pd.read_csv(latest)

    return {
        "gameweek": gw,
        "matches":  enrich_predictions(df),
    }


# ------------------------------------------------------------------------------
# GET /api/predictions/gameweeks
# Tra ve danh sach gameweek dang co file prediction
# ------------------------------------------------------------------------------

@app.get("/api/predictions/gameweeks")
def get_prediction_gameweeks():
    files = glob.glob(os.path.join(PRED_DIR, "gw*_predictions.csv"))

    gameweeks = []
    for path in files:
        try:
            gw = int(os.path.basename(path).replace("gw", "").replace("_predictions.csv", ""))
            gameweeks.append(gw)
        except ValueError:
            continue

    return {"gameweeks": sorted(set(gameweeks))}


# ------------------------------------------------------------------------------
# GET /api/predictions/{gameweek}
# Tra ve ket qua du doan cua 1 gameweek cu the
# ------------------------------------------------------------------------------

@app.get("/api/predictions/{gameweek}")
def get_predictions(gameweek: int):
    path = os.path.join(PRED_DIR, f"gw{gameweek}_predictions.csv")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail=f"Khong tim thay du doan GW{gameweek}")

    df = pd.read_csv(path)
    return {
        "gameweek": gameweek,
        "matches":  enrich_predictions(df),
    }


# ------------------------------------------------------------------------------
# GET /api/accuracy
# Tra ve lich su accuracy theo tung gameweek
# ------------------------------------------------------------------------------

@app.get("/api/accuracy")
def get_accuracy():
    if os.path.exists(ACC_COMPARE):
        df_cmp = pd.read_csv(ACC_COMPARE).sort_values("gameweek")
        df = pd.DataFrame({
            "gameweek": df_cmp["gameweek"],
            "correct":  df_cmp["new_correct"].astype(int),
            "total":    df_cmp["total"].astype(int),
            "accuracy": df_cmp["new_accuracy"].astype(float),
        })

        total_c = int(df["correct"].sum())
        total_t = int(df["total"].sum())
        overall = round(total_c / total_t, 4) if total_t > 0 else None

        return {
            "history": df.to_dict(orient="records"),
            "overall": overall,
            "total_correct": total_c,
            "total_matches": total_t,
            "source": "accuracy_comparison",
        }

    if not os.path.exists(ACC_LOG):
        return {"history": [], "overall": None}

    df  = pd.read_csv(ACC_LOG).sort_values("gameweek")
    total_c = int(df["correct"].sum())
    total_t = int(df["total"].sum())
    overall = round(total_c / total_t, 4) if total_t > 0 else None

    return {
        "history": df.to_dict(orient="records"),
        "overall": overall,
        "total_correct": total_c,
        "total_matches": total_t,
        "source": "accuracy_log",
    }


# ------------------------------------------------------------------------------
# GET /api/standings
# Lay bang xep hang hien tai tu football-data.org
# ------------------------------------------------------------------------------

@app.get("/api/standings")
def get_standings():
    if FOOTBALL_API_KEY == "e35e3e41247a49409496ccf46d76f635":
        # Fallback: tinh bang xep hang tu epl_clean.csv
        return get_standings_from_local()

    try:
        resp = requests.get(
            "https://api.football-data.org/v4/competitions/PL/standings",
            headers={"X-Auth-Token": FOOTBALL_API_KEY},
            timeout=10,
        )
        resp.raise_for_status()
        data  = resp.json()
        table = data["standings"][0]["table"]

        standings = []
        for row in table:
            standings.append({
                "position":       row["position"],
                "team":           row["team"]["name"].replace(" FC", ""),
                "played":         row["playedGames"],
                "won":            row["won"],
                "drawn":          row["draw"],
                "lost":           row["lost"],
                "goals_for":      row["goalsFor"],
                "goals_against":  row["goalsAgainst"],
                "goal_diff":      row["goalDifference"],
                "points":         row["points"],
            })
        return {"standings": standings, "source": "api"}

    except Exception as e:
        # Neu API loi thi fallback sang local
        return get_standings_from_local()


def get_standings_from_local():
    """Tinh bang xep hang tu epl_clean.csv khi khong co API."""
    df = load_history()
    current_season = df["Season"].max()
    df_s = df[df["Season"] == current_season].copy()

    teams = pd.concat([df_s["HomeTeam"], df_s["AwayTeam"]]).unique()
    rows  = []

    for team in teams:
        home = df_s[df_s["HomeTeam"] == team]
        away = df_s[df_s["AwayTeam"] == team]

        played = len(home) + len(away)
        won    = (home["FTR"] == "H").sum() + (away["FTR"] == "A").sum()
        drawn  = (home["FTR"] == "D").sum() + (away["FTR"] == "D").sum()
        lost   = played - won - drawn
        gf     = home["FTHG"].sum() + away["FTAG"].sum()
        ga     = home["FTAG"].sum() + away["FTHG"].sum()
        pts    = won * 3 + drawn

        rows.append({
            "team": team, "played": int(played),
            "won": int(won), "drawn": int(drawn), "lost": int(lost),
            "goals_for": int(gf), "goals_against": int(ga),
            "goal_diff": int(gf - ga), "points": int(pts),
        })

    df_table = pd.DataFrame(rows).sort_values(
        ["points", "goal_diff", "goals_for"], ascending=False
    ).reset_index(drop=True)
    df_table.insert(0, "position", range(1, len(df_table) + 1))

    return {"standings": df_table.to_dict(orient="records"), "source": "local"}


# ------------------------------------------------------------------------------
# GET /api/model/info
# Tra ve thong tin model hien tai
# ------------------------------------------------------------------------------

@app.get("/api/model/info")
def get_model_info():
    if not os.path.exists(MODEL_PATH):
        raise HTTPException(status_code=404, detail="Chua co model")

    data = load_model()
    feature_cols = data.get("feature_cols") or []
    return {
        "model_name":   data.get("model_name"),
        "version":      data.get("version"),
        "val_acc":      data.get("val_acc"),
        "test_acc":     data.get("test_acc"),
        "draw_boost":   data.get("draw_boost"),
        "train_size":   data.get("train_size"),
        "retrained_on": data.get("retrained_on"),
        "feature_cols": feature_cols,
        "feature_count": len(feature_cols),
        "reports": {
            "confusion_matrix": "/reports/confusion_matrix.png",
            "feature_importance": "/reports/feature_importance.png",
        },
    }


# ------------------------------------------------------------------------------
# GET /api/health
# Kiem tra server con song khong
# ------------------------------------------------------------------------------

@app.get("/api/health")
def health():
    return {"status": "ok"}
