"""
Backend API cho EPL Prediction Dashboard
=========================================
Chay: uvicorn main:app --reload --port 8000

Cai thu vien:
  pip install fastapi uvicorn
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
import pickle
import glob
import os
import requests

app = FastAPI(title="EPL Predictor API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Duong dan tuong doi tu thu muc backend/
BASE_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH   = os.path.join(BASE_DIR, "models", "model_best.pkl")
CLEAN_PATH   = os.path.join(BASE_DIR, "data",   "epl_clean.csv")
PRED_DIR     = os.path.join(BASE_DIR, "predictions")
ACC_LOG      = os.path.join(PRED_DIR, "accuracy_log.csv")

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

    latest = max(files, key=extract_gw)
    gw     = extract_gw(latest)
    df     = pd.read_csv(latest)

    return {
        "gameweek": gw,
        "matches":  df.to_dict(orient="records"),
    }


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
        "matches":  df.to_dict(orient="records"),
    }


# ------------------------------------------------------------------------------
# GET /api/accuracy
# Tra ve lich su accuracy theo tung gameweek
# ------------------------------------------------------------------------------

@app.get("/api/accuracy")
def get_accuracy():
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
    return {
        "model_name":   data.get("model_name"),
        "version":      data.get("version"),
        "val_acc":      data.get("val_acc"),
        "train_size":   data.get("train_size"),
        "retrained_on": data.get("retrained_on"),
        "feature_cols": data.get("feature_cols"),
    }


# ------------------------------------------------------------------------------
# GET /api/health
# Kiem tra server con song khong
# ------------------------------------------------------------------------------

@app.get("/api/health")
def health():
    return {"status": "ok"}