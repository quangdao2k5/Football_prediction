"""
BUOC 4c: So sanh du doan + Retrain model sau moi gameweek
==========================================================
Quy trinh moi moi gameweek:
  1. python collect_data.py   <- download ket qua vong vua xong
  2. python clean_data.py     <- tinh lai features
  3. python retrain.py        <- so sanh accuracy + retrain model
  4. python predict.py        <- du doan vong tiep theo

Script nay tu dong:
  - Doc file du doan cua gameweek gan nhat (predictions/gw{N}_predictions.csv)
  - So sanh voi ket qua thuc te vua download ve
  - Luu accuracy vao predictions/accuracy_log.csv
  - Retrain model voi toan bo du lieu moi nhat
  - Luu de ghi de models/model_best.pkl

Chay: python retrain.py
"""

import pandas as pd
import numpy as np
import pickle
import os
import glob
import warnings
warnings.filterwarnings("ignore")

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, log_loss, classification_report

CLEAN_PATH    = "data/epl_clean.csv"
MODEL_PATH    = "models/model_best.pkl"
SCALER_PATH   = "models/scaler.pkl"
ACCURACY_LOG  = "predictions/accuracy_log.csv"
RETRAIN_LOG   = "predictions/retrain_log.csv"

os.makedirs("models",      exist_ok=True)
os.makedirs("predictions", exist_ok=True)

FEATURE_COLS = [
    "wform_diff",
    "adj_form_diff",
    "scored_diff",
    "conceded_diff",
    "h2h_home_winrate",
    "home_advantage",
    "elo_diff",
    "sot_diff",
    "rest_diff",
    "season_gd_diff",
    "ppg_diff",
]
LABEL_COL   = "label"
LABEL_MAP   = {0: "Home Win", 1: "Draw", 2: "Away Win"}
FTR_TO_LABEL = {"H": "Home Win", "D": "Draw", "A": "Away Win"}


# ------------------------------------------------------------------------------
# 1. Load du lieu sau khi da chay collect + clean
# ------------------------------------------------------------------------------

df = pd.read_csv(CLEAN_PATH)
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values("Date").reset_index(drop=True)

missing = [c for c in FEATURE_COLS if c not in df.columns]
if missing:
    print(f"[LOI] Thieu features: {missing}")
    print("Hay chay lai clean_data.py truoc!")
    exit(1)

print(f"Du lieu hien tai : {len(df)} tran, {df['Season'].nunique()} mua")
print(f"Mua moi nhat     : {df['Season'].max()}")
print(f"Tran moi nhat    : {df['Date'].max().date()}")
print()


# ------------------------------------------------------------------------------
# 2. Tu dong phat hien gameweek nao vua co ket qua moi
#
# Logic: Tim tat ca file du doan (gw*_predictions.csv), sau do kiem tra
# xem gameweek nao da co ket qua thuc te trong epl_clean.csv nhung chua
# duoc ghi vao accuracy_log.csv
# ------------------------------------------------------------------------------

def get_completed_gameweeks():
    """
    Lay danh sach gameweek da co file du doan.
    Tra ve list so gameweek, sap xep tang dan.
    """
    pred_files = glob.glob("predictions/gw*_predictions.csv")
    gws = []
    for f in pred_files:
        try:
            gw = int(os.path.basename(f).replace("gw", "").replace("_predictions.csv", ""))
            gws.append(gw)
        except:
            continue
    return sorted(gws)


def get_logged_gameweeks():
    """Lay danh sach gameweek da duoc ghi vao accuracy_log."""
    if not os.path.exists(ACCURACY_LOG):
        return []
    df_log = pd.read_csv(ACCURACY_LOG)
    return df_log["gameweek"].tolist()


def find_actual_result(df_clean, home, away, season="2025/26"):
    """
    Tim ket qua thuc te cua tran dau trong epl_clean.csv.
    Tra ve FTR ("H"/"D"/"A") hoac None neu chua co.
    """
    match = df_clean[
        (df_clean["HomeTeam"] == home) &
        (df_clean["AwayTeam"] == away) &
        (df_clean["Season"]   == season)
    ]
    if len(match) == 0:
        return None
    return match.iloc[0]["FTR"]


# ------------------------------------------------------------------------------
# 3. So sanh du doan vs thuc te cho tung gameweek chua duoc log
# ------------------------------------------------------------------------------

completed_gws = get_completed_gameweeks()
logged_gws    = get_logged_gameweeks()
pending_gws   = [gw for gw in completed_gws if gw not in logged_gws]

season_live = "2025/26"   # mua dang du doan

if not pending_gws:
    print("Khong co gameweek moi nao can so sanh accuracy.")
    print(f"  Da log: {logged_gws}")
else:
    print(f"Phat hien {len(pending_gws)} gameweek can so sanh: {pending_gws}")
    print()

accuracy_rows = []

for gw in pending_gws:
    pred_path = f"predictions/gw{gw}_predictions.csv"
    df_pred   = pd.read_csv(pred_path)

    print(f"Gameweek {gw} — So sanh du doan vs thuc te")
    print("=" * 72)
    print(f"  {'Home':22s} {'Away':22s} {'Du doan':12s} {'Thuc te':12s} {'OK?'}")
    print("-" * 72)

    correct = 0
    total   = 0
    details = []

    for _, row in df_pred.iterrows():
        home      = row["home"]
        away      = row["away"]
        predicted = row["prediction"]

        actual_ftr = find_actual_result(df, home, away, season_live)

        if actual_ftr is None:
            # Tran nay chua co ket qua (chua da hoac du lieu chua cap nhat)
            print(f"  {home:22s} {away:22s} {predicted:12s} {'[Chua co]':12s}")
            continue

        actual_lbl = FTR_TO_LABEL[actual_ftr]
        is_correct = (predicted == actual_lbl)
        if is_correct:
            correct += 1
        total += 1

        status = "OK" if is_correct else "WRONG"
        print(f"  {home:22s} {away:22s} {predicted:12s} {actual_lbl:12s} {status}")

        details.append({
            "gameweek":  gw,
            "home":      home,
            "away":      away,
            "predicted": predicted,
            "actual":    actual_lbl,
            "correct":   int(is_correct),
        })

    if total == 0:
        print("  Chua co ket qua thuc te nao cho gameweek nay.")
        print()
        continue

    acc = correct / total
    print("-" * 72)
    print(f"  Dung: {correct}/{total} tran  |  Accuracy GW{gw}: {acc*100:.1f}%")
    print()

    accuracy_rows.append({
        "gameweek": gw,
        "correct":  correct,
        "total":    total,
        "accuracy": round(acc, 4),
    })

# Luu accuracy log
if accuracy_rows:
    if os.path.exists(ACCURACY_LOG):
        df_acc_log = pd.read_csv(ACCURACY_LOG)
        df_acc_log = pd.concat(
            [df_acc_log, pd.DataFrame(accuracy_rows)], ignore_index=True
        )
    else:
        df_acc_log = pd.DataFrame(accuracy_rows)

    df_acc_log = df_acc_log.sort_values("gameweek").reset_index(drop=True)
    df_acc_log.to_csv(ACCURACY_LOG, index=False)

    # Tong ket tich luy
    total_c = df_acc_log["correct"].sum()
    total_t = df_acc_log["total"].sum()
    overall = total_c / total_t if total_t > 0 else 0
    print(f"Accuracy tich luy ({len(df_acc_log)} gameweeks da log):")
    print(df_acc_log[["gameweek", "correct", "total", "accuracy"]].to_string(index=False))
    print(f"  Tong: {total_c}/{total_t} tran dung ({overall*100:.1f}%)")
    print(f"  Da luu: {ACCURACY_LOG}")
    print()


# ------------------------------------------------------------------------------
# 4. Retrain model voi toan bo du lieu moi nhat
# ------------------------------------------------------------------------------

print("=" * 65)
print("RETRAIN MODEL")
print("=" * 65)

# Train: tat ca tru 2024/25 (giu lam val de kiem tra regression)
train_seasons = [s for s in df["Season"].unique() if s != "2024/25"]
val_seasons   = ["2024/25"]

df_train = df[df["Season"].isin(train_seasons)]
df_val   = df[df["Season"].isin(val_seasons)]

X_train = df_train[FEATURE_COLS].values
y_train = df_train[LABEL_COL].values
X_val   = df_val[FEATURE_COLS].values
y_val   = df_val[LABEL_COL].values

print(f"Train: {len(X_train)} tran (den mua {sorted(train_seasons)[-1]})")
print(f"Val  : {len(X_val)} tran (2024/25 - kiem tra regression)")
print()

# Load model cu de so sanh
old_val_acc = None
if os.path.exists(MODEL_PATH):
    with open(MODEL_PATH, "rb") as f:
        old_data = pickle.load(f)
    old_model   = old_data["model"]
    old_scaler  = old_data.get("scaler")
    old_version = old_data.get("version", "?")

    X_val_old = old_scaler.transform(X_val) if old_scaler else X_val
    old_val_acc = accuracy_score(y_val, old_model.predict(X_val_old))
    print(f"Model cu (version {old_version}): Val Acc = {old_val_acc:.4f}")

# Train model moi
scaler_new = StandardScaler()
X_train_sc = scaler_new.fit_transform(X_train)
X_val_sc   = scaler_new.transform(X_val)

lr_new = LogisticRegression(max_iter=2000, random_state=42, C=0.1)
lr_new.fit(X_train_sc, y_train)

new_pred     = lr_new.predict(X_val_sc)
new_proba    = lr_new.predict_proba(X_val_sc)
new_val_acc  = accuracy_score(y_val, new_pred)
new_val_loss = log_loss(y_val, new_proba)

print(f"Model moi             : Val Acc = {new_val_acc:.4f}  LogLoss = {new_val_loss:.4f}")

if old_val_acc is not None:
    diff = new_val_acc - old_val_acc
    sign = "+" if diff >= 0 else ""
    print(f"So voi model cu       : {sign}{diff*100:.2f}%")
    if diff < -0.01:
        print("  [CANH BAO] Giam qua 1% — kiem tra lai du lieu truoc khi dung.")

# Tang version
try:
    old_ver = old_data.get("version", "v4")
    new_ver = f"v{int(str(old_ver).replace('v','').replace('final','4')) + 1}"
except:
    new_ver = "v5"

# Luu model
model_data = {
    "model":        lr_new,
    "model_name":   "Logistic Regression",
    "feature_cols": FEATURE_COLS,
    "scaler":       scaler_new,
    "val_acc":      round(new_val_acc,  4),
    "val_loss":     round(new_val_loss, 4),
    "train_size":   len(X_train),
    "version":      new_ver,
    "retrained_on": str(df["Date"].max().date()),
}

with open(MODEL_PATH,  "wb") as f:
    pickle.dump(model_data, f)
with open(SCALER_PATH, "wb") as f:
    pickle.dump(scaler_new, f)

print()
print(f"Da luu: {MODEL_PATH} ({new_ver})")
print(f"  Train tren : {len(X_train)} tran")
print(f"  Data moi nhat: {df['Date'].max().date()}")

# Luu retrain log
retrain_row = {
    "version":      new_ver,
    "retrained_on": str(df["Date"].max().date()),
    "train_size":   len(X_train),
    "val_acc":      round(new_val_acc,  4),
    "val_loss":     round(new_val_loss, 4),
    "old_val_acc":  round(old_val_acc,  4) if old_val_acc else None,
}

if os.path.exists(RETRAIN_LOG):
    df_retrain_log = pd.read_csv(RETRAIN_LOG)
    df_retrain_log = pd.concat(
        [df_retrain_log, pd.DataFrame([retrain_row])], ignore_index=True
    )
else:
    df_retrain_log = pd.DataFrame([retrain_row])

df_retrain_log.to_csv(RETRAIN_LOG, index=False)
print(f"Da luu: {RETRAIN_LOG}")
print()
print("Lich su retrain:")
print(df_retrain_log.to_string(index=False))
print()
print("Hoan tat! Tiep theo: chay predict.py de du doan gameweek tiep theo.")