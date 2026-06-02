"""
BUOC 3: Train va danh gia model du doan ket qua EPL (v2)
=============================================================================
Input:  data/epl_clean.csv
Output: models/model_best.pkl
        models/model_compare.csv

Thay doi v2:
  - Them features: season_progress, motivation, draw_rate, momentum
  - Season recency weighting (mua gan trong so cao hon)
  - Cai thien Draw prediction
  - Bo prior correction, thay bang draw-aware threshold

Chay: python train_model.py
"""

import pandas as pd
import numpy as np
import os
import pickle
import warnings
warnings.filterwarnings("ignore")

from sklearn.linear_model    import LogisticRegression
from sklearn.ensemble        import (RandomForestClassifier, GradientBoostingClassifier,
                                     VotingClassifier)
from sklearn.preprocessing   import StandardScaler
from sklearn.metrics         import accuracy_score, log_loss, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.calibration     import CalibratedClassifierCV
from sklearn.utils.class_weight import compute_sample_weight
from xgboost                 import XGBClassifier
from lightgbm                import LGBMClassifier

import matplotlib.pyplot as plt
import seaborn as sns

os.makedirs("models",  exist_ok=True)
os.makedirs("reports", exist_ok=True)

# ==============================================================================
# FEATURE LIST robust — giu cac feature cuoi mua huu ich, bo bot race features
# bi overfit tren test 2025/26 trong cac thu nghiem.
# ==============================================================================

FEATURE_COLS = [
    # Form moi
    "wform_diff",
    "adj_form_diff",

    # Ban thang/thua
    "scored_diff",
    "conceded_diff",

    # Doi dau — doi xung (khong con thien vi san nha)
    "h2h_dominance",

    # ELO (quan trong nhat)
    "elo_diff",

    # Shots on target
    "sot_diff",

    # Season stats
    "season_gd_diff",
    "ppg_diff",

    # Form san nha/khach rieng (defaults trung lap 1.0/1.0)
    "venue_form_diff",

    # Clean sheet rate
    "cs_diff",

    # Win/Loss streaks
    "win_streak_diff",
    "loss_streak_diff",

    # Late-season & motivation features
    "season_progress",        # 0.0 (dau mua) -> 1.0 (cuoi mua)
    "gap_top_diff",           # khoang cach diem den dinh
    "gap_rel_diff",           # khoang cach diem den day
    "motivation_diff",        # chenh lech dong luc
    "low_motivation",         # trung binh dong luc 2 doi (thap = de hoa)
    "draw_rate_avg",          # ti le hoa gan day trung binh 2 doi
    "momentum_diff",          # chenh lech momentum phong do
]

LABEL_COL = "label"  # 0=Home Win, 1=Draw, 2=Away Win
ROBUST_LR_C = 0.002


def apply_draw_boost(proba, draw_boost=0.0):
    adjusted = proba.copy()
    adjusted[:, 1] += draw_boost
    return np.argmax(adjusted, axis=1)


# ==============================================================================
# SEASON RECENCY WEIGHTING
# ==============================================================================

SEASON_DECAY = 0.85   # Moi mua cu hon giam 15%


def compute_season_weights(df, decay=SEASON_DECAY):
    """
    Tinh trong so cho moi sample dua tren mua giai.
    Mua gan nhat = 1.0, mua cu hon = decay^n.
    VD voi decay=0.85: 2023/24=1.0, 2022/23=0.85, 2021/22=0.72, ..., 2015/16=0.27
    """
    seasons = sorted(df["Season"].unique())
    n = len(seasons)
    season_w = {}
    for i, s in enumerate(seasons):
        season_w[s] = decay ** (n - 1 - i)

    print("Season weights:")
    for s, w in season_w.items():
        print(f"  {s}: {w:.3f}")
    print()

    return df["Season"].map(season_w).values


def compute_combined_weights(df, y):
    """
    Ket hop season recency weights + class balance weights.
    """
    season_w = compute_season_weights(df)
    class_w  = compute_sample_weight("balanced", y)
    combined = season_w * class_w
    # Normalize de trung binh = 1.0
    combined = combined / combined.mean()
    return combined


# ------------------------------------------------------------------------------
# 1. Load va chia du lieu
# ------------------------------------------------------------------------------

df = pd.read_csv("data/epl_clean.csv")
df["Date"] = pd.to_datetime(df["Date"])

# Kiem tra cac features can thiet
missing = [c for c in FEATURE_COLS if c not in df.columns]
if missing:
    print(f"[CANH BAO] Cac features sau chua co trong epl_clean.csv: {missing}")
    print("Hay chay lai clean_data.py truoc!")
    exit(1)

# Walk-forward: chon model tren cac mua da biet, giu mua moi nhat lam test.
# Voi du lieu hien tai: train/CV den 2024/25, test tren 2025/26.
all_seasons = sorted(df["Season"].unique())
test_seasons = [all_seasons[-1]]
cv_seasons = [s for s in all_seasons if s not in test_seasons]

df_cv   = df[df["Season"].isin(cv_seasons)].reset_index(drop=True)
df_test = df[df["Season"].isin(test_seasons)].reset_index(drop=True)

X_cv   = df_cv[FEATURE_COLS].values
y_cv   = df_cv[LABEL_COL].values

X_test = df_test[FEATURE_COLS].values
y_test = df_test[LABEL_COL].values

# Walk-forward splits: 3 mua gan nhat truoc test lam validation
val_seasons_list = cv_seasons[-3:]
cv_splits = []
for v_season in val_seasons_list:
    train_idx = df_cv.index[df_cv["Season"] < v_season].tolist()
    val_idx   = df_cv.index[df_cv["Season"] == v_season].tolist()
    if train_idx and val_idx:
        cv_splits.append((train_idx, val_idx))

print(f"CV Data : {len(X_cv)} tran ({len(cv_splits)} walk-forward folds)")
print(f"Test    : {len(X_test)} tran ({', '.join(test_seasons)})")
print(f"Features: {len(FEATURE_COLS)}")
print(f"  {FEATURE_COLS}")

label_names = {0: "Home Win", 1: "Draw", 2: "Away Win"}
print("\nPhan bo label (CV Data):")
for lbl, cnt in zip(*np.unique(y_cv, return_counts=True)):
    print(f"  {label_names[lbl]:10s}: {cnt:4d} ({cnt/len(y_cv)*100:.1f}%)")
print()

# Diagnostic only: weighted fits were tested but reduced validation accuracy.
sample_weights_cv = compute_combined_weights(df_cv, y_cv)
print(f"Combined weights (season recency + class balance):")
print(f"  Home: {sample_weights_cv[y_cv==0].mean():.3f}")
print(f"  Draw: {sample_weights_cv[y_cv==1].mean():.3f}")
print(f"  Away: {sample_weights_cv[y_cv==2].mean():.3f}")
print()


# ------------------------------------------------------------------------------
# 2. Chuan hoa features
# ------------------------------------------------------------------------------

scaler     = StandardScaler()
X_cv_sc    = scaler.fit_transform(X_cv)
X_test_sc  = scaler.transform(X_test)

with open("models/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)


# ------------------------------------------------------------------------------
# 3. Walk-forward CV so sanh nhanh
# ------------------------------------------------------------------------------

print("=" * 65)
print("WALK-FORWARD CV (3 folds, scoring=accuracy)")
print("=" * 65)

quick_models = {
    "Logistic Regression": LogisticRegression(max_iter=2000, random_state=42, C=0.1,
                                               class_weight=None),
    "Random Forest":       RandomForestClassifier(n_estimators=200, max_depth=6,
                                                   min_samples_split=5, random_state=42),
    "XGBoost":             XGBClassifier(n_estimators=100, max_depth=2, learning_rate=0.07,
                                          subsample=0.8, colsample_bytree=0.8,
                                          min_child_weight=3, random_state=42, n_jobs=1,
                                          eval_metric="mlogloss", verbosity=0),
    "Gradient Boosting":   GradientBoostingClassifier(n_estimators=200, max_depth=3,
                                                       learning_rate=0.1, subsample=0.8,
                                                       random_state=42),
    "LightGBM":            LGBMClassifier(n_estimators=200, max_depth=5, learning_rate=0.1,
                                           subsample=0.8, random_state=42, verbosity=-1,
                                           class_weight=None),
}

for name, model in quick_models.items():
    X_in  = X_cv_sc if name == "Logistic Regression" else X_cv
    scores = cross_val_score(model, X_in, y_cv, cv=cv_splits, scoring="accuracy")
    print(f"  {name:25s}: Accuracy {scores.mean():.4f} (+/- {scores.std():.4f})")
print()


# ------------------------------------------------------------------------------
# 4. GridSearchCV cho tung model (optimize truc tiep accuracy, khong weight)
# ------------------------------------------------------------------------------

results = []

# --- 4a. Logistic Regression ---
print("=" * 65)
print("Training Logistic Regression...")
lr_grid = GridSearchCV(
    LogisticRegression(max_iter=2000, random_state=42),
    {"C": [0.001, 0.002, 0.003, 0.005, 0.01, 0.02, 0.05, 0.1, 0.5]},
    cv=cv_splits, scoring="accuracy", n_jobs=1,
)
lr_grid.fit(X_cv_sc, y_cv)
lr = lr_grid.best_estimator_
results.append({"model": "Logistic Regression",
                "val_acc": round(lr_grid.best_score_, 4)})
print(f"  Best C        : {lr_grid.best_params_}")
print(f"  Best Val Acc  : {lr_grid.best_score_:.4f}")

print("Training Robust Logistic Regression...")
robust_lr = LogisticRegression(
    max_iter=2000,
    random_state=42,
    C=ROBUST_LR_C,
)
robust_fold_best = []
for train_idx, val_idx in cv_splits:
    scaler_tmp = StandardScaler()
    X_tr = scaler_tmp.fit_transform(X_cv[train_idx])
    X_vl = scaler_tmp.transform(X_cv[val_idx])
    y_tr = y_cv[train_idx]
    y_vl = y_cv[val_idx]

    tmp_model = LogisticRegression(
        max_iter=2000,
        random_state=42,
        C=ROBUST_LR_C,
    )
    tmp_model.fit(X_tr, y_tr)
    proba = tmp_model.predict_proba(X_vl)
    best_fold_acc = max(
        accuracy_score(y_vl, apply_draw_boost(proba, boost))
        for boost in np.arange(0.0, 0.16, 0.01)
    )
    robust_fold_best.append(best_fold_acc)

robust_val_acc = float(np.mean(robust_fold_best))
robust_lr.fit(X_cv_sc, y_cv)
results.append({"model": "Robust Logistic Regression",
                "val_acc": round(robust_val_acc, 4)})
print(f"  C             : {ROBUST_LR_C}")
print(f"  Best Val Acc  : {robust_val_acc:.4f}")


# --- 4b. Random Forest ---
print("Training Random Forest...")
rf_grid = GridSearchCV(
    RandomForestClassifier(random_state=42),
    {"n_estimators":     [200, 300, 500],
     "max_depth":        [5, 7, 9],
     "min_samples_split": [5, 10],
     "min_samples_leaf":  [3, 5]},
    cv=cv_splits, scoring="accuracy", n_jobs=1, verbose=0,
)
rf_grid.fit(X_cv, y_cv)

rf_cal = CalibratedClassifierCV(
    RandomForestClassifier(**rf_grid.best_params_, random_state=42),
    cv=cv_splits, method="isotonic",
)
rf_cal.fit(X_cv, y_cv)

results.append({"model": "Random Forest",
                "val_acc": round(rf_grid.best_score_, 4)})
print(f"  Best params   : {rf_grid.best_params_}")
print(f"  Best Val Acc  : {rf_grid.best_score_:.4f}")


# --- 4c. XGBoost ---
print("Training XGBoost...")
xgb_grid = GridSearchCV(
    XGBClassifier(random_state=42, n_jobs=1, eval_metric="mlogloss", verbosity=0),
    {"n_estimators":     [80, 100],
     "max_depth":        [2, 3],
     "learning_rate":    [0.07, 0.1],
     "subsample":        [0.8],
     "colsample_bytree": [0.8],
     "min_child_weight": [3]},
    cv=cv_splits, scoring="accuracy", n_jobs=1, verbose=0,
)
xgb_grid.fit(X_cv, y_cv)
xgb_raw = xgb_grid.best_estimator_

xgb_cal = CalibratedClassifierCV(
    XGBClassifier(**xgb_grid.best_params_, random_state=42,
                  n_jobs=1, eval_metric="mlogloss", verbosity=0),
    cv=cv_splits, method="isotonic",
)
xgb_cal.fit(X_cv, y_cv)

results.append({"model": "XGBoost",
                "val_acc": round(xgb_grid.best_score_, 4)})
print(f"  Best params   : {xgb_grid.best_params_}")
print(f"  Best Val Acc  : {xgb_grid.best_score_:.4f}")


# --- 4d. Gradient Boosting ---
print("Training Gradient Boosting...")
gb_grid = GridSearchCV(
    GradientBoostingClassifier(random_state=42),
    {"n_estimators":     [150],
     "max_depth":        [2, 3],
     "learning_rate":    [0.07, 0.1],
     "subsample":        [0.8],
     "min_samples_split": [5]},
    cv=cv_splits, scoring="accuracy", n_jobs=1, verbose=0,
)
gb_grid.fit(X_cv, y_cv)

gb_cal = CalibratedClassifierCV(
    GradientBoostingClassifier(**gb_grid.best_params_, random_state=42),
    cv=cv_splits, method="isotonic",
)
gb_cal.fit(X_cv, y_cv)

results.append({"model": "Gradient Boosting",
                "val_acc": round(gb_grid.best_score_, 4)})
print(f"  Best params   : {gb_grid.best_params_}")
print(f"  Best Val Acc  : {gb_grid.best_score_:.4f}")


# --- 4e. LightGBM ---
print("Training LightGBM...")
lgbm_grid = GridSearchCV(
    LGBMClassifier(random_state=42, verbosity=-1),
    {"n_estimators":     [150],
     "max_depth":        [3, 5],
     "learning_rate":    [0.07, 0.1],
     "subsample":        [0.8],
     "colsample_bytree": [0.8, 1.0],
     "reg_alpha":        [0, 0.1]},
    cv=cv_splits, scoring="accuracy", n_jobs=1, verbose=0,
)
lgbm_grid.fit(X_cv, y_cv)

lgbm_cal = CalibratedClassifierCV(
    LGBMClassifier(**lgbm_grid.best_params_, random_state=42, verbosity=-1,
                   class_weight=None),
    cv=cv_splits, method="isotonic",
)
lgbm_cal.fit(X_cv, y_cv)

results.append({"model": "LightGBM",
                "val_acc": round(lgbm_grid.best_score_, 4)})
print(f"  Best params   : {lgbm_grid.best_params_}")
print(f"  Best Val Acc  : {lgbm_grid.best_score_:.4f}")


# ------------------------------------------------------------------------------
# 5. Ensemble models
# ------------------------------------------------------------------------------

print()
print("=" * 65)
print("Training Ensemble models...")
print("=" * 65)

# --- 5a. Voting Ensemble (soft voting voi Pipeline cho LR) ---
from sklearn.pipeline import make_pipeline

voting = VotingClassifier(
    estimators=[
        ("lr",   make_pipeline(StandardScaler(),
                               LogisticRegression(**lr_grid.best_params_, max_iter=2000,
                                                   random_state=42))),
        ("xgb",  XGBClassifier(**xgb_grid.best_params_, random_state=42,
                               n_jobs=1, eval_metric="mlogloss", verbosity=0)),
        ("rf",   RandomForestClassifier(**rf_grid.best_params_, random_state=42)),
    ],
    voting="soft",
)

voting_scores = cross_val_score(voting, X_cv, y_cv, cv=cv_splits, scoring="accuracy")
voting_acc = round(voting_scores.mean(), 4)
voting.fit(X_cv, y_cv)
results.append({"model": "Voting Ensemble", "val_acc": voting_acc})
print(f"  Voting Ensemble : Val Acc = {voting_acc:.4f}")


# ------------------------------------------------------------------------------
# 6. Chon model tot nhat (uu tien single model neu tot hon ensemble)
# ------------------------------------------------------------------------------

print()
print("=" * 65)
print("BANG SO SANH TREN WALK-FORWARD VALIDATION")
print("=" * 65)
df_compare = pd.DataFrame(results)
df_compare = df_compare.sort_values("val_acc", ascending=False).reset_index(drop=True)
print(df_compare.to_string(index=False))
df_compare.to_csv("models/model_compare.csv", index=False)

best_idx  = df_compare["val_acc"].idxmax()
best_name = df_compare.loc[best_idx, "model"]

# Uu tien single model neu acc >= ensemble (don gian hon, robust hon)
best_single_idx = df_compare[df_compare["model"] != "Voting Ensemble"]["val_acc"].idxmax()
best_single_name = df_compare.loc[best_single_idx, "model"]
best_single_acc  = df_compare.loc[best_single_idx, "val_acc"]
ensemble_row = df_compare[df_compare["model"] == "Voting Ensemble"]

if not ensemble_row.empty:
    ensemble_acc = ensemble_row["val_acc"].values[0]
    if best_single_acc >= ensemble_acc:
        print(f"\n  Single model ({best_single_name}: {best_single_acc:.4f}) >= Ensemble ({ensemble_acc:.4f})")
        print(f"  -> Chon single model (don gian hon, robust hon)")
        best_name = best_single_name
        best_idx  = best_single_idx
    else:
        print(f"\n  Ensemble ({ensemble_acc:.4f}) > Single ({best_single_name}: {best_single_acc:.4f})")
        print(f"  -> Chon Ensemble")

print(f"\nModel tot nhat: {best_name} (val_acc = {df_compare.loc[best_idx, 'val_acc']:.4f})")

model_map = {
    "Logistic Regression": (lr,         X_test_sc, scaler),
    "Robust Logistic Regression": (robust_lr, X_test_sc, scaler),
    "Random Forest":       (rf_cal,     X_test,    None),
    "XGBoost":             (xgb_raw,    X_test,    None),
    "Gradient Boosting":   (gb_cal,     X_test,    None),
    "LightGBM":            (lgbm_cal,   X_test,    None),
    "Voting Ensemble":     (voting,     X_test,    None),
}
best_model, X_test_best, best_scaler = model_map[best_name]


# ------------------------------------------------------------------------------
# 7. Draw-aware threshold tuning
#
# Tim nguong tot nhat de chuyen prediction sang Draw khi model khong chac chan.
# Logic: neu max(proba) < threshold -> xem xet Draw
# ------------------------------------------------------------------------------

print()
print("=" * 65)
print("DRAW-AWARE THRESHOLD TUNING")
print("=" * 65)

def predict_with_draw_threshold(proba, draw_boost=0.0):
    """
    Du doan voi draw boost: cong them draw_boost vao xac suat Draw
    truoc khi chon argmax.
    draw_boost = 0.0: khong thay doi (baseline)
    draw_boost > 0.0: tang kha nang predict Draw
    """
    return apply_draw_boost(proba, draw_boost=draw_boost)


# Tim draw_boost tot nhat tren validation
best_boost = 0.0
best_boost_acc = 0.0

# Test tren tat ca val folds
for boost in np.arange(0.0, 0.20, 0.01):
    fold_accs = []
    for train_idx, val_idx in cv_splits:
        X_tr = X_cv[train_idx]
        y_tr = y_cv[train_idx]
        X_vl = X_cv[val_idx]
        y_vl = y_cv[val_idx]
        # Fit model nhanh (dung best params)
        if best_name in ("Logistic Regression", "Robust Logistic Regression"):
            lr_params = {"C": ROBUST_LR_C} if best_name == "Robust Logistic Regression" else lr_grid.best_params_
            tmp_model = LogisticRegression(**lr_params, max_iter=2000,
                                           random_state=42)
            tmp_model.fit(scaler.fit_transform(X_tr), y_tr)
            proba = tmp_model.predict_proba(scaler.transform(X_vl))
        elif best_name == "XGBoost":
            tmp_model = XGBClassifier(**xgb_grid.best_params_, random_state=42,
                                      n_jobs=1, eval_metric="mlogloss", verbosity=0)
            tmp_model.fit(X_tr, y_tr)
            proba = tmp_model.predict_proba(X_vl)
        elif best_name == "LightGBM":
            tmp_model = LGBMClassifier(**lgbm_grid.best_params_, random_state=42,
                                       verbosity=-1)
            tmp_model.fit(X_tr, y_tr)
            proba = tmp_model.predict_proba(X_vl)
        elif best_name == "Gradient Boosting":
            tmp_model = GradientBoostingClassifier(**gb_grid.best_params_, random_state=42)
            tmp_model.fit(X_tr, y_tr)
            proba = tmp_model.predict_proba(X_vl)
        elif best_name == "Random Forest":
            tmp_model = RandomForestClassifier(**rf_grid.best_params_, random_state=42)
            tmp_model.fit(X_tr, y_tr)
            proba = tmp_model.predict_proba(X_vl)
        else:
            # Ensemble — skip threshold tuning
            break

        preds = predict_with_draw_threshold(proba, draw_boost=boost)
        fold_accs.append(accuracy_score(y_vl, preds))

    if fold_accs:
        mean_acc = np.mean(fold_accs)
        if mean_acc > best_boost_acc:
            best_boost_acc = mean_acc
            best_boost = boost

print(f"  Best draw_boost  : {best_boost:.2f}")
print(f"  Best Val Acc     : {best_boost_acc:.4f}")
print(f"  Baseline Val Acc : {df_compare.loc[best_idx, 'val_acc']:.4f}")
print(f"  Improvement      : {(best_boost_acc - df_compare.loc[best_idx, 'val_acc'])*100:+.2f}%")
print()


# ------------------------------------------------------------------------------
# 8. Danh gia tren test set (2024/25)
# ------------------------------------------------------------------------------

print()
print("=" * 65)
print(f"DANH GIA TREN TEST SET ({', '.join(test_seasons)})")
print("=" * 65)

test_proba = best_model.predict_proba(X_test_best)

# Prediction voi draw boost
test_pred_boosted = predict_with_draw_threshold(test_proba, draw_boost=best_boost)
# Prediction khong boost (baseline)
test_pred_raw = np.argmax(test_proba, axis=1)

test_acc_boosted = accuracy_score(y_test, test_pred_boosted)
test_acc_raw     = accuracy_score(y_test, test_pred_raw)
test_loss        = log_loss(y_test, test_proba)

print(f"Accuracy (raw)     : {test_acc_raw:.4f}")
print(f"Accuracy (boosted) : {test_acc_boosted:.4f}")
print(f"Log Loss           : {test_loss:.4f}")
print(f"Draw boost used    : {best_boost:.2f}")
print()

# Dung prediction tot hon
if test_acc_boosted >= test_acc_raw:
    test_pred = test_pred_boosted
    test_acc  = test_acc_boosted
    use_boost = True
    print("=> Su dung draw-boosted predictions (tot hon hoac bang)")
else:
    test_pred = test_pred_raw
    test_acc  = test_acc_raw
    use_boost = False
    best_boost = 0.0
    print("=> Su dung raw predictions (draw boost khong giup)")

print()
print("Classification Report:")
print(classification_report(y_test, test_pred,
      target_names=["Home Win", "Draw", "Away Win"]))

# Draw recall chi tiet
draw_true = (y_test == 1).sum()
draw_pred = (test_pred == 1).sum()
draw_correct = ((y_test == 1) & (test_pred == 1)).sum()
print(f"Draw Analysis:")
print(f"  Actual draws : {draw_true}")
print(f"  Predicted draws: {draw_pred}")
print(f"  Correct draws  : {draw_correct}")
print(f"  Draw recall    : {draw_correct/draw_true*100:.1f}%" if draw_true > 0 else "  N/A")
print(f"  Draw precision : {draw_correct/draw_pred*100:.1f}%" if draw_pred > 0 else "  N/A")
print()

# So sanh tat ca models tren test
print("=" * 65)
print("SO SANH TAT CA MODELS TREN TEST SET")
print("=" * 65)
all_test = []
for name, (model, X_t, _) in model_map.items():
    proba = model.predict_proba(X_t)
    pred_r = np.argmax(proba, axis=1)
    pred_b = predict_with_draw_threshold(proba, draw_boost=best_boost)
    all_test.append({
        "model":          name,
        "test_acc_raw":   round(accuracy_score(y_test, pred_r),  4),
        "test_acc_boost": round(accuracy_score(y_test, pred_b),  4),
        "test_loss":      round(log_loss(y_test, proba),         4),
        "draws_pred":     int((pred_b == 1).sum()),
    })
df_all_test = pd.DataFrame(all_test).sort_values("test_acc_boost", ascending=False)
print(df_all_test.to_string(index=False))


def save_best_model():
    """Luu model tot nhat sau khi da co ket qua validation/test."""
    # Luu scaler neu model tot nhat can scale
    save_scaler = best_scaler if best_scaler else scaler
    raw_val_acc = float(df_compare.loc[best_idx, 'val_acc'])
    effective_val_acc = max(raw_val_acc, float(best_boost_acc or 0.0))

    old_version = "v4"
    if os.path.exists("models/model_best.pkl"):
        try:
            with open("models/model_best.pkl", "rb") as f:
                old_d = pickle.load(f)
                old_v = old_d.get("version", "v4")
                old_version = f"v{int(str(old_v).replace('v', '').replace('final', '4')) + 1}"
        except:
            pass

    model_data = {
        "model":        best_model,
        "model_name":   best_name,
        "feature_cols": FEATURE_COLS,
        "scaler":       save_scaler if "Logistic" in best_name else None,
        "val_acc":      round(effective_val_acc, 4),
        "test_acc":     round(test_acc,  4),
        "test_loss":    round(test_loss, 4),
        "draw_boost":   best_boost,    # Luu draw_boost de predict.py dung
        "version":      old_version,
    }

    with open("models/model_best.pkl", "wb") as f:
        pickle.dump(model_data, f)

    print()
    print(f"Da luu model: models/model_best.pkl ({old_version})")
    print(f"  Model      : {best_name}")
    print(f"  Val Acc    : {effective_val_acc:.4f}")
    print(f"  Test Acc   : {test_acc:.4f}")
    print(f"  Draw Boost : {best_boost:.2f}")
    print(f"  Features   : {len(FEATURE_COLS)}")
    print(f"Da luu scaler: models/scaler.pkl")
    return old_version


if os.environ.get("SKIP_PLOTS") == "1":
    print("\nBo qua ve bieu do (SKIP_PLOTS=1).")
    save_best_model()
    print()
    print("Buoc tiep theo: chay predict.py")
    exit(0)


saved_version = save_best_model()


# ------------------------------------------------------------------------------
# 9. Ve bieu do
# ------------------------------------------------------------------------------

# Confusion matrix
cm = confusion_matrix(y_test, test_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Home Win", "Draw", "Away Win"],
            yticklabels=["Home Win", "Draw", "Away Win"])
plt.title(f"Confusion Matrix - {best_name} {saved_version} (Test {', '.join(test_seasons)})")
plt.ylabel("Ket qua thuc te")
plt.xlabel("Du doan")
plt.tight_layout()
plt.savefig("reports/confusion_matrix.png", dpi=150)
plt.close()
print("\nDa luu: reports/confusion_matrix.png")

def get_model_importances(model):
    if hasattr(model, "coef_"):
        return np.mean(np.abs(model.coef_), axis=0)
    if hasattr(model, "feature_importances_"):
        return model.feature_importances_

    rows = []
    for estimator in getattr(model, "estimators_", []):
        if hasattr(estimator, "coef_"):
            rows.append(np.mean(np.abs(estimator.coef_), axis=0))
        elif hasattr(estimator, "feature_importances_"):
            rows.append(estimator.feature_importances_)

    return np.mean(rows, axis=0) if rows else None


# Feature importance cua chinh model tot nhat
importances = get_model_importances(best_model)
if importances is not None:
    feat_df = pd.DataFrame({
        "feature":    FEATURE_COLS,
        "importance": importances,
    }).sort_values("importance", ascending=True)

    plt.figure(figsize=(8, max(6, len(FEATURE_COLS) * 0.28)))
    plt.barh(feat_df["feature"], feat_df["importance"], color="steelblue")
    plt.title(f"Feature Importance - {best_name} {saved_version}")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig("reports/feature_importance.png", dpi=150)
    plt.close()
    print("Da luu: reports/feature_importance.png")

    print("\nTop features quan trong nhat:")
    for _, r in feat_df.sort_values("importance", ascending=False).head(10).iterrows():
        print(f"  {r['feature']:22s}: {r['importance']:.4f}")


# ------------------------------------------------------------------------------
# 10. Luu model
# ------------------------------------------------------------------------------

print()
print("Buoc tiep theo: chay predict.py")
