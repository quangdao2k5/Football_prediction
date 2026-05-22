"""
BUOC 3: Train va danh gia model du doan ket qua EPL
=============================================================================
Input:  data/epl_clean.csv
Output: models/model_best.pkl
        models/model_compare.csv

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
                                     VotingClassifier, StackingClassifier)
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
# FEATURE LIST — bo rest_diff (correlation ~0), them features moi
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
]

LABEL_COL = "label"  # 0=Home Win, 1=Draw, 2=Away Win


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

# Walk-forward: train tren cac mua cu, test tren 2024/25
all_seasons = sorted(df["Season"].unique())
test_seasons = ["2024/25"]
cv_seasons = [s for s in all_seasons if s not in test_seasons and s != "2025/26"]

df_cv   = df[df["Season"].isin(cv_seasons)].reset_index(drop=True)
df_test = df[df["Season"].isin(test_seasons)].reset_index(drop=True)

X_cv   = df_cv[FEATURE_COLS].values
y_cv   = df_cv[LABEL_COL].values

X_test = df_test[FEATURE_COLS].values
y_test = df_test[LABEL_COL].values

# Walk-forward splits
val_seasons_list = ["2021/22", "2022/23", "2023/24"]
cv_splits = []
for v_season in val_seasons_list:
    train_idx = df_cv.index[df_cv["Season"] < v_season].tolist()
    val_idx   = df_cv.index[df_cv["Season"] == v_season].tolist()
    if train_idx and val_idx:
        cv_splits.append((train_idx, val_idx))

print(f"CV Data : {len(X_cv)} tran ({len(cv_splits)} walk-forward folds)")
print(f"Test    : {len(X_test)} tran (2024/25)")
print(f"Features: {len(FEATURE_COLS)}")
print(f"  {FEATURE_COLS}")

label_names = {0: "Home Win", 1: "Draw", 2: "Away Win"}
print("\nPhan bo label (CV Data):")
for lbl, cnt in zip(*np.unique(y_cv, return_counts=True)):
    print(f"  {label_names[lbl]:10s}: {cnt:4d} ({cnt/len(y_cv)*100:.1f}%)")
print()

# Class-balanced sample weights — giup model hoc tot hon Draw va Away Win
# (EPL co ~46% Home / ~27% Draw / ~27% Away — khong can bang)
sample_weights_cv = compute_sample_weight("balanced", y_cv)
print(f"Sample weights: Home={sample_weights_cv[y_cv==0].mean():.3f}, "
      f"Draw={sample_weights_cv[y_cv==1].mean():.3f}, "
      f"Away={sample_weights_cv[y_cv==2].mean():.3f}")
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
                                          min_child_weight=3, random_state=42,
                                          eval_metric="mlogloss", verbosity=0),
    "Gradient Boosting":   GradientBoostingClassifier(n_estimators=200, max_depth=3,
                                                       learning_rate=0.1, subsample=0.8,
                                                       random_state=42),
    "LightGBM":            LGBMClassifier(n_estimators=200, max_depth=5, learning_rate=0.1,
                                           subsample=0.8, random_state=42, verbosity=-1),
}

for name, model in quick_models.items():
    X_in  = X_cv_sc if name == "Logistic Regression" else X_cv
    scores = cross_val_score(model, X_in, y_cv, cv=cv_splits, scoring="accuracy")
    print(f"  {name:25s}: Accuracy {scores.mean():.4f} (+/- {scores.std():.4f})")
print()


# ------------------------------------------------------------------------------
# 4. GridSearchCV cho tung model
# ------------------------------------------------------------------------------

results = []

# --- 4a. Logistic Regression ---
print("=" * 65)
print("Training Logistic Regression...")
lr_grid = GridSearchCV(
    LogisticRegression(max_iter=2000, random_state=42, class_weight=None),
    {"C": [0.01, 0.05, 0.1, 0.5, 1.0]},
    cv=cv_splits, scoring="accuracy", n_jobs=1,
)
lr_grid.fit(X_cv_sc, y_cv)
lr = lr_grid.best_estimator_
results.append({"model": "Logistic Regression",
                "val_acc": round(lr_grid.best_score_, 4)})
print(f"  Best C        : {lr_grid.best_params_}")
print(f"  Best Val Acc  : {lr_grid.best_score_:.4f}")


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


# --- 4c. XGBoost (tuned: shallow trees, fewer estimators, moderate LR) ---
print("Training XGBoost...")
xgb_grid = GridSearchCV(
    XGBClassifier(random_state=42, eval_metric="mlogloss", verbosity=0),
    {"n_estimators":     [80, 100, 120, 150],
     "max_depth":        [2, 3],
     "learning_rate":    [0.05, 0.07, 0.1],
     "subsample":        [0.7, 0.8],
     "colsample_bytree": [0.7, 0.8],
     "min_child_weight": [1, 3, 5]},
    cv=cv_splits, scoring="accuracy", n_jobs=-1, verbose=0,
)
xgb_grid.fit(X_cv, y_cv)
xgb_raw = xgb_grid.best_estimator_

xgb_cal = CalibratedClassifierCV(
    XGBClassifier(**xgb_grid.best_params_, random_state=42,
                  eval_metric="mlogloss", verbosity=0),
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
    {"n_estimators":     [200, 300],
     "max_depth":        [3, 5],
     "learning_rate":    [0.05, 0.1],
     "subsample":        [0.8],
     "min_samples_split": [5]},
    cv=cv_splits, scoring="accuracy", n_jobs=-1, verbose=0,
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
    {"n_estimators":     [200, 300],
     "max_depth":        [3, 5, 7],
     "learning_rate":    [0.05, 0.1],
     "subsample":        [0.8],
     "colsample_bytree": [0.7, 1.0],
     "reg_alpha":        [0, 0.1]},
    cv=cv_splits, scoring="accuracy", n_jobs=-1, verbose=0,
)
lgbm_grid.fit(X_cv, y_cv)

lgbm_cal = CalibratedClassifierCV(
    LGBMClassifier(**lgbm_grid.best_params_, random_state=42, verbosity=-1),
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
                                                   random_state=42, class_weight=None))),
        ("xgb",  XGBClassifier(**xgb_grid.best_params_, random_state=42,
                               eval_metric="mlogloss", verbosity=0)),
        ("rf",   RandomForestClassifier(**rf_grid.best_params_, random_state=42)),
    ],
    voting="soft",
)

voting_scores = cross_val_score(voting, X_cv, y_cv, cv=cv_splits, scoring="accuracy")
voting_acc = round(voting_scores.mean(), 4)
voting.fit(X_cv, y_cv)  # Khong truyen sample_weight vi Pipeline(LR) khong ho tro
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
    "Random Forest":       (rf_cal,     X_test,    None),
    "XGBoost":             (xgb_raw,    X_test,    None),
    "Gradient Boosting":   (gb_cal,     X_test,    None),
    "LightGBM":            (lgbm_cal,   X_test,    None),
    "Voting Ensemble":     (voting,     X_test,    None),
}
best_model, X_test_best, best_scaler = model_map[best_name]


# ------------------------------------------------------------------------------
# 7. Danh gia tren test set (2024/25)
# ------------------------------------------------------------------------------

print()
print("=" * 65)
print("DANH GIA TREN TEST SET (2024/25)")
print("=" * 65)

test_pred  = best_model.predict(X_test_best)
test_proba = best_model.predict_proba(X_test_best)
test_acc   = accuracy_score(y_test, test_pred)
test_loss  = log_loss(y_test, test_proba)

print(f"Accuracy : {test_acc:.4f}")
print(f"Log Loss : {test_loss:.4f}")
print()
print("Classification Report:")
print(classification_report(y_test, test_pred,
      target_names=["Home Win", "Draw", "Away Win"]))

# So sanh tat ca models tren test
print("=" * 65)
print("SO SANH TAT CA MODELS TREN TEST SET")
print("=" * 65)
all_test = []
for name, (model, X_t, _) in model_map.items():
    pred  = model.predict(X_t)
    proba = model.predict_proba(X_t)
    all_test.append({
        "model":     name,
        "test_acc":  round(accuracy_score(y_test, pred),  4),
        "test_loss": round(log_loss(y_test, proba),        4),
    })
df_all_test = pd.DataFrame(all_test).sort_values("test_acc", ascending=False)
print(df_all_test.to_string(index=False))


# ------------------------------------------------------------------------------
# 8. Ve bieu do
# ------------------------------------------------------------------------------

# Confusion matrix
cm = confusion_matrix(y_test, test_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Home Win", "Draw", "Away Win"],
            yticklabels=["Home Win", "Draw", "Away Win"])
plt.title(f"Confusion Matrix - {best_name} (Test 2024/25)")
plt.ylabel("Ket qua thuc te")
plt.xlabel("Du doan")
plt.tight_layout()
plt.savefig("reports/confusion_matrix.png", dpi=150)
plt.close()
print("\nDa luu: reports/confusion_matrix.png")

# Feature importance (tu XGBoost raw model)
if hasattr(xgb_raw, "feature_importances_"):
    importances = xgb_raw.feature_importances_
    feat_df = pd.DataFrame({
        "feature":    FEATURE_COLS,
        "importance": importances,
    }).sort_values("importance", ascending=True)

    plt.figure(figsize=(8, 6))
    plt.barh(feat_df["feature"], feat_df["importance"], color="steelblue")
    plt.title(f"Feature Importance - XGBoost")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig("reports/feature_importance.png", dpi=150)
    plt.close()
    print("Da luu: reports/feature_importance.png")

    print("\nTop features quan trong nhat:")
    for _, r in feat_df.sort_values("importance", ascending=False).head(7).iterrows():
        print(f"  {r['feature']:22s}: {r['importance']:.4f}")


# ------------------------------------------------------------------------------
# 9. Luu model
# ------------------------------------------------------------------------------

# Luu scaler neu model tot nhat can scale
save_scaler = best_scaler if best_scaler else scaler

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
    "val_acc":      round(df_compare.loc[best_idx, 'val_acc'], 4),
    "test_acc":     round(test_acc,  4),
    "test_loss":    round(test_loss, 4),
    "version":      old_version,
}

with open("models/model_best.pkl", "wb") as f:
    pickle.dump(model_data, f)

print()
print(f"Da luu model: models/model_best.pkl (v4)")
print(f"  Model     : {best_name}")
print(f"  Val Acc   : {df_compare.loc[best_idx, 'val_acc']:.4f}")
print(f"  Test Acc  : {test_acc:.4f}")
print(f"  Features  : {len(FEATURE_COLS)}")
print(f"Da luu scaler: models/scaler.pkl")
print()
print("Buoc tiep theo: chay predict.py")