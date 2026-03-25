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

from sklearn.linear_model       import LogisticRegression
from sklearn.ensemble           import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing      import StandardScaler
from sklearn.metrics            import accuracy_score, log_loss, classification_report, confusion_matrix
from sklearn.model_selection    import GridSearchCV, cross_val_score
from sklearn.calibration        import CalibratedClassifierCV          
from xgboost                    import XGBClassifier

import matplotlib.pyplot as plt
import seaborn as sns

os.makedirs("models",  exist_ok=True)
os.makedirs("reports", exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE COLUMNS
# ─────────────────────────────────────────────────────────────────────────────

BASE_FEATURE_COLS = [
    "form_diff", "home_form", "away_form",
    "scored_diff", "conceded_diff",
    "h2h_home_winrate", "home_advantage",
    "elo_diff",
    "shots_diff", "sot_diff",
    "streak_diff", "season_gd_diff", "ppg_diff",
    "home_progress",
]


# Tạo sau khi load data (cần cả 2 cột gốc)
INTERACTION_FEATURES = ["elo_x_home_adv"]   # elo_diff * home_advantage

FEATURE_COLS = BASE_FEATURE_COLS + INTERACTION_FEATURES
LABEL_COL    = "label"   # 0=Home Win, 1=Draw, 2=Away Win


# Probability đã được calibrate chuẩn nên argmax là tối ưu nhất cho Accuracy.
# Tăng threshold để chỉ ép Hòa khi cực kỳ chắc chắn, tránh làm giảm Accuracy tổng.
DRAW_PROB_THRESHOLD = 0.35   # Tăng từ 0.24
HOME_AWAY_GAP       = 0.05   # Giảm từ 0.10


# ─────────────────────────────────────────────────────────────────────────────
# HÀM TIỆN ÍCH
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(model, X, y, label=""):
    """Đánh giá model, trả về dict kết quả."""
    proba = model.predict_proba(X)
    pred = model.predict(X)
    acc  = accuracy_score(y, pred)
    loss = log_loss(y, proba)
    if label:
        print(f"  {label}: Acc={acc:.4f}  LogLoss={loss:.4f}")
    return {"acc": acc, "loss": loss, "pred": pred, "proba": proba}


# ─────────────────────────────────────────────────────────────────────────────
# 1. Load & chuẩn bị dữ liệu
# ─────────────────────────────────────────────────────────────────────────────

df = pd.read_csv("data/epl_clean.csv")
df["Date"] = pd.to_datetime(df["Date"])


df["elo_x_home_adv"] = df["elo_diff"] * df["home_advantage"]

cv_seasons   = ["2015/16","2016/17","2017/18","2018/19","2019/20",
                "2020/21","2021/22","2022/23","2023/24"]
test_seasons = ["2024/25"]

df_cv   = df[df["Season"].isin(cv_seasons)].reset_index(drop=True)
df_test = df[df["Season"].isin(test_seasons)].reset_index(drop=True)

X_cv    = df_cv[FEATURE_COLS].values
y_cv    = df_cv[LABEL_COL].values

X_test  = df_test[FEATURE_COLS].values
y_test  = df_test[LABEL_COL].values

# Walk-forward splits (3 mùa cuối làm validation)
val_seasons_list = ["2021/22", "2022/23", "2023/24"]
cv_splits = []
for v_season in val_seasons_list:
    train_idx = df_cv.index[df_cv["Season"] < v_season].tolist()
    val_idx   = df_cv.index[df_cv["Season"] == v_season].tolist()
    cv_splits.append((train_idx, val_idx))


# "balanced" tự tính w_i = n_samples / (n_classes * n_samples_i)
sw_cv = None

print(f"CV Data : {len(X_cv)} tran (dung walk-forward)")
print(f"Test    : {len(X_test)} tran")
print(f"Features: {len(FEATURE_COLS)} (bao gom {len(INTERACTION_FEATURES)} interaction feature)")

# Phân bố label trong CV
label_names = {0: "Home Win", 1: "Draw", 2: "Away Win"}
print("\nPhan bo label (CV Data):")
for lbl, cnt in zip(*np.unique(y_cv, return_counts=True)):
    w_avg = 1.0
    print(f"  {label_names[lbl]:10s}: {cnt:4d}  sample_weight trung binh = {w_avg:.3f}")
print()


# ─────────────────────────────────────────────────────────────────────────────
# 2. Chuẩn hoá
# ─────────────────────────────────────────────────────────────────────────────

scaler       = StandardScaler()
X_cv_sc      = scaler.fit_transform(X_cv)
X_test_sc    = scaler.transform(X_test)

with open("models/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Cross-Validation (dùng neg_log_loss)
# ─────────────────────────────────────────────────────────────────────────────

print("=" * 65)
print("WALK-FORWARD CROSS-VALIDATION (scoring=neg_log_loss)")
print("=" * 65)

cv_models = {
    "Logistic Regression": LogisticRegression(
        max_iter=2000, random_state=42, C=1.0),                          
    "Random Forest":       RandomForestClassifier(
        n_estimators=200, max_depth=6,
        min_samples_split=5, random_state=42),                          
    "XGBoost":             XGBClassifier(
        n_estimators=200, max_depth=3, learning_rate=0.1,
        subsample=0.8, random_state=42,
        eval_metric="mlogloss", verbosity=0),              # sample_weight truyền lúc fit
    "Gradient Boosting":   GradientBoostingClassifier(
        n_estimators=200, max_depth=3,
        learning_rate=0.1, subsample=0.8, random_state=42),
}

for name, model in cv_models.items():
    X_in = X_cv_sc if name == "Logistic Regression" else X_cv
    scores = cross_val_score(model, X_in, y_cv,
                              cv=cv_splits, scoring="accuracy")
    print(f"  {name:25s}: Accuracy {scores.mean():.4f} (+/- {scores.std():.4f})")

print()


# ─────────────────────────────────────────────────────────────────────────────
# 4. GridSearchCV — [FIX 3] dùng neg_log_loss làm scoring
# ─────────────────────────────────────────────────────────────────────────────

results = []

# ── 4a. Logistic Regression ──────────────────────────────────────────────────
print("=" * 65)
print("Training Logistic Regression (Walk-Forward GridSearch)...")
lr_grid = GridSearchCV(
    LogisticRegression(max_iter=2000, random_state=42),           
    {"C": [0.01, 0.1, 1.0, 10.0]},
    cv=cv_splits, scoring="accuracy", n_jobs=1,
)
lr_grid.fit(X_cv_sc, y_cv)
lr = lr_grid.best_estimator_

results.append({"model": "Logistic Regression",
                "val_acc": round(lr_grid.best_score_, 4)})
print(f"  Best C: {lr_grid.best_params_}")
print(f"  Best Val Accuracy: {lr_grid.best_score_:.4f}")


# ── 4b. Random Forest ────────────────────────────────────────────────────────
print("\nTraining Random Forest (Walk-Forward GridSearch)...")
rf_grid = GridSearchCV(
    RandomForestClassifier(random_state=42),      
    {"n_estimators": [100, 200, 300],
     "max_depth":    [4, 6, 8],
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

results.append({"model": "Random Forest (calibrated)",
                "val_acc": round(rf_grid.best_score_, 4)})
print(f"  Best params: {rf_grid.best_params_}")
print(f"  Best Val Accuracy (uncalibrated): {rf_grid.best_score_:.4f}")


# ── 4c. XGBoost ──────────────────────────────────────────────────────────────
print("\nTraining XGBoost (Walk-Forward GridSearch)...")
xgb_grid = GridSearchCV(
    XGBClassifier(random_state=42, eval_metric="mlogloss", verbosity=0),
    {"n_estimators":     [100, 200, 300],
     "max_depth":        [3, 5],
     "learning_rate":    [0.05, 0.1],
     "subsample":        [0.8, 1.0],
     "colsample_bytree": [0.7, 1.0],
     "reg_alpha":        [0, 0.1],
     "reg_lambda":       [1, 2]},
    cv=cv_splits, scoring="accuracy", n_jobs=1, verbose=0,
)

xgb_grid.fit(X_cv, y_cv)
xgb_raw = xgb_grid.best_estimator_


xgb_cal = CalibratedClassifierCV(
    XGBClassifier(**xgb_grid.best_params_, random_state=42,
                  eval_metric="mlogloss", verbosity=0),
    cv=cv_splits, method="isotonic",
)
xgb_cal.fit(X_cv, y_cv)

results.append({"model": "XGBoost (calibrated)",
                "val_acc": round(xgb_grid.best_score_, 4)})
print(f"  Best params: {xgb_grid.best_params_}")
print(f"  Best Val Accuracy (uncalibrated): {xgb_grid.best_score_:.4f}")


# ── 4d. Gradient Boosting ────────────────────────────────────────────────────
print("\nTraining Gradient Boosting (Walk-Forward GridSearch)...")
gb_grid = GridSearchCV(
    GradientBoostingClassifier(random_state=42),
    {"n_estimators":     [100, 200, 300],
     "max_depth":        [3, 5],
     "learning_rate":    [0.05, 0.1],
     "subsample":        [0.8, 1.0],
     "min_samples_split": [5, 10]},
    cv=cv_splits, scoring="accuracy", n_jobs=1, verbose=0,
)
gb_grid.fit(X_cv, y_cv)
gb = gb_grid.best_estimator_


gb_cal = CalibratedClassifierCV(
    GradientBoostingClassifier(**gb_grid.best_params_, random_state=42),
    cv=cv_splits, method="isotonic",
)
gb_cal.fit(X_cv, y_cv)

results.append({"model": "Gradient Boosting (calibrated)",
                "val_acc": round(gb_grid.best_score_, 4)})
print(f"  Best params: {gb_grid.best_params_}")
print(f"  Best Val Accuracy (uncalibrated): {gb_grid.best_score_:.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# 5. So sánh và chọn model tốt nhất theo val_acc  [FIX 3]
# ─────────────────────────────────────────────────────────────────────────────

print()
print("=" * 65)
print("BANG SO SANH TREN WALK-FORWARD VALIDATION")
print("=" * 65)
df_compare = pd.DataFrame(results)[["model", "val_acc"]]
print(df_compare.to_string(index=False))
df_compare.to_csv("models/model_compare.csv", index=False)


best_idx  = df_compare["val_acc"].idxmax()
best_name = df_compare.loc[best_idx, "model"]
print(f"\nModel tot nhat (theo val_acc): {best_name}")

model_map = {
    "Logistic Regression":           (lr,      X_test_sc),
    "Random Forest (calibrated)":    (rf_cal,  X_test),
    "XGBoost (calibrated)":          (xgb_cal, X_test),
    "Gradient Boosting (calibrated)":(gb_cal,  X_test),
}
best_model, X_test_best = model_map[best_name]


# ─────────────────────────────────────────────────────────────────────────────
# 6. Đánh giá trên Test Set (2024/25) — so sánh có/không threshold tuning
# ─────────────────────────────────────────────────────────────────────────────

print()
print("=" * 65)
print("DANH GIA MODEL TOT NHAT TREN TEST SET (2024/25)")
print("=" * 65)

res = evaluate(best_model, X_test_best, y_test)

print("Classification Report:")
print(classification_report(y_test, res["pred"],
      target_names=["Home Win", "Draw", "Away Win"]))


# ── So sánh tất cả models trên test set ──────────────────────────────────────
print("=" * 65)
print("SO SANH TAT CA MODELS TREN TEST SET")
print("=" * 65)
all_test = []
for name, (model, X_t) in model_map.items():
    r = evaluate(model, X_t, y_test)
    all_test.append({"model": name,
                     "test_acc":  round(r["acc"],  4),
                     "test_loss": round(r["loss"], 4)})
print(pd.DataFrame(all_test).to_string(index=False))


# ─────────────────────────────────────────────────────────────────────────────
# 7. Vẽ biểu đồ
# ─────────────────────────────────────────────────────────────────────────────

# Confusion matrix
cm = confusion_matrix(y_test, res["pred"])
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

# Feature importance
raw_model = xgb_raw if "XGBoost" in best_name else (
            rf_grid.best_estimator_ if "Random Forest" in best_name else (
            gb_grid.best_estimator_ if "Gradient" in best_name else None))
if raw_model is not None and hasattr(raw_model, "feature_importances_"):
    importances = raw_model.feature_importances_
    feat_df = pd.DataFrame({
        "feature":    FEATURE_COLS,
        "importance": importances,
    }).sort_values("importance", ascending=True)
    plt.figure(figsize=(8, 8))
    plt.barh(feat_df["feature"], feat_df["importance"], color="steelblue")
    plt.title(f"Feature Importance - {best_name}")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig("reports/feature_importance.png", dpi=150)
    plt.close()
    print("Da luu: reports/feature_importance.png")

    print("\nTop 5 features quan trong nhat:")
    for _, row in feat_df.sort_values("importance", ascending=False).head(5).iterrows():
        print(f"  {row['feature']:25s}: {row['importance']:.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# 8. Lưu model
# ─────────────────────────────────────────────────────────────────────────────

model_data = {
    "model":                best_model,
    "model_name":           best_name,
    "feature_cols":         FEATURE_COLS,
    "scaler":               scaler if "Logistic" in best_name else None,
    "test_acc":             round(res["acc"],  4),
    "test_loss":            round(res["loss"], 4),
    "version": "final",
}

with open("models/model_best.pkl", "wb") as f:
    pickle.dump(model_data, f)

print()
print(f"Da luu model: models/model_best.pkl  (version final)")
print(f"Da luu scaler: models/scaler.pkl")
print()
print("Buoc tiep theo: chay predict.py")
