"""
BUOC 3: Train va danh gia model du doan ket qua EPL (PHIEN BAN CAI THIEN v3)
=============================================================================
Thay doi so voi v2:
  [FIX 1] Class weighting: phạt nặng khi đoán sai Draw (sample_weight / class_weight)
  [FIX 2] Threshold tuning: custom predict ưu tiên Draw khi xác suất gần nhau
  [FIX 3] Chọn model tốt nhất theo val_loss (Log Loss) thay vì val_acc
  [FIX 4] Probability calibration cho XGBoost / tree-based models
  [FIX 5] Thêm feature tương tác: elo_diff * home_advantage

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
from sklearn.calibration        import CalibratedClassifierCV          # [FIX 4]
from sklearn.utils.class_weight import compute_sample_weight           # [FIX 1]
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
    "corners_diff", "rest_diff",
    "streak_diff", "season_gd_diff", "ppg_diff",
    "home_progress",
]

# [FIX 5] Thêm feature tương tác phi tuyến
# Tạo sau khi load data (cần cả 2 cột gốc)
INTERACTION_FEATURES = ["elo_x_home_adv"]   # elo_diff * home_advantage

FEATURE_COLS = BASE_FEATURE_COLS + INTERACTION_FEATURES
LABEL_COL    = "label"   # 0=Home Win, 1=Draw, 2=Away Win

# [FIX 2] Ngưỡng cho custom predict
DRAW_PROB_THRESHOLD = 0.28   # P(Draw) tối thiểu để xét
HOME_AWAY_GAP       = 0.10   # Khoảng cách max giữa P(Home) và P(Away)


# ─────────────────────────────────────────────────────────────────────────────
# HÀM TIỆN ÍCH
# ─────────────────────────────────────────────────────────────────────────────

def predict_with_threshold(proba: np.ndarray,
                            draw_thresh: float = DRAW_PROB_THRESHOLD,
                            gap_thresh:  float = HOME_AWAY_GAP) -> np.ndarray:
    """
    [FIX 2] Custom predict ưu tiên Draw:
    Nếu P(Home) và P(Away) gần nhau (gap < gap_thresh)
    VÀ P(Draw) đủ cao (> draw_thresh) → dự đoán Draw.
    Ngược lại chọn class có xác suất cao nhất (argmax bình thường).

    Columns: 0=Home Win, 1=Draw, 2=Away Win
    """
    preds = np.argmax(proba, axis=1).copy()
    p_home  = proba[:, 0]
    p_draw  = proba[:, 1]
    p_away  = proba[:, 2]
    gap     = np.abs(p_home - p_away)
    mask    = (gap < gap_thresh) & (p_draw > draw_thresh)
    preds[mask] = 1   # Force Draw
    return preds


def evaluate(model, X, y, use_threshold=True, label=""):
    """Đánh giá model, trả về dict kết quả."""
    proba = model.predict_proba(X)
    if use_threshold:
        pred = predict_with_threshold(proba)
    else:
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

# [FIX 5] Tạo feature tương tác
df["elo_x_home_adv"] = df["elo_diff"] * df["home_advantage"]

train_seasons = ["2015/16","2016/17","2017/18","2018/19","2019/20",
                 "2020/21","2021/22","2022/23"]
val_seasons   = ["2023/24"]
test_seasons  = ["2024/25"]

df_train = df[df["Season"].isin(train_seasons)]
df_val   = df[df["Season"].isin(val_seasons)]
df_test  = df[df["Season"].isin(test_seasons)]

X_train = df_train[FEATURE_COLS].values
y_train = df_train[LABEL_COL].values

X_val   = df_val[FEATURE_COLS].values
y_val   = df_val[LABEL_COL].values

X_test  = df_test[FEATURE_COLS].values
y_test  = df_test[LABEL_COL].values

# [FIX 1] Tính sample_weight: tăng trọng số lớp Draw (label=1)
# "balanced" tự tính w_i = n_samples / (n_classes * n_samples_i)
sw_train = compute_sample_weight(class_weight="balanced", y=y_train)

print(f"Train : {len(X_train)} tran")
print(f"Val   : {len(X_val)} tran")
print(f"Test  : {len(X_test)} tran")
print(f"Features: {len(FEATURE_COLS)} (bao gom {len(INTERACTION_FEATURES)} interaction feature)")

# Phân bố label trong train
label_names = {0: "Home Win", 1: "Draw", 2: "Away Win"}
print("\nPhan bo label (train):")
for lbl, cnt in zip(*np.unique(y_train, return_counts=True)):
    w_avg = sw_train[y_train == lbl].mean()
    print(f"  {label_names[lbl]:10s}: {cnt:4d}  sample_weight trung binh = {w_avg:.3f}")
print()


# ─────────────────────────────────────────────────────────────────────────────
# 2. Chuẩn hoá
# ─────────────────────────────────────────────────────────────────────────────

scaler       = StandardScaler()
X_train_sc   = scaler.fit_transform(X_train)
X_val_sc     = scaler.transform(X_val)
X_test_sc    = scaler.transform(X_test)

with open("models/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Cross-Validation (dùng neg_log_loss)
# ─────────────────────────────────────────────────────────────────────────────

print("=" * 65)
print("CROSS-VALIDATION TREN TAP TRAIN (5-fold, scoring=neg_log_loss)")
print("=" * 65)

cv_models = {
    "Logistic Regression": LogisticRegression(
        max_iter=2000, random_state=42, C=1.0,
        class_weight="balanced"),                          # [FIX 1]
    "Random Forest":       RandomForestClassifier(
        n_estimators=200, max_depth=6,
        min_samples_split=5, random_state=42,
        class_weight="balanced"),                          # [FIX 1]
    "XGBoost":             XGBClassifier(
        n_estimators=200, max_depth=3, learning_rate=0.1,
        subsample=0.8, random_state=42,
        eval_metric="mlogloss", verbosity=0),              # sample_weight truyền lúc fit
    "Gradient Boosting":   GradientBoostingClassifier(
        n_estimators=200, max_depth=3,
        learning_rate=0.1, subsample=0.8, random_state=42),
}

for name, model in cv_models.items():
    X_cv = X_train_sc if name == "Logistic Regression" else X_train
    scores = cross_val_score(model, X_cv, y_train,
                              cv=5, scoring="neg_log_loss")
    print(f"  {name:25s}: LogLoss {-scores.mean():.4f} (+/- {scores.std():.4f})")

print()


# ─────────────────────────────────────────────────────────────────────────────
# 4. GridSearchCV — [FIX 3] dùng neg_log_loss làm scoring
# ─────────────────────────────────────────────────────────────────────────────

results = []

# ── 4a. Logistic Regression ──────────────────────────────────────────────────
print("=" * 65)
print("Training Logistic Regression (GridSearch, neg_log_loss)...")
lr_grid = GridSearchCV(
    LogisticRegression(max_iter=2000, random_state=42,
                       class_weight="balanced"),           # [FIX 1]
    {"C": [0.01, 0.1, 1.0, 10.0]},
    cv=5, scoring="neg_log_loss", n_jobs=-1,               # [FIX 3]
)
lr_grid.fit(X_train_sc, y_train)
lr = lr_grid.best_estimator_

res = evaluate(lr, X_val_sc, y_val, label="  LR val")
results.append({"model": "Logistic Regression",
                "val_acc": round(res["acc"], 4),
                "val_loss": round(res["loss"], 4)})
print(f"  Best C: {lr_grid.best_params_}")


# ── 4b. Random Forest ────────────────────────────────────────────────────────
print("Training Random Forest (GridSearch, neg_log_loss)...")
rf_grid = GridSearchCV(
    RandomForestClassifier(random_state=42,
                            class_weight="balanced"),      # [FIX 1]
    {"n_estimators": [100, 200, 300],
     "max_depth":    [4, 6, 8],
     "min_samples_split": [5, 10],
     "min_samples_leaf":  [3, 5]},
    cv=5, scoring="neg_log_loss", n_jobs=-1, verbose=0,    # [FIX 3]
)
rf_grid.fit(X_train, y_train)

# [FIX 4] Calibrate xác suất — dùng cv=5 thay vì cv="prefit" (đã bị xoá từ sklearn>=1.4)
rf_cal = CalibratedClassifierCV(
    RandomForestClassifier(**rf_grid.best_params_, random_state=42,
                            class_weight="balanced"),
    cv=5, method="isotonic",
)
rf_cal.fit(X_train, y_train)

res = evaluate(rf_cal, X_val, y_val, label="  RF val")
results.append({"model": "Random Forest (calibrated)",
                "val_acc": round(res["acc"], 4),
                "val_loss": round(res["loss"], 4)})
print(f"  Best params: {rf_grid.best_params_}")


# ── 4c. XGBoost ──────────────────────────────────────────────────────────────
print("Training XGBoost (GridSearch, neg_log_loss)...")
xgb_grid = GridSearchCV(
    XGBClassifier(random_state=42, eval_metric="mlogloss", verbosity=0),
    {"n_estimators":     [100, 200, 300],
     "max_depth":        [3, 5],
     "learning_rate":    [0.05, 0.1],
     "subsample":        [0.8, 1.0],
     "colsample_bytree": [0.7, 1.0],
     "reg_alpha":        [0, 0.1],
     "reg_lambda":       [1, 2]},
    cv=5, scoring="neg_log_loss", n_jobs=-1, verbose=0,    # [FIX 3]
)
# [FIX 1] Truyền sample_weight cho XGBoost (không hỗ trợ class_weight trực tiếp)
xgb_grid.fit(X_train, y_train, sample_weight=sw_train)
xgb_raw = xgb_grid.best_estimator_

# [FIX 4] Calibrate — cv=5 thay vi cv="prefit"
xgb_cal = CalibratedClassifierCV(
    XGBClassifier(**xgb_grid.best_params_, random_state=42,
                  eval_metric="mlogloss", verbosity=0),
    cv=5, method="isotonic",
)
xgb_cal.fit(X_train, y_train, sample_weight=sw_train)

res = evaluate(xgb_cal, X_val, y_val, label="  XGB val")
results.append({"model": "XGBoost (calibrated)",
                "val_acc": round(res["acc"], 4),
                "val_loss": round(res["loss"], 4)})
print(f"  Best params: {xgb_grid.best_params_}")


# ── 4d. Gradient Boosting ────────────────────────────────────────────────────
print("Training Gradient Boosting (GridSearch, neg_log_loss)...")
gb_grid = GridSearchCV(
    GradientBoostingClassifier(random_state=42),
    {"n_estimators":     [100, 200, 300],
     "max_depth":        [3, 5],
     "learning_rate":    [0.05, 0.1],
     "subsample":        [0.8, 1.0],
     "min_samples_split": [5, 10]},
    cv=5, scoring="neg_log_loss", n_jobs=-1, verbose=0,    # [FIX 3]
)
gb_grid.fit(X_train, y_train, sample_weight=sw_train)
gb = gb_grid.best_estimator_

# [FIX 4] Calibrate — cv=5 thay vi cv="prefit"
gb_cal = CalibratedClassifierCV(
    GradientBoostingClassifier(**gb_grid.best_params_, random_state=42),
    cv=5, method="isotonic",
)
gb_cal.fit(X_train, y_train, sample_weight=sw_train)

res = evaluate(gb_cal, X_val, y_val, label="  GB val")
results.append({"model": "Gradient Boosting (calibrated)",
                "val_acc": round(res["acc"], 4),
                "val_loss": round(res["loss"], 4)})
print(f"  Best params: {gb_grid.best_params_}")


# ─────────────────────────────────────────────────────────────────────────────
# 5. So sánh và chọn model tốt nhất theo val_loss  [FIX 3]
# ─────────────────────────────────────────────────────────────────────────────

print()
print("=" * 65)
print("BANG SO SANH TREN VALIDATION SET (2023/24)")
print("=" * 65)
df_compare = pd.DataFrame(results)[["model", "val_acc", "val_loss"]]
print(df_compare.to_string(index=False))
df_compare.to_csv("models/model_compare.csv", index=False)

# [FIX 3] Chọn theo val_loss THAY VÌ val_acc
best_idx  = df_compare["val_loss"].idxmin()
best_name = df_compare.loc[best_idx, "model"]
print(f"\nModel tot nhat (theo val_loss): {best_name}")

model_map = {
    "Logistic Regression":           (lr,      X_test_sc, X_val_sc),
    "Random Forest (calibrated)":    (rf_cal,  X_test,    X_val),
    "XGBoost (calibrated)":          (xgb_cal, X_test,    X_val),
    "Gradient Boosting (calibrated)":(gb_cal,  X_test,    X_val),
}
best_model, X_test_best, X_val_best = model_map[best_name]


# ─────────────────────────────────────────────────────────────────────────────
# 6. Đánh giá trên Test Set (2024/25) — so sánh có/không threshold tuning
# ─────────────────────────────────────────────────────────────────────────────

print()
print("=" * 65)
print("DANH GIA MODEL TOT NHAT TREN TEST SET (2024/25)")
print("=" * 65)

# Không dùng threshold (baseline)
res_raw   = evaluate(best_model, X_test_best, y_test, use_threshold=False,
                     label="  Khong threshold")
# Có dùng threshold [FIX 2]
res_tuned = evaluate(best_model, X_test_best, y_test, use_threshold=True,
                     label="  Co threshold  ")

print(f"\n  >> Su dung du doan co threshold tuning")
print()
print("Classification Report (co threshold):")
print(classification_report(y_test, res_tuned["pred"],
      target_names=["Home Win", "Draw", "Away Win"]))


# ── So sánh tất cả models trên test set ──────────────────────────────────────
print("=" * 65)
print("SO SANH TAT CA MODELS TREN TEST SET")
print("=" * 65)
all_test = []
for name, (model, X_t, _) in model_map.items():
    r = evaluate(model, X_t, y_test, use_threshold=True)
    all_test.append({"model": name,
                     "test_acc":  round(r["acc"],  4),
                     "test_loss": round(r["loss"], 4)})
print(pd.DataFrame(all_test).to_string(index=False))


# ─────────────────────────────────────────────────────────────────────────────
# 7. Vẽ biểu đồ
# ─────────────────────────────────────────────────────────────────────────────

# Confusion matrix
cm = confusion_matrix(y_test, res_tuned["pred"])
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Home Win", "Draw", "Away Win"],
            yticklabels=["Home Win", "Draw", "Away Win"])
plt.title(f"Confusion Matrix - {best_name} (Test 2024/25)")
plt.ylabel("Ket qua thuc te")
plt.xlabel("Du doan (co threshold)")
plt.tight_layout()
plt.savefig("reports/confusion_matrix.png", dpi=150)
plt.close()
print("\nDa luu: reports/confusion_matrix.png")

# Phân phối xác suất Draw — kiểm tra calibration
draw_proba = res_tuned["proba"][:, 1]
plt.figure(figsize=(6, 4))
plt.hist(draw_proba, bins=20, color="steelblue", edgecolor="white")
plt.axvline(DRAW_PROB_THRESHOLD, color="red", linestyle="--",
            label=f"Draw threshold = {DRAW_PROB_THRESHOLD}")
plt.title("Phan phoi xac suat du doan Draw")
plt.xlabel("P(Draw)")
plt.ylabel("So tran")
plt.legend()
plt.tight_layout()
plt.savefig("reports/draw_prob_dist.png", dpi=150)
plt.close()
print("Da luu: reports/draw_prob_dist.png")

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
    "draw_prob_threshold":  DRAW_PROB_THRESHOLD,
    "home_away_gap":        HOME_AWAY_GAP,
    "predict_fn":           predict_with_threshold,   # helper đi kèm model
    "test_acc":             round(res_tuned["acc"],  4),
    "test_loss":            round(res_tuned["loss"], 4),
    "version":              3,
}

with open("models/model_best.pkl", "wb") as f:
    pickle.dump(model_data, f)

print()
print(f"Da luu model: models/model_best.pkl  (version 3)")
print(f"Da luu scaler: models/scaler.pkl")
print()
print("Buoc tiep theo: chay predict.py")
print()
print("=" * 65)
print("TOM TAT THAY DOI v2 → v3")
print("=" * 65)
print("  [FIX 1] class_weight='balanced' + compute_sample_weight → Draw duoc hoc nhieu hon")
print("  [FIX 2] predict_with_threshold(): ep Draw khi P(H) va P(A) sat nhau")
print("  [FIX 3] GridSearch scoring='neg_log_loss', chon model theo val_loss")
print("  [FIX 4] CalibratedClassifierCV(method='isotonic') cho tree-based models")
print("  [FIX 5] Feature moi: elo_diff * home_advantage (tuong tac phi tuyen)")