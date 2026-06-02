"""
Tao lai cac report tu model_best.pkl hien tai.

Muc dich:
  - Dam bao reports/confusion_matrix.png va reports/feature_importance.png
    luon khop voi model dang duoc dashboard/predict.py su dung.

Chay:
  MPLBACKEND=Agg MPLCONFIGDIR=/private/tmp venv/bin/python generate_reports.py
"""

import os
import pickle
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from sklearn.metrics import accuracy_score, confusion_matrix, log_loss

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


CLEAN_PATH = "data/epl_clean.csv"
MODEL_PATH = "models/model_best.pkl"
REPORT_DIR = "reports"
LABELS = ["Home Win", "Draw", "Away Win"]


def predict_with_draw_boost(proba: np.ndarray, draw_boost: float) -> np.ndarray:
    adjusted = proba.copy()
    adjusted[:, 1] += draw_boost
    return np.argmax(adjusted, axis=1)


def get_feature_importance(model, feature_cols):
    if hasattr(model, "coef_"):
        importance = np.mean(np.abs(model.coef_), axis=0)
        return pd.DataFrame({
            "feature": feature_cols,
            "importance": importance,
        })

    if hasattr(model, "feature_importances_"):
        return pd.DataFrame({
            "feature": feature_cols,
            "importance": model.feature_importances_,
        })

    # Voting/ensemble fallback: lay trung binh tu estimator con neu co.
    rows = []
    estimators = getattr(model, "estimators_", [])
    for estimator in estimators:
        if hasattr(estimator, "coef_"):
            rows.append(np.mean(np.abs(estimator.coef_), axis=0))
        elif hasattr(estimator, "feature_importances_"):
            rows.append(estimator.feature_importances_)

    if rows:
        return pd.DataFrame({
            "feature": feature_cols,
            "importance": np.mean(rows, axis=0),
        })

    return None


def main():
    os.makedirs(REPORT_DIR, exist_ok=True)

    with open(MODEL_PATH, "rb") as f:
        model_data = pickle.load(f)

    model = model_data["model"]
    scaler = model_data.get("scaler")
    feature_cols = model_data["feature_cols"]
    model_name = model_data.get("model_name", type(model).__name__)
    draw_boost = float(model_data.get("draw_boost", 0.0))
    version = model_data.get("version", "?")

    df = pd.read_csv(CLEAN_PATH)
    test_season = sorted(df["Season"].unique())[-1]
    df_test = df[df["Season"] == test_season].copy()

    X = df_test[feature_cols].values
    y = df_test["label"].values
    X_in = scaler.transform(X) if scaler is not None else X

    proba = model.predict_proba(X_in)
    pred = predict_with_draw_boost(proba, draw_boost)

    acc = accuracy_score(y, pred)
    loss = log_loss(y, proba)

    cm = confusion_matrix(y, pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=LABELS,
        yticklabels=LABELS,
    )
    plt.title(f"Confusion Matrix - {model_name} {version} (Test {test_season})")
    plt.ylabel("Ket qua thuc te")
    plt.xlabel("Du doan")
    plt.tight_layout()
    cm_path = os.path.join(REPORT_DIR, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=150)
    plt.close()

    feat_df = get_feature_importance(model, feature_cols)
    fi_path = os.path.join(REPORT_DIR, "feature_importance.png")
    if feat_df is not None:
        feat_df = feat_df.sort_values("importance", ascending=True)
        plt.figure(figsize=(8, max(6, len(feature_cols) * 0.28)))
        plt.barh(feat_df["feature"], feat_df["importance"], color="steelblue")
        plt.title(f"Feature Importance - {model_name} {version}")
        plt.xlabel("Importance")
        plt.tight_layout()
        plt.savefig(fi_path, dpi=150)
        plt.close()
    else:
        plt.figure(figsize=(8, 3))
        plt.text(
            0.5,
            0.5,
            f"Feature importance unavailable for {model_name}",
            ha="center",
            va="center",
        )
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(fi_path, dpi=150)
        plt.close()

    print(f"Model        : {model_name} ({version})")
    print(f"Test season  : {test_season}")
    print(f"Accuracy     : {acc:.4f}")
    print(f"Log loss     : {loss:.4f}")
    print(f"Draw boost   : {draw_boost:.2f}")
    print(f"Saved        : {cm_path}")
    print(f"Saved        : {fi_path}")


if __name__ == "__main__":
    main()
