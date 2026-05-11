"""
model.py - Model Training & Evaluation
=======================================
Trains a Random Forest classifier on the phishing dataset,
evaluates it, and saves the model + metadata to disk.

Run this once before starting the Flask app:
    python model.py
"""

import os
import pickle
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_auc_score, precision_score, recall_score, f1_score,
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from utils import extract_feature_vector, FEATURE_NAMES

# ─── Config ────────────────────────────────────────────────────────────────
BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH  = os.path.join(BASE_DIR, "dataset.csv")
MODEL_PATH    = os.path.join(BASE_DIR, "model.pkl")
META_PATH     = os.path.join(BASE_DIR, "model_meta.json")

RANDOM_STATE  = 42
TEST_SIZE     = 0.2
N_FOLDS       = 5


# ─── Data loading ──────────────────────────────────────────────────────────

def load_dataset(path: str) -> pd.DataFrame:
    """Load CSV, drop nulls, and validate required columns."""
    df = pd.read_csv(path)
    required = {"url", "label"}
    if not required.issubset(df.columns):
        raise ValueError(f"Dataset must contain columns: {required}")
    df = df.dropna(subset=["url", "label"])
    df["label"] = df["label"].astype(int)
    print(f"[Dataset] Loaded {len(df)} samples  "
          f"({(df.label == 0).sum()} legitimate, {(df.label == 1).sum()} phishing)")
    return df


# ─── Feature engineering ───────────────────────────────────────────────────

def build_feature_matrix(df: pd.DataFrame):
    """Apply feature extraction to every URL in the dataset."""
    print("[Features] Extracting features...")
    X = np.array([extract_feature_vector(url) for url in df["url"]])
    y = df["label"].values
    print(f"[Features] Matrix shape: {X.shape}")
    return X, y


# ─── Model building ────────────────────────────────────────────────────────

def build_model() -> Pipeline:
    """
    Ensemble-ready pipeline:
      StandardScaler → RandomForestClassifier
    Random Forest is robust to outliers and naturally ranks feature importance.
    """
    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        min_samples_split=4,
        min_samples_leaf=2,
        class_weight="balanced",   # handles class imbalance
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", clf),
    ])
    return pipeline


# ─── Training & evaluation ────────────────────────────────────────────────

def train_and_evaluate(X, y):
    """Train model, run cross-validation, and return trained pipeline + metrics."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    model = build_model()

    # ── Cross-validation ──
    print(f"[Training] Running {N_FOLDS}-fold cross-validation...")
    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="accuracy")
    print(f"[Training] CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # ── Final fit ──
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # ── Metrics ──
    acc       = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall    = recall_score(y_test, y_pred)
    f1        = f1_score(y_test, y_pred)
    roc_auc   = roc_auc_score(y_test, y_prob)

    print("\n" + "═" * 50)
    print("  MODEL EVALUATION REPORT")
    print("═" * 50)
    print(f"  Accuracy  : {acc:.4f}  ({acc*100:.1f}%)")
    print(f"  Precision : {precision:.4f}")
    print(f"  Recall    : {recall:.4f}")
    print(f"  F1 Score  : {f1:.4f}")
    print(f"  ROC-AUC   : {roc_auc:.4f}")
    print("─" * 50)
    print(classification_report(y_test, y_pred,
                                 target_names=["Legitimate", "Phishing"]))

    # ── Feature importances ──
    rf = model.named_steps["clf"]
    importances = dict(zip(FEATURE_NAMES, rf.feature_importances_.tolist()))
    top_features = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:5]
    print("  Top 5 Features:")
    for feat, imp in top_features:
        bar = "█" * int(imp * 50)
        print(f"    {feat:<35} {imp:.4f}  {bar}")
    print("═" * 50 + "\n")

    metrics = {
        "accuracy": round(acc * 100, 2),
        "precision": round(precision * 100, 2),
        "recall": round(recall * 100, 2),
        "f1_score": round(f1 * 100, 2),
        "roc_auc": round(roc_auc * 100, 2),
        "cv_accuracy_mean": round(cv_scores.mean() * 100, 2),
        "cv_accuracy_std": round(cv_scores.std() * 100, 2),
        "train_samples": int(len(X_train)),
        "test_samples": int(len(X_test)),
        "feature_importances": importances,
        "top_features": [f[0] for f in top_features],
    }
    return model, metrics


# ─── Persistence ──────────────────────────────────────────────────────────

def save_artifacts(model, metrics: dict):
    """Pickle the trained model and save metrics as JSON."""
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    print(f"[Saved] Model → {MODEL_PATH}")

    with open(META_PATH, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[Saved] Metadata → {META_PATH}")


def load_model():
    """Load the saved model and metadata. Called by Flask app."""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            "model.pkl not found. Run `python model.py` first to train the model."
        )
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(META_PATH, "r") as f:
        meta = json.load(f)
    return model, meta


# ─── Entry point ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "═" * 50)
    print("  AI-POWERED PHISHING DETECTION — MODEL TRAINER")
    print("═" * 50 + "\n")

    df               = load_dataset(DATASET_PATH)
    X, y             = build_feature_matrix(df)
    model, metrics   = train_and_evaluate(X, y)
    save_artifacts(model, metrics)

    print("✓ Training complete. You can now run: python app.py")