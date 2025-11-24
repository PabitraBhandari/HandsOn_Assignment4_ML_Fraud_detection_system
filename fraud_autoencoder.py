"""
Hands-On Assignment 4:
Unsupervised Fraud Detection Using PyOD AutoEncoder
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score

from pyod.models.auto_encoder import AutoEncoder


# -----------------------------------------------------------
# Data Loading
# -----------------------------------------------------------
def load_data(file_path: str) -> pd.DataFrame:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at: {path.resolve()}")

    df = pd.read_csv(path)
    return df


# -----------------------------------------------------------
# Preprocessing
# -----------------------------------------------------------
def prepare_features(df: pd.DataFrame):
    if "Class" not in df.columns:
        raise ValueError("Dataset must contain 'Class' column.")

    X = df.drop("Class", axis=1).values
    y = df["Class"].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y


# -----------------------------------------------------------
# Build AutoEncoder (Student B version)
# -----------------------------------------------------------
def build_model(contamination: float) -> AutoEncoder:
    """
    Student B uses:
    - More neurons
    - Slightly shorter training
    - Different batch size
    """

    model = AutoEncoder(
        hidden_neuron_list=[64, 32, 64],
        epoch_num=20,
        batch_size=128,
        dropout_rate=0.05,
        contamination=contamination,
        preprocessing=False,         
        optimizer_name="adam",
        hidden_activation_name="relu",
        verbose=1,
    )
    return model


# -----------------------------------------------------------
# Evaluation
# -----------------------------------------------------------
def evaluate(model, X, y_true):
    print("\nEvaluating model...")

    y_pred = model.labels_
    scores = model.decision_scores_

    cm = confusion_matrix(y_true, y_pred)
    print("\nConfusion Matrix:")
    print(cm)

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, digits=4))

    try:
        auc = roc_auc_score(y_true, scores)
        print(f"ROC-AUC Score: {auc:.4f}")
    except Exception:
        print("ROC-AUC could not be calculated.")


# -----------------------------------------------------------
# Main Pipeline
# -----------------------------------------------------------
def main():
    np.random.seed(42)

    data_path = "data/creditcard.csv"
    print("Loading dataset...")
    df = load_data(data_path)

    print(f"\nDataset Shape: {df.shape}")
    print(df.head())

    print("\nClass Distribution:")
    print(df["Class"].value_counts())

    print("\nPreparing features...")
    X_scaled, y = prepare_features(df)

    contamination = y.mean()
    print(f"\nEstimated contamination: {contamination:.6f}")

    print("\nBuilding model...")
    model = build_model(contamination)

    print("\nTraining model...")
    model.fit(X_scaled)
    print("Training complete.")

    evaluate(model, X_scaled, y)


if __name__ == "__main__":
    main()