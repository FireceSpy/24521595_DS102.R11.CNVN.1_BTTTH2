from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import joblib

from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)


# =========================================================
# 1. CONFIGURATION
# =========================================================

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT_DIR / "Data" / "Vectorized" / "tfidf_data.npz"
RESULTS_DIR = Path(__file__).resolve().parent / "Results"

# 6 = The thao, 7 = Giai tri
SELECTED_LABELS = [6, 7]
CLASS_NAMES = ["The thao", "Giai tri"]

MODEL_CONFIG = {
    "C": 1.0,
    "solver": "lbfgs",
    "max_iter": 2000,
    "random_state": 42
}


# =========================================================
# 2. LOAD AND PREPARE DATA
# =========================================================

def load_data():
    data = np.load(DATA_PATH)

    # Read the same Train / Validation / Test splits
    X_train, y_train = data["X_train"], data["y_train"]
    X_val, y_val = data["X_validation"], data["y_validation"]
    X_test, y_test = data["X_test"], data["y_test"]

    # Keep only Sports (6) and Entertainment (7)
    def select_classes(X, y):
        mask = np.isin(y, SELECTED_LABELS)
        X = X[mask]
        y = np.where(y[mask] == 6, 0, 1)
        return X, y

    X_train, y_train = select_classes(X_train, y_train)
    X_val, y_val = select_classes(X_val, y_val)
    X_test, y_test = select_classes(X_test, y_test)

    return X_train, y_train, X_val, y_val, X_test, y_test


# =========================================================
# 3. EVALUATE AND SAVE RESULTS
# =========================================================

def evaluate_model(model, X, y, dataset_name):
    predictions = model.predict(X)

    accuracy = accuracy_score(y, predictions)
    report = classification_report(
        y,
        predictions,
        labels=[0, 1],
        target_names=CLASS_NAMES,
        digits=4,
        zero_division=0
    )

    print(f"\n========== {dataset_name} EVALUATION ==========")
    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(report)

    # Save classification report
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = RESULTS_DIR / f"logistic_{dataset_name.lower()}_report.txt"
    report_path.write_text(report, encoding="utf-8")

    # Save confusion matrix image
    cm = confusion_matrix(y, predictions, labels=[0, 1])
    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=CLASS_NAMES
    )
    display.plot(cmap="Blues", values_format="d")
    plt.title(f"Logistic Regression - {dataset_name} Confusion Matrix")
    plt.tight_layout()

    figure_path = RESULTS_DIR / (
        f"logistic_{dataset_name.lower()}_confusion_matrix.png"
    )
    plt.savefig(figure_path, dpi=300)
    plt.close()

    print(f"Da luu bao cao: {report_path}")
    print(f"Da luu Confusion Matrix: {figure_path}")

    return accuracy


# =========================================================
# 4. MAIN
# =========================================================

def main():
    print("========== LOAD DATA ==========")

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Khong tim thay: {DATA_PATH}")

    (
        X_train, y_train,
        X_val, y_val,
        X_test, y_test
    ) = load_data()

    print(f"Train      : {X_train.shape}")
    print(f"Validation : {X_val.shape}")
    print(f"Test       : {X_test.shape}")

    print("\n========== INITIALIZE MODEL ==========")

    # Scikit-learn implements Logistic Regression internally.
    model = LogisticRegression(**MODEL_CONFIG)

    print("\n========== MODEL TRAINING ==========")
    model.fit(X_train, y_train)
    print("Training completed.")
    print(f"Number of iterations: {model.n_iter_[0]}")

    # Use Validation for model evaluation.
    val_accuracy = evaluate_model(
        model, X_val, y_val, "Validation"
    )

    # Use Test for final evaluation.
    test_accuracy = evaluate_model(
        model, X_test, y_test, "Test"
    )

    # Save trained model
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = RESULTS_DIR / "logistic_regression_model.joblib"
    joblib.dump(model, model_path)

    print("\n========== FINAL SUMMARY ==========")
    print(f"Validation Accuracy: {val_accuracy:.4f}")
    print(f"Test Accuracy      : {test_accuracy:.4f}")
    print(f"Da luu mo hinh: {model_path}")
    print("\n========== COMPLETED ==========")


if __name__ == "__main__":
    main()

