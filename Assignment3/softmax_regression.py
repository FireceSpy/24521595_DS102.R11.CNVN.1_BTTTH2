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


# =========================
# CONFIGURATION
# =========================

ROOT_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = ROOT_DIR / "Data" / "Vectorized" / "tfidf_data.npz"
RESULTS_DIR = Path(__file__).resolve().parent / "Results"

CLASS_NAMES = [
    "Thoi su",
    "The gioi",
    "Kinh doanh",
    "Khoa hoc cong nghe",
    "Bat dong san",
    "Suc khoe",
    "The thao",
    "Giai tri",
    "Phap luat",
    "Giao duc"
]

MODEL_CONFIG = {
    "C": 1.0,
    "solver": "lbfgs",
    "max_iter": 2000,
    "random_state": 42
}


# =========================
# LOAD DATA
# =========================

def load_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Khong tim thay du lieu: {DATA_PATH}")

    data = np.load(DATA_PATH)

    X_train = data["X_train"]
    y_train = data["y_train"]

    X_validation = data["X_validation"]
    y_validation = data["y_validation"]

    X_test = data["X_test"]
    y_test = data["y_test"]

    return (
        X_train, y_train,
        X_validation, y_validation,
        X_test, y_test
    )


# =========================
# EVALUATION
# =========================

def evaluate_model(model, X, y, dataset_name):
    y_pred = model.predict(X)

    accuracy = accuracy_score(y, y_pred)
    report = classification_report(
        y,
        y_pred,
        labels=list(range(10)),
        target_names=CLASS_NAMES,
        digits=4,
        zero_division=0
    )

    print(f"========== {dataset_name.upper()} EVALUATION ==========")
    print(f"Accuracy: {accuracy:.4f}")
    print("\\nClassification Report:")
    print(report)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    report_path = RESULTS_DIR / f"softmax_{dataset_name}_report.txt"
    with open(report_path, "w", encoding="utf-8") as file:
        file.write(f"{dataset_name.upper()} EVALUATION\\n")
        file.write(f"Accuracy: {accuracy:.4f}\\n\\n")
        file.write(report)

    print(f"Da luu bao cao: {report_path}")

    matrix = confusion_matrix(y, y_pred, labels=list(range(10)))

    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=CLASS_NAMES
    )

    fig, ax = plt.subplots(figsize=(12, 10))
    display.plot(
        ax=ax,
        cmap="Blues",
        xticks_rotation=45,
        values_format="d",
        colorbar=False
    )
    ax.set_title(f"Softmax Regression - {dataset_name.capitalize()} Confusion Matrix")
    fig.tight_layout()

    matrix_path = RESULTS_DIR / f"softmax_{dataset_name}_confusion_matrix.png"
    fig.savefig(matrix_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Da luu Confusion Matrix: {matrix_path}\\n")

    return accuracy


# =========================
# MAIN
# =========================

def main():
    print("========== LOAD DATA ==========")

    (
        X_train, y_train,
        X_validation, y_validation,
        X_test, y_test
    ) = load_data()

    print(f"Train      : {X_train.shape}")
    print(f"Validation : {X_validation.shape}")
    print(f"Test       : {X_test.shape}")

    print("\\n========== INITIALIZE MODEL ==========")

    model = LogisticRegression(**MODEL_CONFIG)

    print("\\n========== MODEL TRAINING ==========")
    model.fit(X_train, y_train)

    print("Training completed.")
    print(f"Number of iterations: {model.n_iter_}")

    validation_accuracy = evaluate_model(
        model,
        X_validation,
        y_validation,
        "validation"
    )

    test_accuracy = evaluate_model(
        model,
        X_test,
        y_test,
        "test"
    )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    model_path = RESULTS_DIR / "softmax_regression_model.joblib"
    joblib.dump(model, model_path)

    print("========== FINAL SUMMARY ==========")
    print(f"Validation Accuracy: {validation_accuracy:.4f}")
    print(f"Test Accuracy      : {test_accuracy:.4f}")
    print(f"Da luu mo hinh: {model_path}")
    print("\\n========== COMPLETED ==========")


if __name__ == "__main__":
    main()


