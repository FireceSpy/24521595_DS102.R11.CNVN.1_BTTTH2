
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path


# ==========================================
# 1. CAU HINH
# ==========================================

ROOT_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = (
    ROOT_DIR / "Data" / "Vectorized" / "tfidf_data.npz"
)

OUTPUT_DIR = Path(__file__).resolve().parent / "Results"

NUM_CLASSES = 10
NUM_FEATURES = 5000

LEARNING_RATE = 0.5
EPOCHS = 1000
LAMBDA = 0.01

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


# ==========================================
# 2. DOC DU LIEU
# ==========================================

def load_data(data, X_name, y_name):

    X = data[X_name].astype(np.float64)
    y = data[y_name].astype(np.int64)

    if X.shape[1] != NUM_FEATURES:
        raise ValueError(
            f"So dac trung khong hop le: {X.shape[1]}"
        )

    if np.any(y < 0) or np.any(y >= NUM_CLASSES):
        raise ValueError("Phat hien nhan khong hop le.")

    return X, y


# ==========================================
# 3. SOFTMAX REGRESSION MODEL
# ==========================================

class SoftmaxRegression:

    def __init__(
        self,
        learning_rate=0.5,
        epochs=1000,
        reg_lambda=0.01
    ):

        self.learning_rate = learning_rate
        self.epochs = epochs
        self.reg_lambda = reg_lambda

        self.W = None
        self.b = None

        self.train_losses = []
        self.validation_losses = []

    # --------------------------------------
    # Ham Softmax
    # --------------------------------------

    def softmax(self, Z):

        # Tru gia tri lon nhat de tranh tran so
        Z = Z - np.max(Z, axis=1, keepdims=True)

        exp_Z = np.exp(Z)

        probabilities = (
            exp_Z / np.sum(exp_Z, axis=1, keepdims=True)
        )

        return probabilities

    # --------------------------------------
    # Du doan xac suat
    # --------------------------------------

    def predict_proba(self, X):

        Z = np.dot(X, self.W) + self.b

        return self.softmax(Z)

    # --------------------------------------
    # Tinh Categorical Cross-Entropy Loss
    # --------------------------------------

    def compute_loss(self, X, y):

        probabilities = self.predict_proba(X)

        number_of_samples = len(y)

        # Lay xac suat cua nhan dung
        correct_probabilities = probabilities[
            np.arange(number_of_samples),
            y
        ]

        correct_probabilities = np.clip(
            correct_probabilities,
            1e-15,
            1.0
        )

        data_loss = -np.mean(
            np.log(correct_probabilities)
        )

        # L2 Regularization
        regularization = (
            self.reg_lambda / 2
        ) * np.sum(self.W ** 2)

        return data_loss + regularization

    # --------------------------------------
    # Huan luyen mo hinh
    # --------------------------------------


    def fit(
        self,
        X_train,
        y_train,
        X_validation,
        y_validation
    ):

        number_of_samples, number_of_features = X_train.shape

        # Khoi tao trong so va bias
        self.W = np.zeros(
            (number_of_features, NUM_CLASSES),
            dtype=np.float64
        )

        self.b = np.zeros(NUM_CLASSES, dtype=np.float64)

        # One-hot encoding
        Y_train = np.zeros(
            (number_of_samples, NUM_CLASSES),
            dtype=np.float64
        )

        Y_train[np.arange(number_of_samples), y_train] = 1.0

        # Early Stopping
        best_validation_loss = float("inf")

        best_W = None
        best_b = None

        patience = 100
        patience_counter = 0

        print("\n========== MODEL TRAINING ==========")

        for epoch in range(self.epochs):

            # Forward propagation
            probabilities = self.predict_proba(X_train)

            # Tinh sai so
            errors = probabilities - Y_train

            # Gradient
            dW = (
                np.dot(X_train.T, errors) / number_of_samples
                + self.reg_lambda * self.W
            )

            db = np.mean(errors, axis=0)

            # Cap nhat tham so
            self.W -= self.learning_rate * dW
            self.b -= self.learning_rate * db

            # Tinh Loss
            train_loss = self.compute_loss(
                X_train,
                y_train
            )

            validation_loss = self.compute_loss(
                X_validation,
                y_validation
            )

            self.train_losses.append(train_loss)
            self.validation_losses.append(validation_loss)

            # Kiem tra Validation Loss tot nhat
            if validation_loss < best_validation_loss:

                best_validation_loss = validation_loss

                best_W = self.W.copy()
                best_b = self.b.copy()

                patience_counter = 0

            else:

                patience_counter += 1

            # In tien trinh
            if (epoch + 1) % 100 == 0:

                print(
                    f"Epoch {epoch + 1}/{self.epochs} | "
                    f"Train Loss: {train_loss:.6f} | "
                    f"Validation Loss: {validation_loss:.6f} | "
                    f"Best Validation Loss: {best_validation_loss:.6f}"
                )

            # Dung som neu khong cai thien
            if patience_counter >= patience:

                print(
                    f"\nEarly Stopping tai Epoch {epoch + 1}"
                )

                break

        # Khoi phuc mo hinh tot nhat
        self.W = best_W
        self.b = best_b

        print(
            f"\nBest Validation Loss: {best_validation_loss:.6f}"
        )

    # --------------------------------------
    # Du doan nhan
    # --------------------------------------

    def predict(self, X):

        probabilities = self.predict_proba(X)

        return np.argmax(probabilities, axis=1)


# ==========================================
# 4. DANH GIA MO HINH
# ==========================================

def evaluate_model(y_true, y_pred):

    confusion_matrix = np.zeros(
        (NUM_CLASSES, NUM_CLASSES),
        dtype=np.int64
    )

    # Tao Confusion Matrix
    for actual, predicted in zip(y_true, y_pred):

        confusion_matrix[actual, predicted] += 1

    print("\n========== CONFUSION MATRIX ==========")

    print("Rows: Actual | Columns: Predicted")

    print("       ", end="")

    for i in range(NUM_CLASSES):
        print(f"{i:5d}", end="")

    print()

    for i in range(NUM_CLASSES):

        print(f"{i:5d}  ", end="")

        for j in range(NUM_CLASSES):
            print(f"{confusion_matrix[i, j]:5d}", end="")

        print()

    # Accuracy
    accuracy = np.trace(confusion_matrix) / len(y_true)

    precisions = []
    recalls = []
    f1_scores = []

    print("\n========== CLASSIFICATION REPORT ==========")

    print(
        f"{'Class':<25}"
        f"{'Precision':>12}"
        f"{'Recall':>12}"
        f"{'F1-score':>12}"
    )

    for i in range(NUM_CLASSES):

        true_positive = confusion_matrix[i, i]

        false_positive = (
            np.sum(confusion_matrix[:, i]) - true_positive
        )

        false_negative = (
            np.sum(confusion_matrix[i, :]) - true_positive
        )

        precision = (
            true_positive / (true_positive + false_positive)
            if true_positive + false_positive > 0
            else 0.0
        )

        recall = (
            true_positive / (true_positive + false_negative)
            if true_positive + false_negative > 0
            else 0.0
        )

        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall > 0
            else 0.0
        )

        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)

        print(
            f"{CLASS_NAMES[i]:<25}"
            f"{precision:>12.4f}"
            f"{recall:>12.4f}"
            f"{f1:>12.4f}"
        )

    # Macro Average
    macro_precision = np.mean(precisions)
    macro_recall = np.mean(recalls)
    macro_f1 = np.mean(f1_scores)

    print("\n========== OVERALL EVALUATION ==========")

    print(f"Accuracy        : {accuracy:.4f}")
    print(f"Macro Precision : {macro_precision:.4f}")
    print(f"Macro Recall    : {macro_recall:.4f}")
    print(f"Macro F1-score  : {macro_f1:.4f}")

    print(
        f"\nSo bai du doan dung: "
        f"{np.trace(confusion_matrix)}/{len(y_true)}"
    )

    return confusion_matrix


# ==========================================
# 5. VE BIEU DO LOSS
# ==========================================

def plot_loss(model):

    plt.figure(figsize=(10, 6))

    plt.plot(
        model.train_losses,
        label="Training Loss"
    )

    plt.plot(
        model.validation_losses,
        label="Validation Loss"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Categorical Cross-Entropy Loss")

    plt.title("Softmax Regression - Loss Function")

    plt.legend()
    plt.grid(True)

    plt.tight_layout()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    figure_path = OUTPUT_DIR / "loss_curve.png"

    plt.savefig(figure_path, dpi=300)

    # Khong dung plt.show() de tranh cho dong cua so
    plt.close()

    print(f"\nDa luu bieu do Loss: {figure_path}")


# ==========================================
# 6. LUU MO HINH
# ==========================================

def save_model(model):

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    model_path = OUTPUT_DIR / "softmax_model.npz"

    np.savez(
        model_path,
        W=model.W,
        b=model.b,
        learning_rate=np.array(model.learning_rate),
        epochs=np.array(model.epochs),
        reg_lambda=np.array(model.reg_lambda)
    )

    print(f"Da luu mo hinh: {model_path}")


# ==========================================
# 7. CHUONG TRINH CHINH
# ==========================================


def main():

    print("========== LOAD DATA ==========")

    if not DATA_PATH.exists():

        raise FileNotFoundError(
            f"Khong tim thay du lieu: {DATA_PATH}"
        )

    data = np.load(DATA_PATH)

    # Doc Train
    X_train, y_train = load_data(
        data,
        "X_train",
        "y_train"
    )

    # Doc Validation
    X_validation, y_validation = load_data(
        data,
        "X_validation",
        "y_validation"
    )

    # Doc Test
    X_test, y_test = load_data(
        data,
        "X_test",
        "y_test"
    )

    print(f"Train: {X_train.shape}")
    print(f"Validation: {X_validation.shape}")
    print(f"Test: {X_test.shape}")

    # Kiem tra phan bo nhan
    print("\n========== KIEM TRA NHAN ==========")

    for i in range(NUM_CLASSES):

        train_count = np.sum(y_train == i)

        validation_count = np.sum(
            y_validation == i
        )

        test_count = np.sum(y_test == i)

        print(
            f"Label {i} - {CLASS_NAMES[i]}: "
            f"Train {train_count}, "
            f"Validation {validation_count}, "
            f"Test {test_count}"
        )

    # Khoi tao mo hinh
    print("\n========== KHOI TAO MODEL ==========")

    model = SoftmaxRegression(
        learning_rate=LEARNING_RATE,
        epochs=EPOCHS,
        reg_lambda=LAMBDA
    )

    # Huan luyen
    model.fit(
        X_train,
        y_train,
        X_validation,
        y_validation
    )

    # Danh gia Validation
    print("\n========== VALIDATION EVALUATION ==========")

    validation_predictions = model.predict(
        X_validation
    )

    evaluate_model(
        y_validation,
        validation_predictions
    )

    # Danh gia Test
    print("\n========== TEST EVALUATION ==========")

    print(f"Test: {X_test.shape}")

    test_predictions = model.predict(
        X_test
    )

    evaluate_model(
        y_test,
        test_predictions
    )

    # Ve bieu do Loss
    print("\n========== LOSS VISUALIZATION ==========")

    plot_loss(model)

    # Luu mo hinh
    save_model(model)

    print("\n========== HOAN TAT ==========")


if __name__ == "__main__":
    main()
