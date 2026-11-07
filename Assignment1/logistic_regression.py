
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# ==========================================
# 1. CAU HINH
# ==========================================

ROOT_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = ROOT_DIR / "Data" / "Vectorized" / "tfidf_data.npz"
OUTPUT_DIR = Path(__file__).resolve().parent / "Results"

# Label 6: The thao -> 0
# Label 7: Giai tri -> 1
CLASS_0 = 6
CLASS_1 = 7

LEARNING_RATE = 0.1
EPOCHS = 1000
LAMBDA = 0.01

THRESHOLD = 0.5


# ==========================================
# 2. DOC VA LOC DU LIEU
# ==========================================

def load_binary_data(data, X_name, y_name):
    X_all = data[X_name]
    y_all = data[y_name]

    # Chi giu hai nhan 6 va 7
    mask = (y_all == CLASS_0) | (y_all == CLASS_1)

    X = X_all[mask]
    original_y = y_all[mask]

    # Chuyen Label 6 -> 0, Label 7 -> 1
    y = (original_y == CLASS_1).astype(np.float32)

    return X.astype(np.float64), y


# ==========================================
# 3. LOGISTIC REGRESSION MODEL
# ==========================================

class LogisticRegression:

    def __init__(self, learning_rate=0.1, epochs=1000, reg_lambda=0.01):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.reg_lambda = reg_lambda

        self.W = None
        self.b = 0.0

        self.train_losses = []
        self.validation_losses = []

    # --------------------------------------
    # Ham Sigmoid
    # --------------------------------------

    def sigmoid(self, z):
        z = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z))

    # --------------------------------------
    # Du doan xac suat
    # --------------------------------------

    def predict_proba(self, X):
        z = np.dot(X, self.W) + self.b
        return self.sigmoid(z)

    # --------------------------------------
    # Tinh Binary Cross-Entropy Loss
    # --------------------------------------

    def compute_loss(self, X, y):
        z = np.dot(X, self.W) + self.b

        # BCE tinh on dinh ve so
        data_loss = np.mean(
            np.logaddexp(0, z) - y * z
        )

        # L2 Regularization, khong phat cho bias
        regularization = (
            self.reg_lambda / 2
        ) * np.sum(self.W ** 2)

        return data_loss + regularization

    # --------------------------------------
    # Huấn luyen bang Batch Gradient Descent
    # --------------------------------------

    def fit(self, X_train, y_train, X_validation, y_validation):

        number_of_samples, number_of_features = X_train.shape

        # Khoi tao trong so
        self.W = np.zeros(number_of_features, dtype=np.float64)
        self.b = 0.0

        for epoch in range(self.epochs):

            # Forward propagation
            probabilities = self.predict_proba(X_train)

            # Sai so du doan
            errors = probabilities - y_train

            # Gradient cua W va b
            dW = (
                np.dot(X_train.T, errors) / number_of_samples
                + self.reg_lambda * self.W
            )

            db = np.mean(errors)

            # Cap nhat tham so
            self.W -= self.learning_rate * dW
            self.b -= self.learning_rate * db

            # Tinh Loss sau khi cap nhat
            train_loss = self.compute_loss(X_train, y_train)
            validation_loss = self.compute_loss(
                X_validation,
                y_validation
            )

            self.train_losses.append(train_loss)
            self.validation_losses.append(validation_loss)

            # In tien trinh
            if (epoch + 1) % 100 == 0:
                print(
                    f"Epoch {epoch + 1}/{self.epochs} | "
                    f"Train Loss: {train_loss:.6f} | "
                    f"Validation Loss: {validation_loss:.6f}"
                )

    # --------------------------------------
    # Du doan nhan
    # --------------------------------------

    def predict(self, X):
        probabilities = self.predict_proba(X)

        return (probabilities >= THRESHOLD).astype(np.int64)


# ==========================================
# 4. CAC CHI SO DANH GIA
# ==========================================

def evaluate_model(y_true, y_pred):

    # Confusion Matrix
    true_positive = np.sum((y_true == 1) & (y_pred == 1))
    true_negative = np.sum((y_true == 0) & (y_pred == 0))
    false_positive = np.sum((y_true == 0) & (y_pred == 1))
    false_negative = np.sum((y_true == 1) & (y_pred == 0))

    accuracy = (
        true_positive + true_negative
    ) / len(y_true)

    precision = (
        true_positive / (true_positive + false_positive)
        if true_positive + false_positive > 0 else 0.0
    )

    recall = (
        true_positive / (true_positive + false_negative)
        if true_positive + false_negative > 0 else 0.0
    )

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall > 0 else 0.0
    )

    print("\n========== CONFUSION MATRIX ==========")
    print("                 Predicted")
    print("                 0       1")
    print(f"Actual 0       {true_negative:4d}    {false_positive:4d}")
    print(f"Actual 1       {false_negative:4d}    {true_positive:4d}")

    print("\n========== EVALUATION ==========")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-score : {f1:.4f}")


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
    plt.ylabel("Binary Cross-Entropy Loss")
    plt.title("Logistic Regression - Loss Function")

    plt.legend()
    plt.grid(True)

    plt.tight_layout()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    figure_path = OUTPUT_DIR / "loss_curve.png"

    plt.savefig(figure_path, dpi=300)
    plt.show()

    print(f"\nDa luu bieu do Loss: {figure_path}")


# ==========================================
# 6. LUU MO HINH
# ==========================================

def save_model(model):

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    model_path = OUTPUT_DIR / "logistic_model.npz"

    np.savez(
        model_path,
        W=model.W,
        b=np.array(model.b),
        learning_rate=np.array(model.learning_rate),
        epochs=np.array(model.epochs),
        reg_lambda=np.array(model.reg_lambda),
        class_0=np.array(CLASS_0),
        class_1=np.array(CLASS_1)
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

    X_train, y_train = load_binary_data(
        data, "X_train", "y_train"
    )

    X_validation, y_validation = load_binary_data(
        data, "X_validation", "y_validation"
    )

    # Khong su dung Test trong giai doan huan luyen
    print(f"Train: {X_train.shape}")
    print(f"Validation: {X_validation.shape}")

    print("\n========== KIEM TRA NHAN ==========")

    print(
        f"Train - The thao (0): {np.sum(y_train == 0)}"
    )
    print(
        f"Train - Giai tri (1): {np.sum(y_train == 1)}"
    )

    print(
        f"Validation - The thao (0): {np.sum(y_validation == 0)}"
    )
    print(
        f"Validation - Giai tri (1): {np.sum(y_validation == 1)}"
    )

    # Khoi tao mo hinh
    print("\n========== KHOI TAO MODEL ==========")

    model = LogisticRegression(
        learning_rate=LEARNING_RATE,
        epochs=EPOCHS,
        reg_lambda=LAMBDA
    )

    # Huan luyen
    print("\n========== TRAINING ==========")

    model.fit(
        X_train,
        y_train,
        X_validation,
        y_validation
    )

    # Danh gia tren Validation
    print("\n========== VALIDATION EVALUATION ==========")

    validation_predictions = model.predict(X_validation)

    evaluate_model(
        y_validation.astype(np.int64),
        validation_predictions
    )
    
    # ======================================
    # DANH GIA CUOI CUNG TREN TEST
    # ======================================

    print("\n========== TEST EVALUATION ==========")

    X_test, y_test = load_binary_data(
        data, "X_test", "y_test"
    )

    print(f"Test: {X_test.shape}")

    print(
        f"Test - The thao (0): {np.sum(y_test == 0)}"
    )

    print(
        f"Test - Giai tri (1): {np.sum(y_test == 1)}"
    )

    test_predictions = model.predict(X_test)

    evaluate_model(
        y_test.astype(np.int64),
        test_predictions
    )
    # Ve Loss
    print("\n========== LOSS VISUALIZATION ==========")

    plot_loss(model)

    # Luu mo hinh
    save_model(model)

    print("\n========== HOAN TAT ==========")


if __name__ == "__main__":
    main()
