
import json
import re
import math
import os
from pathlib import Path
from collections import Counter

import numpy as np


# ==========================================
# 1. CAU HINH
# ==========================================

ROOT_DIR = Path(__file__).resolve().parent.parent

TRAIN_PATH = ROOT_DIR / "Data" / "Splits" / "train.json"
VALIDATION_PATH = ROOT_DIR / "Data" / "Splits" / "validation.json"
TEST_PATH = ROOT_DIR / "Data" / "Splits" / "test.json"

OUTPUT_DIR = ROOT_DIR / "Data" / "Vectorized"

# So luong dac trung toi da
MAX_FEATURES = 5000

# Chi giu tu xuat hien trong it nhat 2 bai
MIN_DF = 2

# Loai tu xuat hien trong hon 95% so bai
MAX_DF_RATIO = 0.95

# Su dung unigram va bigram
USE_BIGRAM = True


# ==========================================
# 2. DOC DU LIEU
# ==========================================

def load_data(file_path):
    if not file_path.exists():
        raise FileNotFoundError(
            f"Khong tim thay file: {file_path}"
        )

    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, list):
        raise ValueError(
            f"Du lieu trong {file_path.name} phai la mot danh sach."
        )

    for index, article in enumerate(data):
        if "text" not in article:
            raise ValueError(
                f"Bai viet thu {index + 1} khong co truong text."
            )

        if "label" not in article:
            raise ValueError(
                f"Bai viet thu {index + 1} khong co truong label."
            )

        if not article["text"].strip():
            raise ValueError(
                f"Bai viet thu {index + 1} co noi dung rong."
            )

    return data


# ==========================================
# 3. TOKENIZATION
# ==========================================

def tokenize(text):
    """
    Tach van ban thanh cac token.
    Giu nguyen dau tieng Viet.
    """

    text = text.lower()

    # Tach tu, loai bo dau cau va khoang trang
    words = re.findall(r"[^\W_]+", text, flags=re.UNICODE)

    tokens = words.copy()

    # Tao bigram: hai tu lien tiep
    if USE_BIGRAM:
        for i in range(len(words) - 1):
            bigram = words[i] + "_" + words[i + 1]
            tokens.append(bigram)

    return tokens


# ==========================================
# 4. XAY DUNG VOCABULARY
# ==========================================

def build_vocabulary(train_data):
    """
    Chi xay dung Vocabulary tu tap Train.
    """

    document_frequency = Counter()

    for article in train_data:
        tokens = tokenize(article["text"])

        # Moi token chi dem 1 lan trong moi bai
        unique_tokens = set(tokens)

        document_frequency.update(unique_tokens)

    total_documents = len(train_data)

    max_document_frequency = math.floor(
        MAX_DF_RATIO * total_documents
    )

    # Loc cac token theo MIN_DF va MAX_DF
    valid_tokens = []

    for token, frequency in document_frequency.items():
        if MIN_DF <= frequency <= max_document_frequency:
            valid_tokens.append((token, frequency))

    # Sap xep theo tan suat giam dan
    # Neu bang nhau thi sap xep theo ten token
    valid_tokens.sort(
        key=lambda item: (-item[1], item[0])
    )

    # Gioi han so luong dac trung
    valid_tokens = valid_tokens[:MAX_FEATURES]

    # Gan moi token mot vi tri
    vocabulary = {}

    for index, (token, frequency) in enumerate(valid_tokens):
        vocabulary[token] = index

    print(f"So luong bai Train: {total_documents}")
    print(f"So token sau khi loc: {len(vocabulary)}")

    return vocabulary, document_frequency


# ==========================================
# 5. TINH IDF
# ==========================================

def calculate_idf(vocabulary, document_frequency, total_documents):
    """
    Cong thuc Smooth IDF:

    IDF(t) = log((N + 1) / (DF(t) + 1)) + 1
    """

    idf = np.zeros(len(vocabulary), dtype=np.float32)

    for token, index in vocabulary.items():
        df = document_frequency[token]

        idf[index] = (
            np.log(
                (total_documents + 1) / (df + 1)
            ) + 1
        )

    return idf


# ==========================================
# 6. CHUYEN VAN BAN THANH TF-IDF
# ==========================================

def transform_data(data, vocabulary, idf):
    """
    Chuyen danh sach bai viet thanh ma tran TF-IDF.

    TF = 1 + log(so lan xuat hien)
    TF-IDF = TF * IDF

    Sau do chuan hoa L2 cho tung bai viet.
    """

    number_of_documents = len(data)
    number_of_features = len(vocabulary)

    X = np.zeros(
        (number_of_documents, number_of_features),
        dtype=np.float32
    )

    y = np.zeros(number_of_documents, dtype=np.int64)

    for document_index, article in enumerate(data):

        tokens = tokenize(article["text"])

        token_counts = Counter(tokens)

        for token, count in token_counts.items():

            if token not in vocabulary:
                continue

            feature_index = vocabulary[token]

            # Sublinear Term Frequency
            tf = 1.0 + np.log(count)

            # TF-IDF
            X[document_index, feature_index] = (
                tf * idf[feature_index]
            )

        # L2 Normalization
        row_norm = np.linalg.norm(X[document_index])

        if row_norm > 0:
            X[document_index] /= row_norm

        y[document_index] = int(article["label"])

    return X, y


# ==========================================
# 7. LUU VOCABULARY VA CAU HINH
# ==========================================

def save_metadata(vocabulary):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    vocabulary_path = OUTPUT_DIR / "vocabulary.json"

    with open(vocabulary_path, "w", encoding="utf-8") as file:
        json.dump(
            vocabulary,
            file,
            ensure_ascii=False,
            indent=4
        )

    config = {
        "max_features": MAX_FEATURES,
        "min_df": MIN_DF,
        "max_df_ratio": MAX_DF_RATIO,
        "use_bigram": USE_BIGRAM,
        "number_of_features": len(vocabulary),
        "normalization": "L2",
        "tf_method": "1 + log(count)",
        "idf_method": "log((N + 1) / (DF + 1)) + 1"
    }

    config_path = OUTPUT_DIR / "vectorizer_config.json"

    with open(config_path, "w", encoding="utf-8") as file:
        json.dump(
            config,
            file,
            ensure_ascii=False,
            indent=4
        )

    print(f"Da luu Vocabulary: {vocabulary_path}")
    print(f"Da luu cau hinh: {config_path}")


# ==========================================
# 8. KIEM TRA KET QUA
# ==========================================

def print_result(name, X, y):
    print(f"\n{name}:")
    print(f"  So bai: {X.shape[0]}")
    print(f"  So dac trung: {X.shape[1]}")
    print(f"  Kich thuoc ma tran: {X.shape}")

    print("  So bai theo nhan:")

    labels, counts = np.unique(y, return_counts=True)

    for label, count in zip(labels, counts):
        print(f"    Label {label}: {count} bai")

    zero_rows = np.sum(
        np.linalg.norm(X, axis=1) == 0
    )

    print(f"  So bai co vector rong: {zero_rows}")


# ==========================================
# 9. CHUONG TRINH CHINH
# ==========================================

def main():
    print("========== DOC DU LIEU ==========")

    train_data = load_data(TRAIN_PATH)
    validation_data = load_data(VALIDATION_PATH)
    test_data = load_data(TEST_PATH)

    print(f"Train: {len(train_data)} bai")
    print(f"Validation: {len(validation_data)} bai")
    print(f"Test: {len(test_data)} bai")

    # Kiem tra nhan
    for dataset_name, dataset in [
        ("Train", train_data),
        ("Validation", validation_data),
        ("Test", test_data)
    ]:
        for article in dataset:
            label = int(article["label"])

            if label < 0 or label > 9:
                raise ValueError(
                    f"Nhan khong hop le trong {dataset_name}: {label}"
                )

    # Xay dung Vocabulary chi tu Train
    print("\n========== XAY DUNG VOCABULARY ==========")

    vocabulary, document_frequency = build_vocabulary(
        train_data
    )

    if len(vocabulary) == 0:
        raise ValueError(
            "Vocabulary rong. Khong the vectorize du lieu."
        )

    # Tinh IDF chi tu Train
    print("\n========== TINH IDF ==========")

    idf = calculate_idf(
        vocabulary,
        document_frequency,
        len(train_data)
    )

    print(f"So gia tri IDF: {len(idf)}")

    # Vectorize ba tap
    print("\n========== VECTORIZE DU LIEU ==========")

    X_train, y_train = transform_data(
        train_data,
        vocabulary,
        idf
    )

    X_validation, y_validation = transform_data(
        validation_data,
        vocabulary,
        idf
    )

    X_test, y_test = transform_data(
        test_data,
        vocabulary,
        idf
    )

    # Hien thi ket qua
    print("\n========== KET QUA VECTORIZE ==========")

    print_result("Train", X_train, y_train)
    print_result("Validation", X_validation, y_validation)
    print_result("Test", X_test, y_test)

    # Tao thu muc dau ra
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Luu cac ma tran va nhan
    output_path = OUTPUT_DIR / "tfidf_data.npz"

    np.savez_compressed(
        output_path,
        X_train=X_train,
        y_train=y_train,
        X_validation=X_validation,
        y_validation=y_validation,
        X_test=X_test,
        y_test=y_test
    )

    # Luu Vocabulary va cau hinh
    save_metadata(vocabulary)

    print("\n========== HOAN TAT ==========")
    print(f"Da luu ma tran TF-IDF: {output_path}")
    print("Vectorization thanh cong!")


if __name__ == "__main__":
    main()
