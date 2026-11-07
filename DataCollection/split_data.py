
import json
import os
import numpy as np

# ==========================================
# 1. CẤU HÌNH
# ==========================================

INPUT_PATH = "Data/Processed/ProcessedData.json"
OUTPUT_DIR = "Data/Splits"

RANDOM_SEED = 42

NUM_CLASSES = 10


# ==========================================
# 2. HÀM LƯU DỮ LIỆU
# ==========================================

def save_data(data, file_path):
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


# ==========================================
# 3. HÀM CHIA DỮ LIỆU
# ==========================================

def split_dataset(data):
    rng = np.random.default_rng(RANDOM_SEED)

    # Gom dữ liệu theo từng nhãn
    label_groups = {
        label: [] for label in range(NUM_CLASSES)
    }

    for article in data:
        label = article.get("label")

        if not isinstance(label, int) or label not in label_groups:
            raise ValueError(f"Nhãn không hợp lệ: {label}")

        label_groups[label].append(article)

    # Chọn ngẫu nhiên 5 nhãn nhận thêm 1 bài Validation
    validation_extra_labels = set(
        rng.choice(NUM_CLASSES, size=5, replace=False).tolist()
    )

    train_data = []
    validation_data = []
    test_data = []

    for label in range(NUM_CLASSES):
        articles = label_groups[label]

        if len(articles) != 70:
            raise ValueError(
                f"Label {label} có {len(articles)} bài, "
                "không đúng 70 bài."
            )

        # Xáo trộn dữ liệu của từng nhãn
        indices = rng.permutation(len(articles))
        shuffled_articles = [articles[i] for i in indices]

        # 70%: 49 bài
        train_data.extend(shuffled_articles[:49])

        # 30% còn lại: 21 bài
        remaining = shuffled_articles[49:]

        # Phân bổ 10 hoặc 11 bài cho Validation
        if label in validation_extra_labels:
            validation_count = 11
        else:
            validation_count = 10

        validation_data.extend(
            remaining[:validation_count]
        )

        test_data.extend(
            remaining[validation_count:]
        )

    # Xáo trộn thứ tự tổng thể trong mỗi tập
    rng.shuffle(train_data)
    rng.shuffle(validation_data)
    rng.shuffle(test_data)

    return train_data, validation_data, test_data


# ==========================================
# 4. KIỂM TRA KẾT QUẢ CHIA TẬP
# ==========================================

def validate_splits(train_data, validation_data, test_data):

    datasets = {
        "Train": train_data,
        "Validation": validation_data,
        "Test": test_data
    }

    expected_sizes = {
        "Train": 490,
        "Validation": 105,
        "Test": 105
    }

    print("\n========== KIỂM TRA CÁC TẬP DỮ LIỆU ==========")

    for name, dataset in datasets.items():

        print(f"\n{name}: {len(dataset)} bài")

        if len(dataset) != expected_sizes[name]:
            raise ValueError(
                f"Số lượng bài trong tập {name} không chính xác!"
            )

        label_count = {
            label: 0 for label in range(NUM_CLASSES)
        }

        for article in dataset:
            label_count[article["label"]] += 1

        for label, count in label_count.items():
            print(f"  Label {label}: {count} bài")

    # Kiểm tra không có ID trùng giữa các tập
    train_ids = {article["id"] for article in train_data}
    validation_ids = {
        article["id"] for article in validation_data
    }
    test_ids = {article["id"] for article in test_data}

    if train_ids & validation_ids:
        raise ValueError("ID bị trùng giữa Train và Validation!")

    if train_ids & test_ids:
        raise ValueError("ID bị trùng giữa Train và Test!")

    if validation_ids & test_ids:
        raise ValueError("ID bị trùng giữa Validation và Test!")

    # Kiểm tra không có URL trùng giữa các tập
    train_urls = {article["url"] for article in train_data}
    validation_urls = {
        article["url"] for article in validation_data
    }
    test_urls = {article["url"] for article in test_data}

    if train_urls & validation_urls:
        raise ValueError("URL bị trùng giữa Train và Validation!")

    if train_urls & test_urls:
        raise ValueError("URL bị trùng giữa Train và Test!")

    if validation_urls & test_urls:
        raise ValueError("URL bị trùng giữa Validation và Test!")

    total = len(train_data) + len(validation_data) + len(test_data)

    if total != 700:
        raise ValueError("Tổng số bài sau khi chia không bằng 700!")

    print("\nKhông có ID hoặc URL trùng giữa các tập.")
    print("Tổng số bài:", total)
    print("Kiểm tra chia tập thành công!")


# ==========================================
# 5. CHƯƠNG TRÌNH CHÍNH
# ==========================================

def main():

    if not os.path.exists(INPUT_PATH):
        print("Không tìm thấy file dữ liệu đã tiền xử lý!")
        return

    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    if len(data) != 700:
        print("Dữ liệu đầu vào không có đủ 700 bài!")
        return

    train_data, validation_data, test_data = split_dataset(data)

    # Kiểm tra trước khi lưu
    validate_splits(
        train_data,
        validation_data,
        test_data
    )

    # Tạo thư mục lưu kết quả
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Lưu từng tập
    save_data(
        train_data,
        os.path.join(OUTPUT_DIR, "train.json")
    )

    save_data(
        validation_data,
        os.path.join(OUTPUT_DIR, "validation.json")
    )

    save_data(
        test_data,
        os.path.join(OUTPUT_DIR, "test.json")
    )

    print("\n========== HOÀN TẤT ==========")
    print("Train:", os.path.join(OUTPUT_DIR, "train.json"))
    print("Validation:", os.path.join(OUTPUT_DIR, "validation.json"))
    print("Test:", os.path.join(OUTPUT_DIR, "test.json"))


if __name__ == "__main__":
    main()
