
import json
from collections import Counter
from pathlib import Path

# Đường dẫn file JSON hiện tại
BASE_DIR = Path(__file__).resolve().parent.parent

DATA_FILE = BASE_DIR / "Data" / "Raw" / "DataReceived.json"

REQUIRED_FIELDS = [
    "id",
    "title",
    "description",
    "content",
    "url",
    "published_date",
    "category",
    "label"
]


def main():

    if not DATA_FILE.exists():
        print("Không tìm thấy file dữ liệu:", DATA_FILE)
        return

    with open(DATA_FILE, "r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, list):
        print("Dữ liệu JSON không phải danh sách.")
        return

    print("=" * 60)
    print("KIỂM TRA CHẤT LƯỢNG DỮ LIỆU")
    print("=" * 60)

    print("Tổng số bài viết:", len(data))

    # Kiểm tra trường bị thiếu
    missing_fields = 0

    for article in data:
        for field in REQUIRED_FIELDS:
            if field not in article or article[field] is None:
                missing_fields += 1
                print(
                    "Thiếu trường:",
                    field,
                    "| ID:",
                    article.get("id")
                )

    # Kiểm tra URL trùng
    urls = [
        article.get("url")
        for article in data
    ]

    url_counts = Counter(urls)

    duplicate_urls = [
        url for url, count in url_counts.items()
        if url and count > 1
    ]

    # Kiểm tra nhãn
    invalid_labels = [
        article for article in data
        if not isinstance(article.get("label"), int)
        or article.get("label") not in range(10)
    ]

    # Thống kê độ dài nội dung
    content_lengths = [
        len(article.get("content", ""))
        for article in data
    ]

    short_articles = [
        article for article in data
        if len(article.get("content", "")) < 200
    ]

    print("\n--- KIỂM TRA LỖI ---")
    print("Số trường bị thiếu:", missing_fields)
    print("Số URL trùng:", len(duplicate_urls))
    print("Số nhãn không hợp lệ:", len(invalid_labels))
    print("Số bài có nội dung dưới 200 ký tự:", len(short_articles))

    print("\n--- ĐỘ DÀI NỘI DUNG ---")

    if content_lengths:
        print("Ngắn nhất:", min(content_lengths))
        print("Trung bình:", round(
            sum(content_lengths) / len(content_lengths), 2
        ))
        print("Dài nhất:", max(content_lengths))

    print("\n--- THỐNG KÊ THEO LABEL ---")

    label_counts = Counter(
        article.get("label")
        for article in data
    )

    for label in range(10):
        print(
            f"Label {label}:",
            label_counts.get(label, 0),
            "bài"
        )

    print("\n--- KẾT QUẢ ---")

    if (
        missing_fields == 0
        and len(duplicate_urls) == 0
        and len(invalid_labels) == 0
        and len(short_articles) == 0
    ):
        print("Các kiểm tra cơ bản đều đạt.")
    else:
        print("Cần xem xét và xử lý các vấn đề phía trên.")

    print("=" * 60)


if __name__ == "__main__":
    main()
