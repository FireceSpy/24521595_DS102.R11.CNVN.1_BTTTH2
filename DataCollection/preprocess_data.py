
import json
import os
import re
import html
import unicodedata

# ==========================================
# 1. CẤU HÌNH ĐƯỜNG DẪN
# ==========================================

INPUT_PATH = "Data/Raw/DataReceived.json"
OUTPUT_PATH = "Data/Processed/ProcessedData.json"


# ==========================================
# 2. HÀM LÀM SẠCH VĂN BẢN
# ==========================================

def clean_text(text):
    if not isinstance(text, str):
        return ""

    # Chuẩn hóa Unicode tiếng Việt
    text = unicodedata.normalize("NFC", text)

    # Giải mã HTML entities
    text = html.unescape(text)

    # Loại bỏ các thẻ HTML còn sót lại
    text = re.sub(r"<[^>]+>", " ", text)

    # Loại bỏ ký tự zero-width và BOM
    text = re.sub(r"[\u200b\u200c\u200d\ufeff]", "", text)

    # Chuẩn hóa xuống dòng
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Loại bỏ ký tự điều khiển không cần thiết
    text = "".join(
        char for char in text
        if char in "\n\t" or unicodedata.category(char)[0] != "C"
    )

    # Chuẩn hóa khoảng trắng trên từng dòng
    lines = []

    for line in text.split("\n"):
        line = re.sub(r"[ \t]+", " ", line)
        line = line.strip()

        if line:
            lines.append(line)

    # Giữ khoảng cách giữa các đoạn văn
    text = "\n".join(lines)

    return text.strip()


# ==========================================
# 3. HÀM TIỀN XỬ LÝ MỘT BÀI BÁO
# ==========================================

def preprocess_article(article):
    title = clean_text(article.get("title", ""))
    description = clean_text(article.get("description", ""))
    content = clean_text(article.get("content", ""))

    # Kết hợp tiêu đề, mô tả và nội dung
    text = "\n".join(
        part for part in [title, description, content]
        if part
    )

    # Giữ lại toàn bộ thông tin gốc
    processed_article = article.copy()

    # Cập nhật các trường văn bản đã làm sạch
    processed_article["title"] = title
    processed_article["description"] = description
    processed_article["content"] = content

    # Trường văn bản dùng cho Machine Learning
    processed_article["text"] = text

    return processed_article


# ==========================================
# 4. CHƯƠNG TRÌNH CHÍNH
# ==========================================

def main():
    if not os.path.exists(INPUT_PATH):
        print("Không tìm thấy file dữ liệu gốc!")
        return

    # Đọc dữ liệu gốc
    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        print("Dữ liệu phải có kiểu danh sách!")
        return

    print("Tổng số bài ban đầu:", len(data))

    processed_data = []
    label_count = {i: 0 for i in range(10)}

    # Tiền xử lý từng bài báo
    for article in data:
        processed_article = preprocess_article(article)

        label = processed_article.get("label")

        # Kiểm tra nhãn
        if not isinstance(label, int) or label not in label_count:
            print("Nhãn không hợp lệ:", article.get("id"))
            return

        # Kiểm tra văn bản sau tiền xử lý
        if not processed_article["text"]:
            print("Bài báo không có nội dung:", article.get("id"))
            return

        processed_data.append(processed_article)
        label_count[label] += 1

    # Tạo thư mục đầu ra nếu chưa tồn tại
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    # Lưu dữ liệu đã tiền xử lý
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(
            processed_data,
            f,
            ensure_ascii=False,
            indent=4
        )

    # ======================================
    # 5. THỐNG KÊ KẾT QUẢ
    # ======================================

    print("\n========== KẾT QUẢ TIỀN XỬ LÝ ==========")

    print("Tổng số bài đã xử lý:", len(processed_data))

    print("\nSố bài theo từng nhãn:")

    for label, count in label_count.items():
        print(f"Label {label}: {count} bài")

    print("\nTổng số nhãn:", len(label_count))

    print("\nĐường dẫn lưu dữ liệu:", OUTPUT_PATH)

    print("\nTiền xử lý hoàn tất!")


if __name__ == "__main__":
    main()
