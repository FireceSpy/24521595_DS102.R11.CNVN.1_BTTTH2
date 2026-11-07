
from pathlib import Path

# Thư mục gốc của dự án
BASE_DIR = Path(__file__).resolve().parent.parent

# Cấu hình 10 danh mục VnExpress
CATEGORIES = {
    0: {
        "name": "Thời sự",
        "url": "https://vnexpress.net/thoi-su"
    },
    1: {
        "name": "Thế giới",
        "url": "https://vnexpress.net/the-gioi"
    },
    2: {
        "name": "Kinh doanh",
        "url": "https://vnexpress.net/kinh-doanh"
    },
    3: {
        "name": "Khoa học công nghệ",
        "url": "https://vnexpress.net/khoa-hoc-cong-nghe",
    },
    4: {
        "name": "Bất động sản",
        "url": "https://vnexpress.net/bat-dong-san"
    },
    5: {
        "name": "Sức khỏe",
        "url": "https://vnexpress.net/suc-khoe"
    },
    6: {
        "name": "Thể thao",
        "url": "https://vnexpress.net/the-thao"
    },
    7: {
        "name": "Giải trí",
        "url": "https://vnexpress.net/giai-tri"
    },
    8: {
        "name": "Pháp luật",
        "url": "https://vnexpress.net/phap-luat"
    },
    9: {
        "name": "Giáo dục",
        "url": "https://vnexpress.net/giao-duc"
    }
}

# Số bài viết cần thu thập cho mỗi danh mục
MAX_ARTICLES_PER_CATEGORY = 70

# Thời gian nghỉ giữa hai yêu cầu HTTP
REQUEST_DELAY = 2

# Thời gian chờ tối đa cho mỗi yêu cầu
REQUEST_TIMEOUT = 20

# Đường dẫn lưu dữ liệu
OUTPUT_FILE = BASE_DIR / "Data" / "Raw" / "DataReceived.json"

# Độ dài nội dung tối thiểu để chấp nhận bài viết
MIN_CONTENT_LENGTH = 200

# User-Agent nhận diện crawler phục vụ nghiên cứu
HEADERS = {
    "User-Agent": "StudentResearchBot/1.0"
}
