
import json
import os

DATA_PATH = "Data/Raw/DataReceived.json"

if not os.path.exists(DATA_PATH):
    print("Không tìm thấy file dữ liệu!")
    exit()

with open(DATA_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

print("Kiểu dữ liệu:", type(data).__name__)

if isinstance(data, list):
    print("Tổng số bản ghi:", len(data))

    if len(data) > 0:
        print("\nCác trường của bản ghi đầu tiên:")
        print(list(data[0].keys()))

        print("\nNội dung bản ghi đầu tiên:")
        print(json.dumps(data[0], ensure_ascii=False, indent=4))

elif isinstance(data, dict):
    print("\nCác khóa cấp cao nhất:")
    print(list(data.keys()))

    print("\nCấu trúc dữ liệu:")
    print(json.dumps(data, ensure_ascii=False, indent=4)[:5000])
