import re
import json
import time

import requests

from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse

from config import (
    CATEGORIES,
    MAX_ARTICLES_PER_CATEGORY,
    REQUEST_DELAY,
    REQUEST_TIMEOUT,
    OUTPUT_FILE,
    HEADERS
)

from extract_article import extract_article



def load_existing_data():
    """
    Đọc dữ liệu đã thu thập trước đó.
    """

    if not OUTPUT_FILE.exists():
        print("Chưa có file JSON. Bắt đầu thu thập mới.")
        return []

    if OUTPUT_FILE.stat().st_size == 0:
        print("File JSON đang rỗng. Bắt đầu thu thập mới.")
        return []

    try:
        with open(
            OUTPUT_FILE,
            "r",
            encoding="utf-8"
        ) as file:

            data = json.load(file)

            if isinstance(data, list):
                print("Đã đọc dữ liệu cũ:", len(data), "bài")
                return data

            print("Lỗi: Dữ liệu JSON phải là một danh sách.")
            return None

    except (json.JSONDecodeError, OSError) as error:
        print("Không thể đọc dữ liệu cũ:", error)
        return None



def save_data(data):
    """
    Lưu toàn bộ dữ liệu vào JSON.
    """

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    temp_file = OUTPUT_FILE.with_suffix(".tmp")

    with open(
        temp_file,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            data,
            file,
            ensure_ascii=False,
            indent=4
        )

    # Thay file cũ bằng file mới sau khi ghi thành công
    temp_file.replace(OUTPUT_FILE)


def normalize_url(url):
    """
    Chuẩn hóa URL bài viết.
    """

    url = urljoin("https://vnexpress.net", url)

    parsed = urlparse(url)

    if parsed.scheme not in ("http", "https"):
        return None

    if parsed.netloc not in (
        "vnexpress.net",
        "www.vnexpress.net"
    ):
        return None

    # Chỉ nhận URL bài viết kết thúc bằng .html
    if not parsed.path.endswith(".html"):
        return None

    return parsed._replace(
        query="",
        fragment=""
    ).geturl()



def get_category_page_data(session, category_url):
    """
    Lấy URL bài viết và URL các trang danh mục tiếp theo.

    Chỉ sử dụng đường dẫn phân trang tìm thấy trong HTML,
    không tự tạo URL trang tiếp theo.
    """

    try:
        response = session.get(
            category_url,
            timeout=REQUEST_TIMEOUT
        )

        response.raise_for_status()

        soup = BeautifulSoup(
            response.text,
            "html.parser"
        )

        article_links = []
        seen_articles = set()

        # Lấy URL bài viết
        for link in soup.select("a[href]"):

            parent = link.find_parent(
                class_="title-news"
            )

            if not parent:
                continue

            title = link.get_text(" ", strip=True)

            if not title:
                continue

            url = normalize_url(link["href"])

            if not url or url in seen_articles:
                continue

            seen_articles.add(url)

            article_links.append({
                "title": title,
                "url": url
            })

        # Tìm các liên kết phân trang
        page_urls = []
        seen_pages = set()

        base_path = urlparse(category_url).path.rstrip("/")

        for link in soup.select("a[href]"):

            href = link.get("href", "").strip()

            if not href:
                continue

            full_url = urljoin(category_url, href)
            parsed = urlparse(full_url)

            if parsed.netloc not in (
                "vnexpress.net",
                "www.vnexpress.net"
            ):
                continue

            path = parsed.path.rstrip("/")

            # Nhận diện một số dạng URL phân trang:
            # /thoi-su-p2
            # /thoi-su/page/2
            # ?page=2
            is_category_page = (
                path.startswith(base_path + "-p")
                or path.startswith(base_path + "/page/")
                or (
                    path == base_path
                    and re.search(r"[?&]page=\d+", full_url)
                )
            )

            if not is_category_page:
                continue

            page_url = parsed._replace(
                fragment=""
            ).geturl()

            if page_url == category_url:
                continue

            if page_url in seen_pages:
                continue

            seen_pages.add(page_url)
            page_urls.append(page_url)

        return article_links, page_urls

    except requests.RequestException as error:
        print("Lỗi truy cập trang danh mục:", category_url)
        print("Chi tiết:", error)

        return [], []


def get_all_category_links(session, category_url, max_pages=20):
    """
    Duyệt các trang danh mục và gom URL bài viết.

    max_pages là giới hạn an toàn để tránh duyệt vô hạn.
    """

    page_queue = [category_url]
    visited_pages = set()

    all_links = []
    seen_articles = set()

    while page_queue and len(visited_pages) < max_pages:

        current_page = page_queue.pop(0)

        if current_page in visited_pages:
            continue

        visited_pages.add(current_page)

        print(
            f"\nĐang đọc trang danh mục "
            f"{len(visited_pages)}: {current_page}"
        )

        links, next_pages = get_category_page_data(
            session,
            current_page
        )

        for item in links:

            if item["url"] not in seen_articles:
                seen_articles.add(item["url"])
                all_links.append(item)

        # Thêm các trang tiếp theo chưa được duyệt
        for next_page in next_pages:

            if (
                next_page not in visited_pages
                and next_page not in page_queue
            ):
                page_queue.append(next_page)

        print("Số URL bài viết tìm được:", len(all_links))
        print("Số trang danh mục đang chờ:", len(page_queue))

        time.sleep(REQUEST_DELAY)

    print("\nTổng số trang danh mục đã duyệt:", len(visited_pages))
    print("Tổng số URL bài viết tìm được:", len(all_links))

    return all_links



def crawl_category(
    session,
    label,
    category_info,
    all_data,
    seen_urls
):
    """
    Thu thập đủ số bài mục tiêu cho một danh mục.
    """

    category_name = category_info["name"]
    category_url = category_info["url"]

    print("\n" + "=" * 60)
    print("DANH MỤC:", category_name)
    print("LABEL:", label)
    print("=" * 60)

    # Đếm số bài đã có của danh mục này
    existing_count = sum(
        1
        for article in all_data
        if article.get("label") == label
    )

    remaining = MAX_ARTICLES_PER_CATEGORY - existing_count

    print("Số bài đã có:", existing_count)
    print("Số bài cần thu thập thêm:", max(0, remaining))

    if remaining <= 0:
        print("Danh mục đã đạt mục tiêu.")
        return 0

    # Lấy liên kết từ nhiều trang danh mục
    links = get_all_category_links(
        session,
        category_url
    )

    collected = 0

    for item in links:

        if collected >= remaining:
            break

        url = item["url"]

        # Bỏ qua bài đã thu thập
        if url in seen_urls:
            continue

        print("\nĐang xử lý:", item["title"])

        article = extract_article(
            session,
            url,
            category_name,
            label
        )

        # Nghỉ sau mỗi lần truy cập bài viết
        time.sleep(REQUEST_DELAY)

        if article is None:
            continue

        article["id"] = (
            max(
                (
                    a.get("id", 0)
                    for a in all_data
                    if isinstance(a.get("id"), int)
                ),
                default=0
            ) + 1
        )

        all_data.append(article)
        seen_urls.add(url)

        # Lưu ngay sau khi thu thập thành công
        save_data(all_data)

        collected += 1

        print("Đã lưu thành công:", article["title"])
        print(
            "Tiến độ danh mục:",
            existing_count + collected,
            "/",
            MAX_ARTICLES_PER_CATEGORY
        )

    final_count = existing_count + collected

    print("\nHoàn thành:", category_name)
    print("Số bài mới:", collected)
    print("Tổng số bài của danh mục:", final_count)

    if final_count < MAX_ARTICLES_PER_CATEGORY:
        print(
            "CHƯA ĐỦ DỮ LIỆU:",
            MAX_ARTICLES_PER_CATEGORY - final_count,
            "bài còn thiếu."
        )

    return collected



def print_statistics(data):
    """
    Thống kê dữ liệu theo từng danh mục.
    """

    print("\n" + "=" * 60)
    print("THỐNG KÊ DỮ LIỆU")
    print("=" * 60)

    print("Tổng số bài viết:", len(data))

    for label, category_info in CATEGORIES.items():

        count = sum(
            1
            for article in data
            if article.get("label") == label
        )

        print(
            f"Label {label} - "
            f"{category_info['name']}: {count} bài"
        )

    print("=" * 60)


def main():
    print("ĐANG BẮT ĐẦU HÀM MAIN")
    print("Đường dẫn lưu JSON:", OUTPUT_FILE.resolve())

    print("BẮT ĐẦU THU THẬP DỮ LIỆU VNEXPRESS")

    # Đọc dữ liệu cũ
    all_data = load_existing_data()

    if all_data is None:
        print("Dừng chương trình để tránh ghi đè dữ liệu lỗi.")
        return

    # Tạo tập URL đã tồn tại
    seen_urls = {
        article.get("url")
        for article in all_data
        if article.get("url")
    }

    print("Số bài viết đã có:", len(all_data))

    session = requests.Session()
    print("\nĐÃ KHỞI TẠO SESSION")
    print("BẮT ĐẦU DUYỆT 10 DANH MỤC")
    session.headers.update(HEADERS)

    total_new = 0

    try:
        for label, category_info in CATEGORIES.items():

            collected = crawl_category(
                session,
                label,
                category_info,
                all_data,
                seen_urls
            )

            total_new += collected

            # Nghỉ giữa các danh mục
            time.sleep(REQUEST_DELAY)

    finally:
        session.close()

    # Lưu lại dữ liệu cuối cùng
    save_data(all_data)

    print("\nĐÃ HOÀN THÀNH THU THẬP!")
    print("Số bài viết mới:", total_new)
    print("Tổng số bài hiện có:", len(all_data))
    print("File lưu dữ liệu:", OUTPUT_FILE)

    print_statistics(all_data)


if __name__ == "__main__":
    main()
