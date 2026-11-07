
import re
import time
import requests

from bs4 import BeautifulSoup
from urllib.parse import urlparse

from config import (
    HEADERS,
    REQUEST_TIMEOUT,
    REQUEST_DELAY,
    MIN_CONTENT_LENGTH
)


def clean_text(text):
    """
    Làm sạch văn bản:
    - Loại bỏ khoảng trắng thừa.
    - Chuẩn hóa xuống dòng.
    """

    if not text:
        return ""

    text = re.sub(r"\s+", " ", text)

    return text.strip()


def get_article_date(soup):
    """
    Lấy ngày đăng bài viết từ metadata.
    """

    meta = soup.select_one(
        'meta[property="article:published_time"]'
    )

    if meta and meta.get("content"):
        return meta["content"].strip()

    time_element = soup.select_one("time[datetime]")

    if time_element:
        return time_element["datetime"].strip()

    return ""


def extract_article(session, url, category, label):
    """
    Truy cập một bài báo và trích xuất thông tin.
    """

    try:
        response = session.get(
            url,
            timeout=REQUEST_TIMEOUT
        )

        response.raise_for_status()

        soup = BeautifulSoup(
            response.text,
            "html.parser"
        )

        # Lấy tiêu đề
        title_element = soup.select_one(
            "h1.title-detail"
        )

        # Lấy mô tả
        description_element = soup.select_one(
            "p.description"
        )

        # Lấy phần nội dung
        content_element = soup.select_one(
            "article.fck_detail"
        )

        if not content_element:
            print("Không tìm thấy phần nội dung:", url)
            return None

        title = (
            clean_text(title_element.get_text(" ", strip=True))
            if title_element else ""
        )

        description = (
            clean_text(
                description_element.get_text(" ", strip=True)
            )
            if description_element else ""
        )

        # Lấy các đoạn văn trong nội dung bài viết
        paragraphs = content_element.find_all("p")

        content_parts = []
        seen_paragraphs = set()

        for paragraph in paragraphs:

            # Bỏ qua chú thích ảnh và nội dung không phải bài viết
            if paragraph.find_parent(
                ["figure", "figcaption"]
            ):
                continue

            if paragraph.find_parent(
                class_=re.compile(
                    r"caption|author|related|social|share",
                    re.I
                )
            ):
                continue

            text = clean_text(
                paragraph.get_text(" ", strip=True)
            )

            if not text:
                continue

            # Không lưu đoạn mô tả nếu bị lặp lại nguyên văn
            if description and text == description:
                continue

            # Loại bỏ đoạn văn trùng
            if text in seen_paragraphs:
                continue

            seen_paragraphs.add(text)
            content_parts.append(text)

        content = "\n".join(content_parts)

        # Kiểm tra dữ liệu
        if not title:
            print("Bài viết không có tiêu đề:", url)
            return None

        if len(content) < MIN_CONTENT_LENGTH:
            print("Nội dung quá ngắn:", url)
            return None

        article = {
            "title": title,
            "description": description,
            "content": content,
            "url": url,
            "published_date": get_article_date(soup),
            "category": category,
            "label": label
        }

        return article

    except requests.RequestException as error:
        print("Lỗi khi truy cập bài viết:", url)
        print("Chi tiết:", error)

        return None

    except Exception as error:
        print("Lỗi xử lý bài viết:", url)
        print("Chi tiết:", error)

        return None
