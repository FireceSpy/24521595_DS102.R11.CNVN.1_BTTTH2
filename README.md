# DS102.R11.CNVN - Bài tập thực hành 2

## 1. Giới thiệu

Dự án được thực hiện trong khuôn khổ môn học **Học Máy Thống Kê**.

Mục tiêu của dự án là xây dựng và đánh giá các mô hình Machine Learning phục vụ bài toán phân loại văn bản tin tức. Dữ liệu được thu thập từ trang báo điện tử VnExpress để thay thế cho tập dữ liệu MNIST và được sử dụng xuyên suốt cả ba Assignment.

Dự án triển khai hai thuật toán phân loại chính:
- **Logistic Regression:** Phân loại văn bản thành 2 lớp.
- **Softmax Regression:** Phân loại văn bản thành nhiều lớp, ở đây chỉ xét 10 lớp.

Các thuật toán được triển khai theo hai hướng:
- Tự cài đặt bằng thư viện NumPy.
- Sử dụng thư viện Scikit-learn.

## 2. Công nghệ sử dụng
| Công nghệ | Mục đích |
|---|---|
| Python | Ngôn ngữ lập trình chính |
| NumPy | Xử lý dữ liệu số và tự cài đặt thuật toán |
| Scikit-learn | Huấn luyện và đánh giá mô hình Machine Learning |
| Matplotlib | Trực quan hóa kết quả và Confusion Matrix |
| Joblib | Lưu mô hình đã huấn luyện |
| JSON | Lưu trữ dữ liệu bài viết |
| TF-IDF | Chuyển đổi văn bản thành vector đặc trưng |

## 3. Thu thập và xây dựng bộ dữ liệu

### 3.1. Nguồn dữ liệu

Dữ liệu được thu thập từ trang web: https://vnexpress.net/

Bộ dữ liệu gồm 700 bài viết thuộc 10 chuyên mục, mỗi chuyên mục có 70 bài viết.

### 3.2. Các lớp dữ liệu

| Label | Chuyên mục |
|---:|---|
| 0 | Thời sự |
| 1 | Thế giới |
| 2 | Kinh doanh |
| 3 | Khoa học công nghệ |
| 4 | Bất động sản |
| 5 | Sức khỏe |
| 6 | Thể thao |
| 7 | Giải trí |
| 8 | Pháp luật |
| 9 | Giáo dục |

Mỗi bài viết được lưu trữ với các thông tin như tiêu đề, mô tả, nội dung, đường dẫn, ngày đăng và nhãn phân loại.

### 3.3. Tiền xử lý dữ liệu

Dữ liệu sau khi thu thập được xử lý trước khi đưa vào các mô hình Machine Learning.

Các bước chính:
1. Đọc dữ liệu bài viết từ file JSON.
2. Kết hợp tiêu đề, mô tả và nội dung bài viết thành một trường văn bản.
3. Làm sạch văn bản và chuẩn hóa dữ liệu.
4. Giữ nguyên dấu tiếng Việt để bảo toàn thông tin ngôn ngữ.
5. Chia dữ liệu thành các tập Train, Validation và Test.
6. Sử dụng TF-IDF để chuyển đổi văn bản thành vector đặc trưng.

### 3.4. Chia tập dữ liệu

Bộ dữ liệu gồm 700 bài viết được chia như sau:
| Tập dữ liệu | Số lượng | Tỷ lệ |
|---|---:|---:|
| Train | 490 | 70% |
| Validation | 105 | 15% |
| Test | 105 | 15% |
| **Tổng cộng** | **700** | **100%** |

Việc chia dữ liệu được thực hiện một lần và các Assignment sử dụng chung các tập dữ liệu này.

Vocabulary và IDF được xây dựng từ tập Train để tránh rò rỉ thông tin từ Validation và Test.

### 3.5. Vector hóa văn bản

Sử dụng phương pháp TF-IDF để biểu diễn văn bản dưới dạng vector số.

- Kích thước Vocabulary: 5.000 đặc trưng.
- Train: 490 × 5.000.
- Validation: 105 × 5.000.
- Test: 105 × 5.000.

Dữ liệu vector hóa được lưu trong file `tfidf_data.npz`.

## 4. Cấu trúc thư mục dự án

```text
24521595_DS102.R11.CNVN.1_BTTH2/
│
├── DataCollection/                  # Thu thập và kiểm tra dữ liệu từ VnExpress
│   ├── config.py                    # Cấu hình các thông số và danh mục thu thập dữ liệu
│   ├── extract_article.py           # Trích xuất nội dung bài viết từ trang báo
│   ├── crawler.py                   # Thu thập các bài viết thuộc 10 danh mục
│   ├── validate_data.py             # Kiểm tra tính hợp lệ và chất lượng dữ liệu
│   ├── inspect_data.py              # Thống kê, kiểm tra và quan sát dữ liệu đã thu thập
│   ├── preprocess_data.py           # Làm sạch và tiền xử lý dữ liệu văn bản
│   └── split_data.py                # Chia dữ liệu thành Train, Validation và Test
│
├── Data/                            # Lưu trữ dữ liệu phục vụ cả 3 Assignment
│   ├── Raw/                         # Dữ liệu thô ban đầu
│   │   └── DataReceived.json        # Dữ liệu bài viết được thu thập từ VnExpress
│   │
│   ├── Processed/                   # Dữ liệu sau khi tiền xử lý
│   │   └── ProcessedData.json       # Dữ liệu đã làm sạch và chuẩn hóa văn bản
│   │
│   ├── Splits/                      # Dữ liệu được chia thành các tập
│   │   ├── train.json               # Tập huấn luyện mô hình (Train)
│   │   ├── validation.json          # Tập kiểm định mô hình (Validation)
│   │   └── test.json                # Tập kiểm tra mô hình (Test)
│   │
│   └── Vectorized/                  # Dữ liệu văn bản sau khi chuyển thành vector
│       ├── vocabulary.json          # Danh sách từ vựng được xây dựng từ tập Train
│       ├── vectorizer_config.json   # Cấu hình của quá trình vector hóa TF-IDF
│       └── tfidf_data.npz           # Các ma trận TF-IDF của Train, Validation và Test
│
├── Assignment1/                     # Bài tập 1: Xây dựng Binary Logistic Regression bằng NumPy
│   ├── logistic_regression.py       # Cài đặt Logistic Regression bằng NumPy
│   └── Results/                     # Lưu mô hình và biểu đồ kết quả Assignment 1
│
├── Assignment2/                     # Bài tập 2: Xây dựng Softmax Regression bằng NumPy
│   ├── softmax_regression.py        # Cài đặt Softmax Regression bằng NumPy
│   └── Results/                     # Lưu mô hình và biểu đồ kết quả Assignment 2
│
├── Assignment3/                     # Bài tập 3: Logistic Regression và Multinomial Regression sử dụng thư viện Scikit Learn
│   ├── logistic_regression.py       # Huấn luyện Logistic Regression nhị phân bằng Scikit-learn
│   ├── softmax_regression.py        # Huấn luyện Multinomial Logistic Regression bằng Scikit-learn
│   └── Results/                     # Lưu mô hình, báo cáo đánh giá và biểu đồ kết quả
│
├── Common/                          # Các module dùng chung giữa các Assignment
│   └── text_vectorizer.py           # Xây dựng từ điển và chuyển văn bản thành vector TF-IDF
│
└── README.md                        # Giới thiệu dự án, cấu trúc thư mục và hướng dẫn sử dụng
```

## 5. Nội dung thực hiện các Assignment

### Assignment 1 - Logistic Regression sử dụng NumPy

Mục tiêu: Tự cài đặt thuật toán Logistic Regression để giải quyết bài toán phân loại văn bản thành 2 lớp.

Hai lớp được lựa chọn cho bài tập này:

- Thể thao (Label 6).
- Giải trí (Label 7).

Các nội dung triển khai:

- Sử dụng NumPy để xử lý dữ liệu và thực hiện các phép toán.
- Xây dựng hàm Sigmoid.
- Xây dựng hàm Loss.
- Tính Gradient và cập nhật trọng số bằng Gradient Descent.
- Huấn luyện mô hình trên tập Train.
- Đánh giá mô hình bằng Validation và Test.
- Tính Accuracy, Precision, Recall và F1-score.
- Trực quan hóa quá trình huấn luyện và lưu mô hình.

### Assignment 2 - Softmax Regression sử dụng NumPy

Mục tiêu: Tự cài đặt thuật toán Softmax Regression để phân loại văn bản thành 10 lớp.

Các nội dung triển khai:

- Sử dụng NumPy để xây dựng mô hình Softmax Regression.
- Xây dựng hàm Softmax.
- Sử dụng Categorical Cross-Entropy làm hàm Loss.
- Tính Gradient và cập nhật trọng số.
- Áp dụng L2 Regularization.
- Sử dụng Early Stopping để hỗ trợ kiểm soát quá trình huấn luyện.
- Đánh giá mô hình trên Validation và Test.
- Tính các chỉ số Accuracy, Precision, Recall và F1-score.
- Lưu kết quả và mô hình đã huấn luyện.

### Assignment 3 - Sử dụng thư viện Scikit-learn

Mục tiêu: Sử dụng thư viện Scikit-learn để xây dựng và đánh giá các mô hình phân loại văn bản.

Assignment này gồm hai mô hình:

**1. Logistic Regression**

- Sử dụng  lớp Thể thao và Giải trí.
- Sử dụng `sklearn.linear_model.LogisticRegression`.
- Huấn luyện mô hình bằng tập Train.
- Đánh giá trên Validation và Test.

**2. Softmax Regression**

- Sử dụng toàn bộ 10 lớp dữ liệu.
- Sử dụng `sklearn.linear_model.LogisticRegression` với cấu hình `lbfgs` cho bài toán phân loại đa lớp.
- Huấn luyện và đánh giá trên các tập dữ liệu đã chuẩn bị.

Các kết quả đánh giá được lưu dưới dạng báo cáo TXT và hình ảnh Confusion Matrix. Mô hình sau huấn luyện được lưu bằng Joblib.

## 6. Cài đặt môi trường

Mở Terminal tại thư mục gốc dự án.

### 6.1. Cài đặt thư viện cho Assignment 1 và Assignment 2

Hai Assignment đầu tiên sử dụng NumPy để tự cài đặt các thuật toán Logistic Regression và Softmax Regression.

Cài đặt các thư viện cần thiết:

```bash
pip install numpy matplotlib
```

### 6.2. Cài đặt thư viện cho Assignment 3

Assignment 3 sử dụng Scikit-learn để xây dựng và huấn luyện các mô hình Machine Learning.

Cài đặt các thư viện cần thiết:

```bash
pip install scikit-learn joblib
```

### 6.3. Các thư viện hỗ trợ khác

Một số thư viện được sử dụng trong quá trình thu thập, xử lý và lưu trữ dữ liệu:

```bash
pip install requests beautifulsoup4 scipy
```

Lưu ý: Nếu các thư viện đã được cài đặt trước đó thì không cần cài đặt lại.

## 7. Cách chạy chương trình

### 7.1. Chuẩn bị dữ liệu

Ở dự án này đã có sẵn dữ liệu và các file vector hóa trong thư mục `Data`, có thể chạy trực tiếp các Assignment.

Nếu cần thực hiện lại quá trình thu thập và chuẩn bị dữ liệu từ trang web của VNExpress, chạy lần lượt:

```bash
python DataCollection/crawler.py
python DataCollection/validate_data.py
python DataCollection/preprocess_data.py
python DataCollection/split_data.py
python common/text_vectorizer.py
```

Lưu ý: Chỉ chạy lại quá trình thu thập dữ liệu khi thực sự cần thiết. Việc tạo lại dữ liệu hoặc chia tập dữ liệu có thể làm thay đổi kết quả thực nghiệm.

### 7.2. Chạy Assignment 1

```bash
python Assignment1/logistic_regression.py
```

### 7.3. Chạy Assignment 2

```bash
python Assignment2/softmax_regression.py
```

### 7.4. Chạy Assignment 3

Logistic Regression:

```bash
python Assignment3/logistic_regression.py
```

Softmax Regression:

```bash
python Assignment3/softmax_regression.py
```
## 8. Kết quả thực nghiệm

Các kết quả của từng Assignment được lưu trong thư mục `Results` tương ứng.

Các loại kết quả bao gồm:

- Báo cáo đánh giá mô hình dạng TXT.
- Confusion Matrix dạng PNG.
- Loss Curve dạng PNG đối với các mô hình tự cài đặt có lưu biểu đồ Loss.
- Mô hình đã huấn luyện để có thể sử dụng lại mà không cần huấn luyện từ đầu.

Các mô hình được đánh giá trên tập Test gồm 105 bài viết.

| Assignment | Mô hình | Test Accuracy |
|---|---|---:|
| Assignment 1 | Logistic Regression - NumPy | 95,24% |
| Assignment 2 | Softmax Regression - NumPy | 82,86% |
| Assignment 3 | Logistic Regression - Scikit-learn | 95,24% |
| Assignment 3 | Softmax Regression - Scikit-learn | 85,71% |

Kết quả cho thấy các mô hình có khả năng phân loại tương đối tốt trên bộ dữ liệu VnExpress.

Trong hai mô hình phân loại 10 lớp, mô hình sử dụng Scikit-learn đạt Test Accuracy cao hơn mô hình tự cài đặt bằng NumPy trong lần thực nghiệm này.

## 9. Kết luận

Dự án đã xây dựng được quy trình phân loại văn bản tin tức từ dữ liệu thu thập thực tế trên VnExpress.

Thông qua ba Assignment, dự án đã thực hiện việc tự cài đặt các thuật toán Logistic Regression và Softmax Regression bằng NumPy, đồng thời sử dụng Scikit-learn để xây dựng các mô hình tương ứng.

Các mô hình được đánh giá trên cùng bộ dữ liệu với các chỉ số Accuracy, Precision, Recall và F1-score, qua đó hỗ trợ so sánh hiệu quả giữa phương pháp tự cài đặt thuật toán và phương pháp sử dụng thư viện Machine Learning.

Dự án giúp củng cố kiến thức về xử lý dữ liệu văn bản, biểu diễn văn bản bằng TF-IDF, xây dựng mô hình phân loại và đánh giá hiệu quả các thuật toán Machine Learning.