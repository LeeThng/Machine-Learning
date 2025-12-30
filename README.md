# Machine-Learning
# Dự án: Dự đoán đánh giá sản phẩm
## Giới thiệu đề tài
### Bài toán
Trong kỷ nguyên thương mại điện tử, đánh giá của khách hàng (Reviews) đóng vai trò then chốt trong việc xây dựng uy tín sản phẩm. Tuy nhiên, với số lượng hàng nghìn bình luận mỗi ngày, việc đọc và phân loại thủ công là bất khả thi. Doanh nghiệp cần một giải pháp tự động để hiểu được cảm xúc khách hàng (Sentiment) thông qua văn bản và các chỉ số đi kèm.
### Mục tiêu
Xây dựng mô hình Machine Learning dự đoán liệu một khách hàng có Khuyên dùng (Recommended) sản phẩm hay không, dựa trên:

- Dữ liệu văn bản: Nội dung bình luận (Review Text).

- Dữ liệu số: Số sao đánh giá (Rating) và Tuổi khách hàng (Age).
## Dataset (Bộ dữ liệu) 
Nguồn dữ liệu: Kaggle

Tên bộ dữ liệu: Women's E-Commerce Clothing Reviews

Link tải: Click vào đây để tải

Mô tả các cột chính:

- Review Text: Nội dung bình luận chi tiết của khách hàng.

- Rating: Điểm đánh giá sản phẩm (1-5 sao).

- Age: Tuổi của người đánh giá.

- Recommended IND: (Target Variable) Nhãn dự đoán. 1 là Khuyên dùng, 0 là Không khuyên dùng.

- Division/Department/Class Name: Thông tin phân loại sản phẩm.
## Pipeline (Quy trình xử lý)
Dự án tuân theo quy trình Machine Learning (End-to-End):

1. Exploratory Data Analysis (EDA): Phân tích thống kê mô tả, vẽ biểu đồ phân bố Rating, Age, WordCloud.

2. Data Preprocessing (Tiền xử lý):

    - Text Cleaning: Chuyển chữ thường, xóa Stopwords, xóa dấu câu, Lemmatization.

    - Feature Engineering: Sử dụng kỹ thuật Hybrid Learning:

- Vector hóa văn bản bằng TF-IDF (2000 features).

- Chuẩn hóa (Scaling) dữ liệu số (Age, Rating).

- Kết hợp (Feature Union) hai loại dữ liệu này thành vector đầu vào.

3. Handling Imbalance (Xử lý mất cân bằng): Áp dụng thuật toán SMOTE để cân bằng lại tỷ lệ giữa nhãn 0 và 1.

4. Model Training (Huấn luyện): Thử nghiệm trên 4 thuật toán khác nhau.

5. Evaluation (Đánh giá): Sử dụng Accuracy, Precision, Recall, F1-Score và Confusion Matrix.

## Mô hình sử dụng

- Logistic Regression: Đơn giản, tốc độ xử lý cực nhanh với dữ liệu thưa (Sparse data). Lý do chọn: Baseline chuẩn mực cho bài toán phân loại văn bản, kết quả thực nghiệm tốt nhất.
- Random Forest: Mô hình tổ hợp (Ensemble) phi tuyến. Lý do chọn: Kiểm thử khả năng bắt các mẫu dữ liệu phức tạp mà mô hình tuyến tính bỏ sót.
- SVM (Support Vector Machine): Tìm siêu phẳng tối ưu. Lý do chọn: Hiệu quả cao trong không gian nhiều chiều (do TF-IDF tạo ra).
- KNN (K-Nearest Neighbors): Dựa trên khoảng cách. Lý do chọn: Dùng để so sánh hiệu năng với các mô hình học máy phức tạp hơn.

## Kết quả

Logistic Regression (Best Model):
- Accuracy: ~94.28%
- F1-score: Cao và cân bằng giữa 2 lớp.
- Ưu điểm: Tốc độ train và predict nhanh nhất.

SVM & Random Forest
- Accuracy: ~93% - 94% (ngang ngửa Logistic Regression).
- Tuy nhiên thời gian huấn luyện lâu hơn đáng kể.

KNN:
- Accuracy thấp nhất trong 4 mô hình do hạn chế của việc tính khoảng cách trong không gian nhiều chiều.

### Kết luận: Chọn Logistic Regression để triển khai Demo.

## Hướng dẫn chạy

### Cài môi trường
1. Python 3.8 trở lên.
2. Cài dependencies: pip install -r requirements.txt

### Chạy train
Chạy trên Google Colab:
1. Truy cập Google Colab.
2. Chọn File > Upload notebook, chọn file
3. Chọn menu Runtime > Run all (Ctrl + F9). Lưu ý:
   - Cần upload file dataset Womens Clothing E-Commerce Reviews.csv lên Colab.
  
### Chạy demo
- Truy cập streamlit.io
- Chọn My app > Create app > Deploy a public app from GitHub > Paste GitHub URL
- Gán link của file demo vô GitHub URL
- Click Deloy
  
### Cấu trúc thư mục dự án
- app/: Source code chính.
- demo/: demo.py.
- data/: Data và hướng dẫn tải data.
- reports/: Báo cáo.
- slides/: Slide thuyết trình.
- requirements.txt
- README.md
- .gitignore


## Tác giả
- Họ tên: Lê Đức Thắng
- Mã SV: 12423032
- Lớp: 12431
- GV hướng dẫn: PGS. TS. Nguyễn Văn Hậu
