import streamlit as st
import joblib
import numpy as np
import scipy.sparse as sp
import re
import os

# 1. CẤU HÌNH TRANG WEB
st.set_page_config(
    page_title="AI Phân Tích Cảm Xúc",
    page_icon="🛍️",
    layout="centered"
)

# 2. HÀM LOAD MODEL (Tự động tìm file bất chấp vị trí)
@st.cache_resource
def load_models():
    # Lấy đường dẫn hiện tại của file code
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Tạo đường dẫn tuyệt đối
    model_path = os.path.join(current_dir, 'sentiment_model.pkl')
    tfidf_path = os.path.join(current_dir, 'tfidf_vectorizer.pkl')
    scaler_path = os.path.join(current_dir, 'scaler.pkl')

    if not os.path.exists(model_path):
        return None, None, None, model_path
    
    try:
        model = joblib.load(model_path)
        tfidf = joblib.load(tfidf_path)
        scaler = joblib.load(scaler_path)
        return model, tfidf, scaler, None
    except Exception as e:
        return None, None, None, str(e)

# Load model
model, tfidf, scaler, error_msg = load_models()

# 3. GIAO DIỆN NGƯỜI DÙNG
st.title("🛍️ DỰ ĐOÁN ĐÁNH GIÁ SẢN PHẨM")
st.write("Dự đoán khách hàng **Hài Lòng (Positive)** hay **Thất Vọng (Negative)**.")

# Kiểm tra lỗi load file
if model is None:
    st.error("❌ LỖI: Không tìm thấy file bộ não (.pkl).")
    st.info("Hãy upload 3 file .pkl vào cùng thư mục Github với file app.py.")
    st.stop()

# 4. FORM NHẬP LIỆU (Đã thêm phần nhập Like)
st.markdown("### 1. Thông tin khách hàng & Tương tác")
col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input("Tuổi khách", 18, 90, 30, help="Tuổi của người viết review")
with col2:
    rating = st.slider("Rating (Sao)", 1, 5, 5)
with col3:
    # --- MỚI THÊM: NHẬP SỐ LIKE ---
    pos_feedback = st.number_input("Số lượt Like", 0, 1000, 0, help="Số người thấy review này hữu ích")

st.markdown("### 2. Nội dung bình luận")
review_input = st.text_area("Review (Tiếng Anh)", height=150, 
                           placeholder="Example: The dress fits perfectly and looks amazing!")

# Hiển thị thông số ẩn để bạn kiểm soát
if review_input:
    word_count = len(re.sub(r'[^\w\s]', '', review_input).split())
    st.caption(f"ℹ️ Hệ thống đã tự động đếm được: **{word_count} từ** (Đây là tham số thứ 4 cho Model).")

# 5. XỬ LÝ DỰ ĐOÁN
if st.button("🔍 PHÂN TÍCH NGAY", type="primary"):
    if not review_input.strip():
        st.warning("Vui lòng nhập nội dung review!")
    else:
        try:
            # A. Xử lý Text (NLP)
            clean_text = review_input.lower()
            clean_text = re.sub(r'[^\w\s]', '', clean_text)
            vec_text = tfidf.transform([clean_text])
            
            # B. Xử lý Số (QUAN TRỌNG: ĐỦ 4 CỘT)
            # Tính lại số từ (để chắc chắn)
            final_word_count = len(clean_text.split())
            
            # Gom 4 biến vào mảng theo đúng thứ tự Data gốc
            # Thứ tự thường gặp: [Age, Rating, Like, WordCount]
            features_row = [[age, rating, pos_feedback, final_word_count]]
            
            # Chuẩn hóa số liệu
            vec_num = scaler.transform(features_row)
            
            # C. Ghép 2 loại dữ liệu
            vec_final = sp.hstack((vec_text, vec_num))
            
            # D. Dự đoán
            pred = model.predict(vec_final)[0]
            
            # Tính độ tin cậy
            confidence = ""
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(vec_final).max() * 100
                confidence = f"- Độ tin cậy: {proba:.1f}%"
            
            # E. Hiển thị kết quả
            st.markdown("---")
            if pred == 1:
                st.success(f"😊 KẾT QUẢ: TÍCH CỰC (Hài lòng) {confidence}")
                st.balloons()
            else:
                st.error(f"☹️ KẾT QUẢ: TIÊU CỰC (Thất vọng) {confidence}")
                
        except ValueError as e:
            st.error(f"⚠️ Lỗi dữ liệu đầu vào: {e}")
            st.warning("Gợi ý: Có thể thứ tự 4 cột số (Age, Rating, Like, WordCount) chưa khớp với lúc Train.")
        except Exception as e:
            st.error(f"Lỗi hệ thống: {e}")
