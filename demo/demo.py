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

# 2. HÀM LOAD MODEL (SỬA LẠI ĐỂ TỰ ĐỘNG TÌM FILE)
@st.cache_resource
def load_models():
    # A. Lấy đường dẫn thư mục hiện tại (nơi chứa file app.py này)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # B. Tạo đường dẫn tuyệt đối đến 3 file .pkl
    # (Dù bạn để ở đâu, máy cũng sẽ tự ghép đường dẫn đúng)
    model_path = os.path.join(current_dir, 'sentiment_model.pkl')
    tfidf_path = os.path.join(current_dir, 'tfidf_vectorizer.pkl')
    scaler_path = os.path.join(current_dir, 'scaler.pkl')

    # C. Kiểm tra xem file có tồn tại không
    if not os.path.exists(model_path):
        # Trả về đường dẫn để báo lỗi cho chính xác
        return None, None, None, model_path
    
    try:
        # D. Load file bằng đường dẫn tuyệt đối
        model = joblib.load(model_path)
        tfidf = joblib.load(tfidf_path)
        scaler = joblib.load(scaler_path)
        return model, tfidf, scaler, None
    except Exception as e:
        return None, None, None, str(e)

# Load ngay khi mở web
model, tfidf, scaler, error_msg = load_models()

# 3. GIAO DIỆN NGƯỜI DÙNG
st.title("🛍️ DỰ ĐOÁN ĐÁNH GIÁ SẢN PHẨM")
st.write("Dự đoán khách hàng **Hài Lòng (Positive)** hay **Thất Vọng (Negative)**.")

# Kiểm tra lỗi thiếu file hoặc load lỗi
if model is None:
    st.error("❌ LỖI: Không tìm thấy hoặc không đọc được file bộ não (.pkl).")
    st.warning(f"Máy đang cố tìm tại đường dẫn này: `{error_msg}`")
    st.info("👉 Hãy chắc chắn bạn đã upload 3 file .pkl vào CÙNG THƯ MỤC với file app.py trên GitHub.")
    st.stop()

# 4. FORM NHẬP LIỆU
col1, col2 = st.columns(2)
with col1:
    age = st.number_input("Tuổi khách hàng", 18, 90, 25)
with col2:
    rating = st.slider("Rating (Sao)", 1, 5, 5)

review_input = st.text_area("Nội dung Review (Tiếng Anh)", height=150, 
                           placeholder="Example: The dress fits perfectly and looks amazing!")

# 5. XỬ LÝ DỰ ĐOÁN
if st.button("🔍 PHÂN TÍCH NGAY", type="primary"):
    if not review_input.strip():
        st.warning("Vui lòng nhập nội dung review!")
    else:
        try:
            # A. Xử lý Text
            clean_text = review_input.lower()
            clean_text = re.sub(r'[^\w\s]', '', clean_text)
            vec_text = tfidf.transform([clean_text])
            
            # B. Xử lý Số (Age, Rating) - Thứ tự [Age, Rating]
            vec_num = scaler.transform([[age, rating]])
            
            # C. Ghép lại
            vec_final = sp.hstack((vec_text, vec_num))
            
            # D. Dự đoán
            pred = model.predict(vec_final)[0]
            
            # Tính độ tin cậy (nếu model hỗ trợ)
            confidence = "Cao"
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(vec_final).max() * 100
                confidence = f"{proba:.1f}%"
            
            # E. Hiển thị kết quả
            st.markdown("---")
            if pred == 1:
                st.success(f"😊 KẾT QUẢ: TÍCH CỰC (Hài lòng) - Độ tin cậy: {confidence}")
                st.balloons()
            else:
                st.error(f"☹️ KẾT QUẢ: TIÊU CỰC (Thất vọng) - Độ tin cậy: {confidence}")
                
        except Exception as e:
            st.error(f"Có lỗi xảy ra: {e}")
