import streamlit as st
import joblib
import numpy as np
import scipy.sparse as sp
import re
import os

import streamlit as st

# --- CAMERA GIÁM SÁT (DEBUG) ---
st.title("🕵️ CHẾ ĐỘ THÁM TỬ")

# 1. Xem Streamlit đang đứng ở đâu?
current_path = os.getcwd()
st.info(f"📍 Streamlit đang đứng tại: `{current_path}`")

# 2. Xem xung quanh có những file gì?
files_here = os.listdir(current_path)
st.write("📂 Danh sách file nó nhìn thấy:", files_here)

# 3. Kiểm tra cụ thể xem có file model không?
target_file = 'sentiment_model.pkl' # Tên file bạn cần tìm
if target_file in files_here:
    st.success(f"✅ Đã tìm thấy '{target_file}'! (Nó ở ngay đây)")
else:
    st.error(f"❌ KHÔNG THẤY '{target_file}' đâu cả!")
    # Thử tìm xem nó có nằm trong thư mục con nào không
    for root, dirs, files in os.walk(current_path):
        if target_file in files:
            found_path = os.path.join(root, target_file)
            st.warning(f"⚠️ Á đù! Tìm thấy nó trốn ở đây này: `{found_path}`")
            st.markdown(f"👉 **Cách sửa:** Bạn phải đổi code load thành: `joblib.load('{found_path}')`")

st.markdown("---")
# -------------------------------


# 1. CẤU HÌNH TRANG WEB
st.set_page_config(
    page_title="AI Phân Tích Cảm Xúc",
    page_icon="🛍️",
    layout="centered"
)

# 2. HÀM LOAD MODEL (SIÊU TỐC ĐỘ)
@st.cache_resource
def load_models():
    # Kiểm tra xem file có tồn tại không
    if not os.path.exists('sentiment_model.pkl'):
        return None, None, None
    
    # Load 3 file .pkl lên bộ nhớ
    model = joblib.load('sentiment_model.pkl')
    tfidf = joblib.load('tfidf_vectorizer.pkl')
    scaler = joblib.load('scaler.pkl')
    return model, tfidf, scaler

# Load ngay khi mở web
model, tfidf, scaler = load_models()

# 3. GIAO DIỆN NGƯỜI DÙNG
st.title("🛍️ DỰ ĐOÁN ĐÁNH GIÁ SẢN PHẨM")
st.write("Dự đoán khách hàng **Hài Lòng (Positive)** hay **Thất Vọng (Negative)**.")

# Kiểm tra lỗi thiếu file
if model is None:
    st.error("❌ LỖI: Không tìm thấy file bộ não (.pkl).")
    st.warning("Bạn cần upload 3 file: sentiment_model.pkl, tfidf_vectorizer.pkl, scaler.pkl lên cùng thư mục GitHub!")
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


