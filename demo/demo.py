import streamlit as st
import joblib
import numpy as np
import scipy.sparse as sp
import re
import os

# ==========================================
# 1. CẤU HÌNH & GIAO DIỆN
# ==========================================
st.set_page_config(
    page_title="Review Sentiment AI",
    page_icon="🛍️",
    layout="centered"
)

# CSS làm đẹp giao diện
st.markdown("""
    <style>
    .stButton>button {
        width: 100%;
        background-color: #ff4b4b;
        color: white;
        font-weight: bold;
        padding: 10px;
    }
    .result-box {
        padding: 20px;
        border-radius: 10px;
        margin-top: 20px;
        text-align: center;
        font-size: 20px;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. HÀM LOAD MODEL (TỰ ĐỘNG DÒ TÌM)
# ==========================================
@st.cache_resource
def load_ai_system():
    # Lấy đường dẫn thư mục hiện tại chứa file app.py
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Tạo đường dẫn tuyệt đối cho các file .pkl
    model_path = os.path.join(current_dir, 'sentiment_model.pkl')
    tfidf_path = os.path.join(current_dir, 'tfidf_vectorizer.pkl')
    scaler_path = os.path.join(current_dir, 'scaler.pkl')

    # Kiểm tra file tồn tại
    if not os.path.exists(model_path):
        return None, None, None, f"❌ Không tìm thấy file: {model_path}"
    
    try:
        model = joblib.load(model_path)
        tfidf = joblib.load(tfidf_path)
        scaler = joblib.load(scaler_path)
        return model, tfidf, scaler, None
    except Exception as e:
        return None, None, None, f"❌ Lỗi khi đọc file model: {str(e)}"

# Load model ngay khi khởi động
model, tfidf, scaler, error_msg = load_ai_system()

# ==========================================
# 3. GIAO DIỆN NHẬP LIỆU
# ==========================================
st.title("🛍️ DỰ ĐOÁN ĐÁNH GIÁ SẢN PHẨM")
st.markdown("---")

if model is None:
    st.error(error_msg)
    st.info("👉 Vui lòng upload 3 file .pkl (sentiment_model, tfidf_vectorizer, scaler) vào cùng thư mục GitHub với file app.py này.")
    st.stop()

# Chia cột nhập liệu
st.subheader("1. Thông tin khách hàng")
c1, c2, c3 = st.columns(3)
with c1:
    age = st.number_input("Tuổi (Age)", 18, 99, 30)
with c2:
    rating = st.slider("Đánh giá (Rating)", 1, 5, 5)
with c3:
    pos_feedback = st.number_input("Số Like (Feedback)", 0, 1000, 0, help="Số người thấy review này hữu ích")

st.subheader("2. Nội dung bình luận")
review_text = st.text_area("Nhập review (Tiếng Anh)", height=150, 
                          placeholder="Example: I absolutely love this dress! The material is soft and fits perfectly.")

# Hiển thị thông tin thời gian thực
if review_text:
    # Logic từ Notebook của bạn: Review_Len là độ dài ký tự (len của string)
    char_len = len(review_text)
    st.caption(f"ℹ️ Độ dài review: **{char_len} ký tự** (Model sẽ dùng số này để tính toán).")

# ==========================================
# 4. XỬ LÝ AI (LOGIC QUAN TRỌNG)
# ==========================================
if st.button("🔍 PHÂN TÍCH NGAY"):
    if not review_text.strip():
        st.warning("Vui lòng nhập nội dung review!")
    else:
        with st.spinner("AI đang phân tích..."):
            try:
                # --- BƯỚC 1: XỬ LÝ VĂN BẢN (TEXT) ---
                # Làm sạch cơ bản
                clean_text = review_text.lower()
                clean_text = re.sub(r'[^\w\s]', '', clean_text)
                
                # Vector hóa (TF-IDF)
                vec_text = tfidf.transform([clean_text]) # Shape: (1, 2000)

                # --- BƯỚC 2: XỬ LÝ SỐ (NUMERIC) ---
                # Chuẩn bị 4 tham số đúng như lúc Train (Dựa trên Snippet 2 trong Notebook)
                # Thứ tự: Age, Rating, Positive Feedback Count, Review_Len
                review_len = len(review_text) # Tính độ dài ký tự gốc
                
                features_row = np.array([[age, rating, pos_feedback, review_len]])
                
                # Scaler chuẩn hóa
                vec_num = scaler.transform(features_row) # Shape: (1, 4)

                # --- BƯỚC 3: GHÉP (STACKING) ---
                # Ghép Text và Số lại với nhau
                final_input = sp.hstack((vec_text, vec_num)) # Shape hiện tại

                # --- BƯỚC 4: VÁ LỖI THIẾU CỘT CATEGORY (MAGIC FIX) ---
                # Model của bạn mong đợi thêm các cột One-Hot (Division, Department...)
                # Nhưng ở đây ta không nhập, nên ta sẽ chèn số 0 vào cho đủ kích thước.
                
                expected_features = model.n_features_in_
                current_features = final_input.shape[1]
                
                if current_features < expected_features:
                    missing_cols = expected_features - current_features
                    # Tạo ma trận số 0 bù vào phần thiếu
                    zeros_padding = sp.csr_matrix(np.zeros((1, missing_cols)))
                    # Ghép vào cuối
                    final_input = sp.hstack((final_input, zeros_padding))
                    # st.warning(f"Đã tự động bù {missing_cols} cột dữ liệu thiếu để Model chạy được.")
                
                # --- BƯỚC 5: DỰ ĐOÁN ---
                pred = model.predict(final_input)[0]
                
                # Lấy độ tin cậy
                proba_score = 0
                if hasattr(model, "predict_proba"):
                    proba_score = np.max(model.predict_proba(final_input)) * 100

                # --- BƯỚC 6: HIỂN THỊ KẾT QUẢ ---
                if pred == 1:
                    st.markdown(f"""
                    <div class="result-box" style="background-color: #d4edda; color: #155724; border: 2px solid #c3e6cb;">
                        😊 KHÁCH HÀNG HÀI LÒNG<br>
                        <span style="font-size: 16px; font-weight: normal;">(Độ tin cậy: {proba_score:.1f}%)</span>
                    </div>
                    """, unsafe_allow_html=True)
                    st.balloons()
                else:
                    st.markdown(f"""
                    <div class="result-box" style="background-color: #f8d7da; color: #721c24; border: 2px solid #f5c6cb;">
                        ☹️ KHÁCH HÀNG THẤT VỌNG<br>
                        <span style="font-size: 16px; font-weight: normal;">(Độ tin cậy: {proba_score:.1f}%)</span>
                    </div>
                    """, unsafe_allow_html=True)

            except Exception as e:
                st.error("GẶP LỖI XỬ LÝ:")
                st.code(str(e))
                st.warning("Gợi ý: Kiểm tra lại xem thứ tự cột trong Scaler lúc train có đúng là [Age, Rating, Like, Len] không?")
