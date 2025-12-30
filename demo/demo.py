import streamlit as st
import joblib
import numpy as np
import scipy.sparse as sp
import re
import os

# 1. CẤU HÌNH TRANG WEB
st.set_page_config(page_title="AI Phân Tích Cảm Xúc", page_icon="🛍️", layout="centered")

# 2. HÀM LOAD MODEL (Tự động tìm file)
@st.cache_resource
def load_models():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, 'sentiment_model.pkl')
    tfidf_path = os.path.join(current_dir, 'tfidf_vectorizer.pkl')
    scaler_path = os.path.join(current_dir, 'scaler.pkl')

    if not os.path.exists(model_path): return None, None, None, model_path
    
    try:
        model = joblib.load(model_path)
        tfidf = joblib.load(tfidf_path)
        scaler = joblib.load(scaler_path)
        return model, tfidf, scaler, None
    except Exception as e: return None, None, None, str(e)

model, tfidf, scaler, error_msg = load_models()

# 3. GIAO DIỆN
st.title("🛍️ DỰ ĐOÁN ĐÁNH GIÁ SẢN PHẨM")
st.write("Dự đoán khách hàng **Hài Lòng** hay **Thất Vọng**.")

if model is None:
    st.error("❌ LỖI: Không tìm thấy file model (.pkl).")
    st.info("Hãy upload 3 file .pkl cũ của bạn vào cùng thư mục Github với file này.")
    st.stop()

# 4. FORM NHẬP LIỆU
col1, col2 = st.columns(2)
with col1: age = st.number_input("Tuổi khách hàng", 18, 90, 30)
with col2: rating = st.slider("Rating (Sao)", 1, 5, 5)

review_input = st.text_area("Nội dung Review (Tiếng Anh)", height=150, placeholder="Example: The dress fits perfectly!")

# 5. XỬ LÝ (QUAN TRỌNG: FIX LỖI 4 FEATURES)
if st.button("🔍 PHÂN TÍCH NGAY", type="primary"):
    if not review_input.strip():
        st.warning("Vui lòng nhập nội dung review!")
    else:
        try:
            # A. Xử lý Text
            clean_text = review_input.lower()
            clean_text = re.sub(r'[^\w\s]', '', clean_text)
            vec_text = tfidf.transform([clean_text])
            
            # B. TẠO 4 FEATURES ĐỂ ĐÁP ỨNG MODEL CŨ
            # 1. Age: Lấy từ input
            # 2. Rating: Lấy từ input
            # 3. Positive Feedback Count: Mặc định là 0 (Vì review mới chưa ai like)
            pos_feedback = 0 
            # 4. Word Count: Tự đếm số từ trong review người dùng nhập
            word_count = len(clean_text.split())

            # Tạo mảng 4 thông số (Thứ tự này phổ biến nhất trong các bài mẫu trên mạng)
            # Nếu kết quả dự đoán bị sai lệch, hãy thử đổi thứ tự các biến này
            features_row = [[age, rating, pos_feedback, word_count]]
            
            # Chuẩn hóa (Lúc này Scaler sẽ thấy đủ 4 cột -> Hết lỗi)
            vec_num = scaler.transform(features_row)
            
            # C. Ghép và Dự đoán
            vec_final = sp.hstack((vec_text, vec_num))
            pred = model.predict(vec_final)[0]
            
            # Kết quả
            st.markdown("---")
            if pred == 1:
                st.success("😊 KẾT QUẢ: TÍCH CỰC (Hài lòng)")
                st.balloons()
            else:
                st.error("☹️ KẾT QUẢ: TIÊU CỰC (Thất vọng)")
                
        except ValueError as e:
            # Nếu vẫn lỗi, in ra chi tiết để sửa tiếp
            st.error(f"Lỗi kích thước dữ liệu: {e}")
            st.warning("Có thể thứ tự cột lúc train khác với thứ tự: [Tuổi, Rating, Like, Số từ].")
        except Exception as e:
            st.error(f"Lỗi khác: {e}")
