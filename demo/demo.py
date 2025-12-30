import streamlit as st
import pandas as pd
import numpy as np
import scipy.sparse as sp
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# ================== 1. CẤU HÌNH TRANG WEB ==================
st.set_page_config(
    page_title="AI Phân Tích Đánh Giá Sản Phẩm",
    page_icon="🛍️",
    layout="centered"
)

# ================== 2. TẢI DỮ LIỆU & HUẤN LUYỆN (TỰ ĐỘNG) ==================
@st.cache_resource
def load_and_train_model():
    # --- A. Đọc dữ liệu ---
    # Cố gắng đọc file CSV từ cùng thư mục trên GitHub
    try:
        # Giả sử file csv nằm cùng cấp với file code này trên GitHub
        df = pd.read_csv("Womens Clothing E-Commerce Reviews.csv")
    except:
        st.error("❌ Không tìm thấy file dữ liệu 'Womens Clothing E-Commerce Reviews.csv'. Vui lòng upload nó lên GitHub cùng file code!")
        return None, None, None

    # --- B. Tiền xử lý nhanh ---
    # 1. Xóa dòng thiếu quan trọng
    df = df.dropna(subset=['Review Text', 'Age', 'Rating', 'Recommended IND'])
    
    # 2. Xử lý văn bản (Hàm đơn giản hóa)
    def simple_clean(text):
        text = str(text).lower()
        text = re.sub(r'[^\w\s]', '', text)
        return text
    
    df['Clean_Text'] = df['Review Text'].apply(simple_clean)

    # --- C. Chuẩn bị dữ liệu Train ---
    # 1. Vector hóa văn bản (TF-IDF)
    # Giới hạn 2000 từ để train cho nhanh trên Web
    tfidf = TfidfVectorizer(max_features=2000, stop_words='english')
    X_text = tfidf.fit_transform(df['Clean_Text'])

    # 2. Xử lý số (Age, Rating)
    scaler = StandardScaler()
    X_num = scaler.fit_transform(df[['Age', 'Rating']])

    # 3. Ghép lại (Hybrid)
    X_final = sp.hstack((X_text, X_num))
    y = df['Recommended IND']

    # --- D. Huấn luyện Mô hình ---
    # Dùng class_weight='balanced' thay cho SMOTE để train nhanh hơn mà vẫn cân bằng
    model = LogisticRegression(class_weight='balanced', max_iter=1000)
    model.fit(X_final, y)

    return model, tfidf, scaler

# Gọi hàm huấn luyện (Chỉ chạy 1 lần khi mở web nhờ @st.cache)
with st.spinner('Đang tải dữ liệu và huấn luyện AI... Vui lòng đợi 30s...'):
    model, tfidf, scaler = load_and_train_model()

# ================== 3. GIAO DIỆN NGƯỜI DÙNG ==================
st.title("🛍️ DỰ ĐOÁN CẢM XÚC KHÁCH HÀNG")
st.markdown("Hệ thống tự động học từ dữ liệu và dự đoán: **Hài Lòng** hay **Thất Vọng**.")
st.markdown("---")

if model is not None:
    # Form nhập liệu
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Tuổi khách hàng", 18, 90, 25)
    with col2:
        rating = st.slider("Rating (Sao)", 1, 5, 5)
    
    review_input = st.text_area("Nội dung Review (Tiếng Anh)", height=100, 
                               placeholder="Example: The dress fits perfectly and looks amazing!")

    # Nút dự đoán
    if st.button("🔍 PHÂN TÍCH NGAY", type="primary"):
        if not review_input.strip():
            st.warning("Vui lòng nhập nội dung review!")
        else:
            try:
                # 1. Xử lý input giống hệt lúc train
                clean_input = review_input.lower()
                clean_input = re.sub(r'[^\w\s]', '', clean_input)
                
                # 2. Biến đổi
                vec_text = tfidf.transform([clean_input])
                vec_num = scaler.transform([[age, rating]])
                
                # 3. Ghép
                vec_final = sp.hstack((vec_text, vec_num))
                
                # 4. Dự đoán
                pred = model.predict(vec_final)[0]
                proba = model.predict_proba(vec_final).max() * 100
                
                # 5. Kết quả
                st.markdown("---")
                if pred == 1:
                    st.success(f"😊 **DỰ ĐOÁN: TÍCH CỰC (Hài lòng)** - Độ tin cậy: {proba:.1f}%")
                    st.balloons()
                else:
                    st.error(f"☹️ **DỰ ĐOÁN: TIÊU CỰC (Thất vọng)** - Độ tin cậy: {proba:.1f}%")
            except Exception as e:
                st.error(f"Lỗi: {e}")