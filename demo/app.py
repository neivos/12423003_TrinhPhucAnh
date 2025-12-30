import streamlit as st
import pandas as pd
import joblib

# ================= 1. SETUP & LOAD RESOURCES =================
st.set_page_config(page_title="Credit Risk Prediction", layout="centered")
st.title("💳 Credit Risk Prediction App")
st.write("Nhập thông tin khách hàng để dự đoán rủi ro tín dụng")

@st.cache_resource
def load_resources():
    # Load các mô hình vào một Dictionary để dùng cho Selectbox
    # (Đảm bảo bạn đã có các file .pkl này từ bước Training)
    models = {
        "XGBoost": joblib.load("xgboost.pkl"),
        # Nếu bạn có các model khác thì bỏ comment dòng dưới:
         "Random Forest": joblib.load("random_forest.pkl"), 
         "Decision Tree": joblib.load("decision_tree.pkl"),
    }
    
    # Load Encoders
    encoders = {
        col: joblib.load(f"{col}_encoder.pkl")
        for col in ["Sex", "Housing", "Saving accounts", "Checking account"]
    }
    
    return models, encoders

try:
    models, encoders = load_resources()
except FileNotFoundError as e:
    st.error(f"Thiếu file mô hình hoặc encoder: {e}")
    st.stop()

# ================= 2. MODEL SELECTION (THÊM MỚI) =================
# Tạo Selectbox để chọn thuật toán
selected_model_name = st.selectbox("🤖 Chọn mô hình dự báo:", list(models.keys()))
model = models[selected_model_name]

# ================= 3. INPUT FORM =================
with st.form("input_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input("Tuổi (Age)", min_value=18, max_value=80, value=30)
        sex = st.selectbox("Giới tính (Sex)", encoders["Sex"].classes_)
        job = st.number_input("Công việc (Job 0-3)", min_value=0, max_value=3, value=1)
        housing = st.selectbox("Nhà ở (Housing)", encoders["Housing"].classes_)
        
    with col2:
        saving_accounts = st.selectbox("TK Tiết kiệm (Saving)", encoders["Saving accounts"].classes_)
        checking_account = st.selectbox("TK Vãng lai (Checking)", encoders["Checking account"].classes_)
        credit_amount = st.number_input("Số tiền vay (Credit Amount)", min_value=0, value=1000)
        duration = st.number_input("Kỳ hạn (Duration - months)", min_value=1, value=12)
    
    # Nút submit nằm trong form
    submitted = st.form_submit_button("🔮 Dự báo ngay")

# ================= 4. PREPROCESS & PREDICT =================
if submitted:
    try:
        # Tạo DataFrame từ input
        # Lưu ý: Nếu encoders dùng LabelEncoder, cách transform bên dưới là đúng.
        input_data = {
            "Age": [age],
            "Sex": [encoders["Sex"].transform([sex])[0]],
            "Job": [job],
            "Housing": [encoders["Housing"].transform([housing])[0]],
            "Saving accounts": [encoders["Saving accounts"].transform([saving_accounts])[0]],
            "Checking account": [encoders["Checking account"].transform([checking_account])[0]],
            "Credit amount": [credit_amount],
            "Duration": [duration]
            # CẢNH BÁO: Nếu lúc train bạn có cột 'Purpose', bạn phải thêm vào đây, 
            # nếu không model sẽ báo lỗi thiếu feature.
        }
        
        input_df = pd.DataFrame(input_data)

        # Đảm bảo đúng thứ tự cột như lúc train
        if hasattr(model, "feature_names_in_"):
            # Kiểm tra xem có thiếu cột nào không
            missing_cols = set(model.feature_names_in_) - set(input_df.columns)
            if missing_cols:
                st.error(f"Lỗi: Thiếu các cột sau trong dữ liệu nhập: {missing_cols}")
                st.stop()
            input_df = input_df[model.feature_names_in_]

        # Dự báo
        pred = model.predict(input_df)[0]
        
        # Lấy xác suất (nếu model hỗ trợ)
        if hasattr(model, "predict_proba"):
            proba_bad = model.predict_proba(input_df)[0][1] # Xác suất lớp 1 (Bad)
        else:
            proba_bad = 0

        st.markdown("---")
        st.subheader(f"Kết quả dự báo từ: {selected_model_name}")

        # ================= LOGIC QUAN TRỌNG ĐÃ SỬA =================
        # Logic chuẩn: 1 = Bad (Rủi ro), 0 = Good (An toàn)
        if pred == 1:
            st.error(f"❌ KẾT QUẢ: RỦI RO CAO (BAD CREDIT)")
            st.write(f"Xác suất vỡ nợ: **{proba_bad*100:.2f}%**")
            st.warning("Hệ thống khuyến nghị: **Cân nhắc từ chối** hoặc yêu cầu thêm tài sản đảm bảo.")
        else:
            st.success(f"✅ KẾT QUẢ: AN TOÀN (GOOD CREDIT)")
            st.write(f"Độ tin cậy: **{(1-proba_bad)*100:.2f}%**")
            st.info("Hệ thống khuyến nghị: **Phê duyệt khoản vay**.")

    except Exception as e:
        st.error(f"Đã xảy ra lỗi trong quá trình dự báo: {e}")