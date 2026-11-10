import re
import io
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(page_title="Dự đoán giá xe cũ - Nhóm 2", page_icon="🚗", layout="wide")

st.title("🚗 ỨNG DỤNG DỰ ĐOÁN GIÁ XE CŨ")
st.caption("Nhóm 2 | Linear Regression • Random Forest • K-means")
st.info(
    "Ứng dụng được xây dựng theo báo cáo học thuật của Nhóm 2. "
    "Dữ liệu nghiên cứu là thị trường xe cũ Mỹ, giá tính bằng USD. "
    "Kết quả chỉ có giá trị tham khảo, không phải báo giá mua bán chính thức."
)

NUMERIC_FEATURES = ["model_year", "milage", "hp", "liter", "accident", "clean_title"]
CATEGORICAL_FEATURES = ["brand", "fuel_type", "transmission"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
CLUSTER_FEATURES = ["model_year", "milage", "hp", "liter", "price"]

def to_number(value):
    """Chuyển chuỗi có dấu phẩy, đơn vị hoặc ký hiệu tiền thành số."""
    if pd.isna(value):
        return np.nan
    if isinstance(value, (int, float, np.number)):
        return float(value)
    s = str(value).replace(",", "")
    match = re.search(r"-?\d+(?:\.\d+)?", s)
    return float(match.group()) if match else np.nan

def parse_engine(engine):
    """Lấy mã lực (HP) và dung tích (L) từ cột engine."""
    if pd.isna(engine):
        return pd.Series({"hp": np.nan, "liter": np.nan})
    s = str(engine).replace(",", "")
    hp_match = re.search(r"(\d+(?:\.\d+)?)\s*HP", s, re.I)
    liter_match = re.search(r"(\d+(?:\.\d+)?)\s*L(?:\s|$)", s, re.I)
    return pd.Series({
        "hp": float(hp_match.group(1)) if hp_match else np.nan,
        "liter": float(liter_match.group(1)) if liter_match else np.nan
    })

def yes_no(value):
    if pd.isna(value):
        return np.nan
    s = str(value).strip().lower()
    if s in {"1", "yes", "y", "true"} or "clean" in s:
        return 1
    if s in {"0", "no", "n", "false"}:
        return 0
    return np.nan

def accident_flag(value):
    if pd.isna(value):
        return np.nan
    s = str(value).strip().lower()
    if s in {"0", "no", "none", "no accident"} or "none reported" in s:
        return 0
    if s in {"1", "yes", "true"} or "accident" in s or "damage" in s:
        return 1
    return np.nan

def prepare_data(raw):
    df = raw.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]

    # Hỗ trợ tên cột thường gặp trong Used Car Price Prediction Dataset.
    if "price" not in df.columns:
        raise ValueError("Không tìm thấy cột 'price' trong dữ liệu.")
    if "model_year" not in df.columns or "milage" not in df.columns:
        raise ValueError("Dữ liệu cần có hai cột 'model_year' và 'milage'.")

    df["price"] = df["price"].apply(to_number)
    df["model_year"] = df["model_year"].apply(to_number)
    df["milage"] = df["milage"].apply(to_number)

    if "engine" in df.columns:
        parsed = df["engine"].apply(parse_engine)
        if "hp" not in df.columns:
            df["hp"] = parsed["hp"]
        else:
            df["hp"] = df["hp"].fillna(parsed["hp"])
        if "liter" not in df.columns:
            df["liter"] = parsed["liter"]
        else:
            df["liter"] = df["liter"].fillna(parsed["liter"])

    for col in ["hp", "liter"]:
        if col not in df.columns:
            df[col] = np.nan
        df[col] = df[col].apply(to_number)

    if "accident" in df.columns:
        df["accident"] = df["accident"].apply(accident_flag)
    else:
        df["accident"] = np.nan

    if "clean_title" in df.columns:
        df["clean_title"] = df["clean_title"].apply(yes_no)
    else:
        df["clean_title"] = np.nan

    for col in CATEGORICAL_FEATURES:
        if col not in df.columns:
            df[col] = "Unknown"
        df[col] = df[col].fillna("Unknown").astype(str).str.strip()
        df.loc[df[col].eq(""), col] = "Unknown"

    # Loại các dòng không có giá mục tiêu hoặc các giá trị không hợp lệ.
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["price"])
    df = df[df["price"] > 0]
    if len(df) < 10:
        raise ValueError("Dữ liệu hợp lệ còn quá ít. Vui lòng kiểm tra lại tệp.")

    for col in NUMERIC_FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

def build_pipeline(model, scale_numeric=False):
    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))
    numeric_pipe = Pipeline(numeric_steps)
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])
    preprocessor = ColumnTransformer([
        ("num", numeric_pipe, NUMERIC_FEATURES),
        ("cat", categorical_pipe, CATEGORICAL_FEATURES)
    ])
    return Pipeline([("preprocess", preprocessor), ("model", model)])

def money(value):
    return f"${value:,.0f} USD"

with st.sidebar:
    st.header("📂 Dữ liệu huấn luyện")
    uploaded = st.file_uploader("Tải bộ dữ liệu CSV hoặc Excel", type=["csv", "xlsx", "xls"])
    st.markdown(
        "**Bộ dữ liệu theo báo cáo:** "
        "[Used Car Price Prediction Dataset trên Kaggle]"
        "(https://www.kaggle.com/datasets/taeefnajib/used-car-price-prediction-dataset)"
    )
    st.caption("Tệp cần có cột price, model_year, milage. Nếu có engine, ứng dụng tự trích hp và liter.")

if uploaded is None:
    st.warning("Hãy tải tệp dữ liệu xe cũ ở thanh bên trái để bắt đầu huấn luyện mô hình.")
    st.subheader("Các bước sử dụng")
    st.markdown(
        "1. Tải dữ liệu CSV/XLSX từ Kaggle.\n"
        "2. Chọn tệp ở thanh bên trái.\n"
        "3. Nhấn **Huấn luyện mô hình** ở tab Huấn luyện.\n"
        "4. Xem kết quả, so sánh thuật toán và nhập thông số xe để dự đoán."
    )
    st.stop()

try:
    if uploaded.name.lower().endswith(".csv"):
        raw_df = pd.read_csv(uploaded)
    else:
        raw_df = pd.read_excel(uploaded)
    data = prepare_data(raw_df)
except Exception as exc:
    st.error(f"Không đọc hoặc xử lý được dữ liệu: {exc}")
    st.stop()

st.success(f"Đã đọc dữ liệu: {len(data):,} dòng hợp lệ.")
with st.expander("Xem trước dữ liệu sau tiền xử lý"):
    st.dataframe(data[FEATURES + ["price"]].head(10), use_container_width=True)
    st.write(f"Kích thước dữ liệu: {data.shape[0]:,} dòng × {data.shape[1]} cột")

tab_train, tab_predict, tab_cluster, tab_about = st.tabs([
    "📊 Huấn luyện & đánh giá", "🚘 Dự đoán giá xe", "🧩 Phân cụm K-means", "ℹ️ Giới thiệu"
])

with tab_train:
    st.subheader("Huấn luyện hai thuật toán")
    test_size = st.slider("Tỷ lệ dữ liệu kiểm tra", min_value=0.1, max_value=0.4, value=0.2, step=0.05)
    n_estimators = st.slider("Số cây Random Forest", 50, 500, 200, 50)
    if st.button("Huấn luyện mô hình", type="primary"):
        X = data[FEATURES]
        y = data["price"]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        linear = build_pipeline(LinearRegression(), scale_numeric=True)
        forest = build_pipeline(
            RandomForestRegressor(n_estimators=n_estimators, random_state=42, n_jobs=-1),
            scale_numeric=False
        )
        with st.spinner("Đang huấn luyện hai mô hình..."):
            linear.fit(X_train, y_train)
            forest.fit(X_train, y_train)
        pred_linear = np.maximum(0, linear.predict(X_test))
        pred_forest = np.maximum(0, forest.predict(X_test))

        def metrics(y_true, y_pred):
            return {
                "MAE (USD)": mean_absolute_error(y_true, y_pred),
                "RMSE (USD)": np.sqrt(mean_squared_error(y_true, y_pred)),
                "R²": r2_score(y_true, y_pred)
            }

        results = pd.DataFrame({
            "Linear Regression": metrics(y_test, pred_linear),
            "Random Forest": metrics(y_test, pred_forest)
        }).T
        st.session_state["models"] = {"Linear Regression": linear, "Random Forest": forest}
        st.session_state["metrics"] = results
        st.session_state["test_data"] = X_test.copy()
        st.session_state["y_test"] = y_test.copy()
        st.session_state["predictions"] = {
            "Linear Regression": pred_linear,
            "Random Forest": pred_forest
        }
        st.success("Huấn luyện hoàn tất!")

    if "metrics" in st.session_state:
        st.markdown("#### Bảng chỉ số đánh giá trên tập test")
        st.dataframe(st.session_state["metrics"].style.format({
            "MAE (USD)": "${:,.0f}", "RMSE (USD)": "${:,.0f}", "R²": "{:.4f}"
        }), use_container_width=True)
        best = st.session_state["metrics"]["MAE (USD)"].idxmin()
        st.info(f"Theo MAE, mô hình có sai số trung bình thấp hơn là: **{best}**.")
        model_choice = st.selectbox("Xem bảng dự đoán của mô hình", ["Linear Regression", "Random Forest"])
        y_true = st.session_state["y_test"].to_numpy()
        y_pred = st.session_state["predictions"][model_choice]
        comparison = pd.DataFrame({
            "Giá thực tế (USD)": y_true,
            "Giá dự đoán (USD)": y_pred,
            "Sai lệch tuyệt đối (USD)": np.abs(y_true - y_pred)
        })
        st.dataframe(comparison.head(20).round(0), use_container_width=True)
        st.download_button(
            "⬇️ Tải bảng dự đoán CSV",
            comparison.to_csv(index=False).encode("utf-8-sig"),
            file_name="ket_qua_du_doan_gia_xe.csv",
            mime="text/csv"
        )
        fig, ax = plt.subplots()
        ax.scatter(y_true, y_pred, alpha=0.5)
        lim = max(float(np.max(y_true)), float(np.max(y_pred)))
        ax.plot([0, lim], [0, lim], linestyle="--")
        ax.set_xlabel("Giá thực tế (USD)")
        ax.set_ylabel("Giá dự đoán (USD)")
        ax.set_title(f"Giá thực tế và dự đoán - {model_choice}")
        st.pyplot(fig)
        plt.close(fig)
    else:
        st.caption("Chưa có kết quả. Nhấn nút Huấn luyện mô hình để bắt đầu.")

with tab_predict:
    st.subheader("Nhập thông tin xe cần dự đoán")
    if "models" not in st.session_state:
        st.warning("Bạn cần huấn luyện mô hình trước ở tab “Huấn luyện & đánh giá”.")
    else:
        c1, c2, c3 = st.columns(3)
        with c1:
            year = st.number_input("Năm sản xuất", min_value=1980, max_value=2026, value=2019, step=1)
            mileage = st.number_input("Số dặm đã đi (mile)", min_value=0, value=50000, step=1000)
            hp = st.number_input("Công suất (HP)", min_value=1.0, value=300.0, step=5.0)
        with c2:
            liter = st.number_input("Dung tích động cơ (L)", min_value=0.1, value=3.5, step=0.1)
            brands = sorted(data["brand"].dropna().astype(str).unique().tolist())
            brand = st.selectbox("Hãng xe", brands)
            fuels = sorted(data["fuel_type"].dropna().astype(str).unique().tolist())
            fuel = st.selectbox("Nhiên liệu", fuels)
        with c3:
            transmissions = sorted(data["transmission"].dropna().astype(str).unique().tolist())
            transmission = st.selectbox("Hộp số", transmissions)
            accident = st.selectbox("Từng bị tai nạn?", ["Không", "Có"])
            clean_title = st.selectbox("Giấy tờ sạch?", ["Có", "Không"])
            model_name = st.selectbox("Thuật toán dự đoán", ["Random Forest", "Linear Regression"])
        if st.button("🔍 Dự đoán giá xe", type="primary"):
            row = pd.DataFrame([{
                "model_year": year, "milage": mileage, "hp": hp, "liter": liter,
                "accident": 1 if accident == "Có" else 0,
                "clean_title": 1 if clean_title == "Có" else 0,
                "brand": brand, "fuel_type": fuel, "transmission": transmission
            }], columns=FEATURES)
            prediction = float(st.session_state["models"][model_name].predict(row)[0])
            prediction = max(0, prediction)
            st.metric("Giá xe dự đoán", money(prediction))
            st.caption("Đây là giá ước tính theo dữ liệu huấn luyện, không phải giá giao dịch được bảo đảm.")

with tab_cluster:
    st.subheader("Phân cụm xe cũ bằng K-means (k = 3)")
    st.write("Các thuộc tính sử dụng: năm sản xuất, số dặm, mã lực, dung tích động cơ và giá xe.")
    cluster_data = data[CLUSTER_FEATURES].dropna().copy()
    if len(cluster_data) < 3:
        st.warning("Không đủ dòng có đầy đủ 5 thuộc tính để phân cụm.")
    else:
        scaler = StandardScaler()
        Z = scaler.fit_transform(cluster_data)
        km = KMeans(n_clusters=3, init="k-means++", n_init=10, random_state=42)
        labels = km.fit_predict(Z)
        clustered = cluster_data.copy()
        clustered["cluster"] = labels
        summary = clustered.groupby("cluster").agg(
            So_xe=("price", "size"),
            Nam_san_xuat_TB=("model_year", "mean"),
            So_dam_TB=("milage", "mean"),
            Ma_luc_TB=("hp", "mean"),
            Gia_TB_USD=("price", "mean")
        ).round(1)
        st.dataframe(summary, use_container_width=True)
        fig, ax = plt.subplots()
        scatter = ax.scatter(clustered["milage"], clustered["price"], c=labels, alpha=0.6)
        ax.set_xlabel("Số dặm đã đi (mile)")
        ax.set_ylabel("Giá xe (USD)")
        ax.set_title("Kết quả phân cụm K-means (k = 3)")
        st.pyplot(fig)
        plt.close(fig)
        st.download_button(
            "⬇️ Tải dữ liệu đã gắn cụm CSV",
            clustered.to_csv(index=False).encode("utf-8-sig"),
            file_name="nhom2_gia_xe_phan_cum.csv",
            mime="text/csv"
        )

with tab_about:
    st.subheader("Thông tin đề tài")
    st.markdown("""
    **Đề tài:** Thực hành huấn luyện mô hình AI – Dự đoán giá xe cũ.

    **Nguồn dữ liệu:** Used Car Price Prediction Dataset trên Kaggle.

    **Biến mục tiêu:** `price` (USD).

    **Thuật toán:**
    - Linear Regression: mô hình hồi quy tuyến tính, dùng làm mô hình nền để đối chiếu.
    - Random Forest: kết hợp nhiều cây quyết định, có khả năng mô tả quan hệ phi tuyến.
    - K-means: phân nhóm xe có đặc điểm tương đồng thành 3 cụm.

    **Lưu ý:** Ứng dụng tự huấn luyện lại mô hình từ tệp dữ liệu bạn tải lên. Vì vậy, kết quả có thể khác số liệu trong báo cáo nếu dữ liệu, cách tiền xử lý hoặc tham số không hoàn toàn giống nhau.
    """)
