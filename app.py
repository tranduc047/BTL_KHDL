import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# === 1. Đọc và xử lý dữ liệu ===
@st.cache_data
def load_data():
    df = pd.read_csv("train.csv")

    # Lựa chọn một số cột đặc trưng
    features = [
        "GrLivArea", "BedroomAbvGr", "FullBath", "YearBuilt",
        "GarageCars", "GarageArea", "Neighborhood", "OverallQual", "TotalBsmtSF"
    ]
    df = df[features + ["SalePrice"]]

    # Xử lý missing
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].fillna(df[col].mode()[0])
        else:
            df[col] = df[col].fillna(df[col].median())

    return df

df = load_data()

# === 2. Feature engineering ===
X = df.drop("SalePrice", axis=1)
y = df["SalePrice"]
X = pd.get_dummies(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === 3. Huấn luyện mô hình ===
@st.cache_resource
def train_model(model_type="RandomForest"):
    if model_type == "RandomForest":
        model = RandomForestRegressor(n_estimators=100, random_state=42)
    else:
        model = LinearRegression()
    model.fit(X_train, y_train)
    return model

# === 4. Streamlit App ===
st.title("🏡 Dự đoán giá nhà")

# Sidebar: chọn model
model_type = st.sidebar.radio("Chọn mô hình", ("RandomForest", "LinearRegression"))
model = train_model(model_type)

# Sidebar: nhập dữ liệu
st.sidebar.header("Nhập thông tin nhà")
def user_input():
    grliv = st.sidebar.slider("Diện tích (GrLivArea)", 400, 6000, 1500)
    beds = st.sidebar.slider("Số phòng ngủ", 0, 10, 3)
    baths = st.sidebar.slider("Số phòng tắm", 0, 4, 2)
    garage = st.sidebar.slider("Số chỗ đậu xe", 0, 5, 2)
    garage_area = st.sidebar.slider("Diện tích garage", 0, 1500, 400)
    bsmt = st.sidebar.slider("Diện tích tầng hầm", 0, 2000, 800)
    year = st.sidebar.slider("Năm xây", 1870, 2023, 2000)
    overall = st.sidebar.slider("Chất lượng tổng thể", 1, 10, 5)
    neigh = st.sidebar.selectbox("Khu vực", df["Neighborhood"].unique())

    data = {
        "GrLivArea": grliv,
        "BedroomAbvGr": beds,
        "FullBath": baths,
        "YearBuilt": year,
        "GarageCars": garage,
        "GarageArea": garage_area,
        "TotalBsmtSF": bsmt,
        "OverallQual": overall,
        "Neighborhood": neigh
    }
    return pd.DataFrame([data])

input_df = user_input()

# Chuẩn hóa input
input_df_full = pd.get_dummies(input_df)
input_df_full = input_df_full.reindex(columns=X.columns, fill_value=0)

# Dự đoán
prediction = model.predict(input_df_full)[0]
st.subheader("💰 Giá nhà dự đoán:")
st.success(f"${prediction:,.0f}")

# === 5. Đánh giá mô hình ===
st.subheader("📈 Đánh giá mô hình")

# Dự đoán trên tập test
y_pred = model.predict(X_test)

# RMSE
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

# MAPE
def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

mape = mean_absolute_percentage_error(y_test, y_pred)

st.write(f"**RMSE** (Sai số trung bình): ${rmse:,.2f}")
st.write(f"**MAPE** (Sai số phần trăm): {mape:.2f}%")

# === 6. Phân tích dữ liệu bằng biểu đồ ===
st.subheader("📊 Phân tích dữ liệu")

# Phân phối giá nhà
fig1, ax1 = plt.subplots()
sns.histplot(df["SalePrice"], kde=True, ax=ax1)
ax1.set_title("Phân phối giá nhà")
st.pyplot(fig1)

# Diện tích vs Giá
fig2, ax2 = plt.subplots()
sns.scatterplot(x="GrLivArea", y="SalePrice", data=df, ax=ax2)
ax2.set_title("Diện tích so với Giá")
st.pyplot(fig2)

# Chất lượng tổng thể vs Giá
fig3, ax3 = plt.subplots()
sns.boxplot(x="OverallQual", y="SalePrice", data=df, ax=ax3)
ax3.set_title("Chất lượng tổng thể và Giá")
st.pyplot(fig3)
