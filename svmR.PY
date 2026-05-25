import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)
from sklearn.preprocessing import StandardScaler

# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Diabetes SVM Regression App",
    layout="wide"
)

# =========================================================
# BACKGROUND COLOR
# =========================================================

st.markdown(
    """
    <style>
    .stApp {
        background-color: #F3E8FF;
    }

    h1, h2, h3 {
        color: #5B21B6;
    }

    .stButton>button {
        background-color: #5B21B6;
        color: white;
        border-radius: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# =========================================================
# TITLE
# =========================================================

st.title("SVM Regression on Diabetes Dataset")

st.write(
    "This application performs Support Vector "
    "Machine Regression on the Diabetes Dataset."
)

# =========================================================
# LOAD DATASET
# =========================================================

df = pd.read_csv("diabetes.csv")

# =========================================================
# DATASET PREVIEW
# =========================================================

st.subheader("Dataset Preview")

st.dataframe(df.head())

# =========================================================
# DATASET SHAPE
# =========================================================

st.subheader("Dataset Shape")

st.write("Rows:", df.shape[0])

st.write("Columns:", df.shape[1])

# =========================================================
# DATA TYPES
# =========================================================

st.subheader("Data Types")

st.write(df.dtypes)

# =========================================================
# MISSING VALUES
# =========================================================

st.subheader("Missing Values")

st.write(df.isnull().sum())

# =========================================================
# STATISTICAL SUMMARY
# =========================================================

st.subheader("Statistical Summary")

st.write(df.describe())

# =========================================================
# CORRELATION HEATMAP
# =========================================================

st.subheader("Correlation Heatmap")

fig, ax = plt.subplots(figsize=(6, 4))

sns.heatmap(
    df.corr(),
    annot=True,
    cmap="coolwarm",
    ax=ax
)

st.pyplot(fig)

# =========================================================
# FEATURES AND TARGET
# =========================================================

X = df.drop("Outcome", axis=1)

y = df["Outcome"]

# =========================================================
# FEATURE COLUMNS
# =========================================================

st.subheader("Feature Columns")

st.write(X.columns.tolist())

# =========================================================
# TRAIN TEST SPLIT
# =========================================================

test_size = st.slider(
    "Select Test Size",
    min_value=0.1,
    max_value=0.5,
    value=0.2,
    step=0.05
)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=test_size,
    random_state=42
)

# =========================================================
# FEATURE SCALING
# =========================================================

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)

# =========================================================
# SELECT KERNEL
# =========================================================

st.subheader("Select SVM Kernel")

kernel = st.selectbox(
    "Choose Kernel",
    ["linear", "rbf", "poly", "sigmoid"]
)

# =========================================================
# MODEL TRAINING
# =========================================================

model = SVR(kernel=kernel)

model.fit(X_train, y_train)

st.success("SVM Regression Model Trained Successfully")

# =========================================================
# PREDICTIONS
# =========================================================

y_pred = model.predict(X_test)

# =========================================================
# MODEL EVALUATION
# =========================================================

mse = mean_squared_error(y_test, y_pred)

mae = mean_absolute_error(y_test, y_pred)

rmse = np.sqrt(mse)

r2 = r2_score(y_test, y_pred)

st.subheader("Model Evaluation")

st.write(f"Mean Absolute Error (MAE): {mae:.4f}")

st.write(f"Mean Squared Error (MSE): {mse:.4f}")

st.write(f"Root Mean Squared Error (RMSE): {rmse:.4f}")

st.write(f"R² Score: {r2:.4f}")

# =========================================================
# ACTUAL VS PREDICTED GRAPH
# =========================================================

st.subheader("Actual vs Predicted")

fig2, ax2 = plt.subplots(figsize=(5, 3))

ax2.scatter(y_test, y_pred)

ax2.set_xlabel("Actual")

ax2.set_ylabel("Predicted")

ax2.set_title("Actual vs Predicted")

st.pyplot(fig2)

# =========================================================
# RESIDUAL PLOT
# =========================================================

st.subheader("Residual Plot")

residuals = y_test - y_pred

fig3, ax3 = plt.subplots(figsize=(5, 3))

ax3.scatter(y_pred, residuals)

ax3.axhline(y=0, color='red')

ax3.set_xlabel("Predicted")

ax3.set_ylabel("Residuals")

ax3.set_title("Residual Plot")

st.pyplot(fig3)

# =========================================================
# CUSTOM USER INPUT
# =========================================================

st.subheader("Predict Diabetes Outcome")

pregnancies = st.number_input(
    "Pregnancies",
    value=float(df["Pregnancies"].mean())
)

glucose = st.number_input(
    "Glucose",
    value=float(df["Glucose"].mean())
)

bloodpressure = st.number_input(
    "BloodPressure",
    value=float(df["BloodPressure"].mean())
)

skinthickness = st.number_input(
    "SkinThickness",
    value=float(df["SkinThickness"].mean())
)

insulin = st.number_input(
    "Insulin",
    value=float(df["Insulin"].mean())
)

bmi = st.number_input(
    "BMI",
    value=float(df["BMI"].mean())
)

diabetespedigreefunction = st.number_input(
    "DiabetesPedigreeFunction",
    value=float(df["DiabetesPedigreeFunction"].mean())
)

age = st.number_input(
    "Age",
    value=float(df["Age"].mean())
)

# =========================================================
# PREDICT BUTTON
# =========================================================

if st.button("Predict"):

    input_data = pd.DataFrame({

        "Pregnancies": [pregnancies],

        "Glucose": [glucose],

        "BloodPressure": [bloodpressure],

        "SkinThickness": [skinthickness],

        "Insulin": [insulin],

        "BMI": [bmi],

        "DiabetesPedigreeFunction": [diabetespedigreefunction],

        "Age": [age]

    })

    # Scale Input Data
    input_scaled = scaler.transform(input_data)

    # Prediction
    prediction = model.predict(input_scaled)

    st.success(
        f"Predicted Diabetes Outcome: {prediction[0]:.4f}"
    )