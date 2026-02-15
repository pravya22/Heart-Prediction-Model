import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib

# -------------------------
# Page Configuration
# -------------------------
st.set_page_config(page_title="Heart Disease Prediction", layout="wide")

# -------------------------
# Custom Styling
# -------------------------
st.markdown("""
    <style>
    .main {
        background-color: #f5f7fa;
    }
    h1 {
        color: #c0392b;
    }
    </style>
""", unsafe_allow_html=True)

# -------------------------
# Title
# -------------------------
st.title("❤️ Heart Disease Prediction System")
st.write("Machine Learning Project - Predicting Heart Disease Risk")

# -------------------------
# Load Model
# -------------------------
model = joblib.load("heart_disease_model.pkl")

# -------------------------
# Sidebar - Patient Input
# -------------------------
st.sidebar.header("Enter Patient Details")

age = st.sidebar.slider("Age", 20, 80, 40)
sex = st.sidebar.selectbox("Sex", ["Female", "Male"])
cp = st.sidebar.selectbox("Chest Pain Type", ["Type 0", "Type 1", "Type 2", "Type 3"])
chol = st.sidebar.slider("Cholesterol Level", 100, 600, 200)
thalch = st.sidebar.slider("Maximum Heart Rate", 60, 220, 150)
exang = st.sidebar.selectbox("Exercise Induced Angina", ["No", "Yes"])

# Convert inputs
sex = 1 if sex == "Male" else 0
exang = 1 if exang == "Yes" else 0
cp = int(cp.split()[1])

input_data = pd.DataFrame([[age, sex, cp, chol, thalch, exang]],
                          columns=["age", "sex", "cp", "chol", "thalch", "exang"])

# -------------------------
# Prediction Section
# -------------------------
st.subheader("Prediction Result")

if st.button("Predict"):
    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.error("⚠️ High Risk of Heart Disease")
    else:
        st.success("✅ Low Risk of Heart Disease")

# -------------------------
# Model Comparison Section
# -------------------------
st.subheader("Model Accuracy Comparison")

models = ["Logistic Regression", "Decision Tree", "Random Forest"]
accuracies = [0.80, 0.75, 0.80]  # Replace with your real values if needed

fig1, ax1 = plt.subplots()
ax1.bar(models, accuracies)
ax1.set_ylabel("Accuracy")
ax1.set_ylim(0.6, 1)
plt.xticks(rotation=20)

st.pyplot(fig1)

# -------------------------
# Feature Importance Section
# -------------------------
st.subheader("Feature Importance")

importance = abs(model.coef_[0])
features = ["age", "sex", "cp", "chol", "thalch", "exang"]

fig2, ax2 = plt.subplots()
ax2.bar(features, importance)
ax2.set_title("Feature Importance")
plt.xticks(rotation=45)

st.pyplot(fig2)

# -------------------------
# Project Description
# -------------------------
st.subheader("Project Description")

st.write("""
This project predicts the risk of heart disease using Machine Learning.

• 6 Important medical features were selected.
• Missing values were handled using median and mode.
• 3 ML models were trained and compared.
• Logistic Regression achieved the highest accuracy.
• The model predicts whether a patient is at High Risk or Low Risk of heart disease.
""")
