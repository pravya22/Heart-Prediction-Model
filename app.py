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
trestbps = st.sidebar.slider("Resting Blood Pressure", 90, 200, 120)
chol = st.sidebar.slider("Cholesterol Level", 100, 600, 200)
fbs = st.sidebar.selectbox("Fasting Blood Sugar > 120 mg/dl", ["No", "Yes"])
restecg = st.sidebar.selectbox("Resting ECG Result", ["Normal", "ST-T Abnormality", "Left Ventricular Hypertrophy"])
thalch = st.sidebar.slider("Maximum Heart Rate", 60, 220, 150)
exang = st.sidebar.selectbox("Exercise Induced Angina", ["No", "Yes"])
oldpeak = st.sidebar.slider("ST Depression Induced by Exercise", 0.0, 6.0, 1.0)
slope = st.sidebar.selectbox("Slope of ST Segment", ["Upsloping", "Flat", "Downsloping"])
ca = st.sidebar.slider("Number of Major Vessels Colored by Fluoroscopy (0-3)", 0, 3, 0)
thal = st.sidebar.selectbox("Thalassemia Type", ["Normal", "Fixed Defect", "Reversible Defect"])

# Convert categorical inputs to numeric
sex = 1 if sex == "Male" else 0
fbs = 1 if fbs == "Yes" else 0
exang = 1 if exang == "Yes" else 0
cp = int(cp.split()[1])
restecg_dict = {"Normal": 0, "ST-T Abnormality": 1, "Left Ventricular Hypertrophy": 2}
restecg = restecg_dict[restecg]
slope_dict = {"Upsloping": 0, "Flat": 1, "Downsloping": 2}
slope = slope_dict[slope]
thal_dict = {"Normal": 1, "Fixed Defect": 2, "Reversible Defect": 3}
thal = thal_dict[thal]

# Prepare input dataframe
input_data = pd.DataFrame([[age, sex, cp, trestbps, chol, fbs, restecg,
                            thalch, exang, oldpeak, slope, ca, thal]],
                          columns=["age","sex","cp","trestbps","chol","fbs","restecg",
                                   "thalch","exang","oldpeak","slope","ca","thal"])

# -------------------------
# Prediction Section
# -------------------------
st.subheader("Prediction Result")

if st.button("Predict"):
    prediction = model.predict(input_data)
    prediction_proba = model.predict_proba(input_data)[:,1]  # probability of high risk
    st.write(f"Probability of High Risk: {prediction_proba[0]:.2f}")

    if prediction[0] == 1:
        st.error("⚠️ High Risk of Heart Disease")
    else:
        st.success("✅ Low Risk of Heart Disease")

# -------------------------
# Model Accuracy Comparison
# -------------------------
st.subheader("Model Accuracy Comparison")

models = ["Logistic Regression", "Decision Tree", "Random Forest"]
accuracies = [0.80, 0.75, 0.80]  # Replace with real values

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

try:
    importance = abs(model.coef_[0])
    features = input_data.columns
    fig2, ax2 = plt.subplots()
    ax2.bar(features, importance)
    ax2.set_title("Feature Importance")
    plt.xticks(rotation=45)
    st.pyplot(fig2)
except:
    st.write("Feature importance not available for this model.")

# -------------------------
# Project Description
# -------------------------
st.subheader("Project Description")

st.write("""
This project predicts the risk of heart disease using Machine Learning.

• 13 medical features are considered.
• Missing values are handled during training.
• 3 ML models were trained and compared.
• Logistic Regression achieved the highest accuracy.
• The model predicts whether a patient is at High Risk or Low Risk of heart disease.
""")
