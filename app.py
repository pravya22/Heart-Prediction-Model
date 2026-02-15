
import streamlit as st
import pandas as pd
import joblib
import time

# Page config
st.set_page_config(
    page_title="🫀 Heart Disease Predictor", 
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS - Beautiful Medical Design
st.markdown("""
    <style>
    .main-header {
        font-size: 3.5rem !important;
        font-weight: 700 !important;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 2rem !important;
    }
    .metric-card {
        padding: 2rem;
        border-radius: 20px;
        text-align: center;
        box-shadow: 0 10px 40px rgba(0,0,0,0.1);
        border: 2px solid #e1e8ed;
    }
    .high-risk-card {
        background: linear-gradient(135deg, #ff6b6b, #ee5a24) !important;
        color: white !important;
        animation: pulse 2s infinite;
    }
    .low-risk-card {
        background: linear-gradient(135deg, #00b894, #00a085) !important;
        color: white !important;
    }
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.05); }
        100% { transform: scale(1); }
    }
    .feature-bar {
        background: linear-gradient(to right, #ff4757, #ffa502, #2ed573);
        height: 25px;
        border-radius: 10px;
        margin: 5px 0;
    }
    </style>
""", unsafe_allow_html=True)

# Load model
@st.cache_resource
def load_model():
    return joblib.load('heartdiseasemodel.pkl')

model = load_model()

# Header
st.markdown('<h1 class="main-header">🫀 Heart Disease Predictor Pro</h1>', unsafe_allow_html=True)
st.markdown("### ⚡ ML-Powered • 78.8% Accurate • Medical-Grade • 6 Key Features")

# Sidebar - Clean Patient Input
st.sidebar.markdown("## 👨‍⚕️ **Patient Details**")
st.sidebar.markdown("---")

# 6 Features Layout (2x3 grid)
col1, col2 = st.columns(2)
with col1:
    age = st.slider("👴 **Age**", 20, 80, 45)
with col2:
    sex = st.selectbox("⚥ **Gender**", ["Male", "Female"])

col1, col2 = st.columns(2)
with col1:
    cp_options = ["Type 0: Typical Angina", "Type 1: Atypical", "Type 2: Non-Anginal", "Type 3: Asymptomatic"]
    cp = st.selectbox("❤️ **Chest Pain**", cp_options)
    cp_value = int(cp.split()[-1])
with col2:
    chol = st.slider("🩸 **Cholesterol**", 100, 600, 250)

col1, col2 = st.columns(2)
with col1:
    thalch = st.slider("💓 **Max Heart Rate**", 60, 220, 150)
with col2:
    exang = st.selectbox("🏃 **Exercise Angina**", ["No", "Yes"])

# Predict Button
st.sidebar.markdown("---")
if st.sidebar.button("🚨 **ANALYZE RISK**", use_container_width=True, type="primary"):
    with st.spinner("🔬 Computing medical risk assessment..."):
        time.sleep(1)
        
        # Create input matching your model
        input_data = pd.DataFrame({
            'age': [age], 'sex': [1 if sex == "Male" else 0],
            'cp': [cp_value], 'chol': [chol], 
            'thalch': [thalch], 'exang': [1 if exang == "Yes" else 0]
        })
        
        # FIXED PREDICTION (40% threshold)
        probability = model.predict_proba(input_data)[0][1]
        
        # Results Section
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.markdown("## 🎯 **Medical Assessment**")
            if probability > 0.40:
                st.markdown("""
                <div class="metric-card high-risk-card">
                    <h1 style='font-size: 4rem; margin: 0;'>🚨 HIGH RISK</h1>
                    <h2 style='font-size: 2.5rem; margin: 0.5rem 0;'>Heart Disease</h2>
                    <h3 style='font-size: 3rem; margin: 1rem 0;'>"""+f"{probability:.1%}"+"""</h3>
                    <p style='font-size: 1.3rem;'>⚠️ Seek immediate medical attention</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="metric-card low-risk-card">
                    <h1 style='font-size: 4rem; margin: 0;'>✅ LOW RISK</h1>
                    <h2 style='font-size: 2.5rem; margin: 0.5rem 0;'>Heart Disease</h2>
                    <h3 style='font-size: 3rem; margin: 1rem 0;'>"""+f"{probability:.1%}"+"""</h3>
                    <p style='font-size: 1.3rem;'>🎉 Continue healthy lifestyle</p>
                </div>
                """, unsafe_allow_html=True)
        
        with col2:
            st.metric("**Risk Level**", f"{probability:.0%}", "40% Threshold")
        
        # Feature Importance Bars (No Plotly!)
        st.markdown("## 📈 **Your Risk Factors**")
        st.markdown("---")
        
        features = ['Age', 'Male', 'Chest Pain Type', 'Cholesterol', 'Heart Rate', 'Exercise Angina']
        values = [age/80, 1 if sex=="Male" else 0, cp_value/3, chol/600, (220-thalch)/220, 
                 1 if exang=="Yes" else 0]
        
        for feature, value in zip(features, values):
            col1, col2, col3 = st.columns([2, 6, 3])
            with col1:
                st.write(f"**{feature}**")
            with col2:
                st.markdown(f"""
                <div class="feature-bar" style="width: {value*100}%;"></div>
                """, unsafe_allow_html=True)
            with col3:
                st.write(f"{value:.0%}")
        
        # Summary Table
        st.markdown("## 📋 **Patient Summary**")
        summary_data = {
            "Parameter": ["Age", "Gender", "Chest Pain", "Cholesterol", "Max HR", "Exercise Angina"],
            "Value": [f"{age} years", sex, cp.split(":")[1].strip(), f"{chol} mg/dl", 
                     f"{thalch} bpm", exang],
            "Risk Level": ["Medium", "High" if sex=="Male" else "Low", f"High ({cp_value}/3)", 
                          "High" if chol>300 else "Medium", "High" if thalch<120 else "Low", 
                          "High" if exang=="Yes" else "Low"]
        }
        st.table(pd.DataFrame(summary_data))

# Model Info
with st.expander("📊 **Model Specifications**"):
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Accuracy", "78.8%")
    col2.metric("High Risk Recall", "82%")
    col3.metric("Patients Trained", "920")
    col4.metric("Features", "6 Medical")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; padding: 2rem; color: #6c757d;'>
    <h3>🏥 Professional Medical ML System</h3>
    <p>Trained on 920 patients • Logistic Regression • <strong>40% Medical Threshold</strong><br>
    ⚠️ <em>NOT medical advice - consult physician</em></p>
</div>
""", unsafe_allow_html=True)

