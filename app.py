import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import time

# Page config
st.set_page_config(
    page_title="Heart Disease Predictor",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3.5rem !important;
        font-weight: 700 !important;
        color: #2c3e50 !important;
        text-align: center;
        margin-bottom: 2rem !important;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
    }
    .high-risk {
        background: linear-gradient(135deg, #ff6b6b, #ee5a24) !important;
        animation: pulse 2s infinite;
    }
    .low-risk {
        background: linear-gradient(135deg, #00b894, #00a085) !important;
    }
    @keyframes pulse {
        0% { box-shadow: 0 0 0 0 rgba(255,107,107,0.7); }
        70% { box-shadow: 0 0 0 20px rgba(255,107,107,0); }
        100% { box-shadow: 0 0 0 0 rgba(255,107,107,0); }
    }
    .stSlider > div > div > div {
        background: linear-gradient(90deg, #667eea, #764ba2);
    }
    </style>
""", unsafe_allow_html=True)

# Load model
@st.cache_resource
def load_model():
    return joblib.load('heartdiseasemodel.pkl')

model = load_model()

# Title
st.markdown('<h1 class="main-header">🫀 Heart Disease Predictor</h1>', unsafe_allow_html=True)
st.markdown("### Professional ML System • 78.8% Accuracy • 6 Key Features")

# Sidebar - Patient Info
st.sidebar.markdown("## 👤 Patient Information")
st.sidebar.markdown("---")

col1, col2 = st.columns(2)
with col1:
    age = st.slider("**Age**", 20, 80, 45, help="Patient age")
with col2:
    sex = st.selectbox("**Sex**", ["Male", "Female"], index=0)

col1, col2 = st.columns(2)
with col1:
    cp = st.selectbox("**Chest Pain Type**", 
                     ["Type 0: Typical Angina", "Type 1: Atypical Angina", 
                      "Type 2: Non-Anginal", "Type 3: Asymptomatic"], index=0)
    cp_value = int(cp.split()[-1])  # Extract number
with col2:
    chol = st.slider("**Cholesterol (mg/dl)**", 100, 600, 250, help="Serum cholesterol")

col1, col2 = st.columns(2)
with col1:
    thalch = st.slider("**Max Heart Rate**", 60, 220, 150)
with col2:
    exang = st.selectbox("**Exercise Angina**", ["No", "Yes"])

st.sidebar.markdown("---")
if st.sidebar.button("🔬 **Generate Prediction**", type="primary", use_container_width=True):
    with st.spinner("Analyzing heart risk..."):
        time.sleep(1.5)
        
        # Create input data
        input_data = pd.DataFrame({
            'age': [age],
            'sex': [1 if sex == "Male" else 0],
            'cp': [cp_value],
            'chol': [chol],
            'thalch': [thalch],
            'exang': [1 if exang == "Yes" else 0]
        })
        
        # Predict with 40% threshold (MEDICAL STANDARD)
        probability = model.predict_proba(input_data)[0][1]
        is_high_risk = probability > 0.40
        
        # Main Results Section
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("## 🎯 **Prediction Result**")
            if is_high_risk:
                st.markdown("""
                <div class="metric-card high-risk">
                    <h2 style="margin: 0; font-size: 2.5rem;">🚨 HIGH RISK</h2>
                    <p style="margin: 0.5rem 0; font-size: 1.3rem;">Heart Disease Probability</p>
                    <h1 style="margin: 0; font-size: 3.5rem;">{:.1%}</h1>
                </div>
                """.format(probability), unsafe_allow_html=True)
                st.warning("⚠️ **Immediate medical consultation recommended**")
            else:
                st.markdown("""
                <div class="metric-card low-risk">
                    <h2 style="margin: 0; font-size: 2.5rem;">✅ LOW RISK</h2>
                    <p style="margin: 0.5rem 0; font-size: 1.3rem;">Heart Disease Probability</p>
                    <h1 style="margin: 0; font-size: 3.5rem;">{:.1%}</h1>
                </div>
                """.format(probability), unsafe_allow_html=True)
                st.success("🎉 Continue healthy lifestyle")
        
        with col2:
            st.metric("**Risk Score**", f"{probability:.1%}", f"{probability*100:.0f}")
        
        st.markdown("---")
        
        # Feature Contributions
        st.markdown("## 📊 **Feature Analysis**")
        feature_names = ['Age', 'Male', 'Chest Pain', 'Cholesterol', 'Heart Rate', 'Exercise Angina']
        feature_values = [age/80, input_data['sex'].values[0], cp_value/3, chol/600, thalch/220, input_data['exang'].values[0]]
        
        fig = px.bar(x=feature_names, y=feature_values, 
                    title="Your Risk Factors (Normalized)",
                    color=feature_values, color_continuous_scale="RdYlGn_r")
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        
        # Risk Interpretation
        st.markdown("## 💡 **Risk Interpretation**")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.info(f"**Age**: {age} years")
            st.info(f"**Cholesterol**: {chol} mg/dl")
        with col2:
            st.info(f"**Max Heart Rate**: {thalch} bpm")
            st.warning(f"**Chest Pain Type**: {cp.split(':')[1].strip()}")
        with col3:
            st.info(f"**Exercise Angina**: {'Yes' if exang=='Yes' else 'No'}")
            st.info(f"**Sex**: {sex}")

# Performance Metrics Section
with st.expander("📈 **Model Performance**", expanded=False):
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Test Accuracy", "78.8%")
    col2.metric("High Risk Detection", "82% Recall")
    col3.metric("Features Used", "6 Key Medical")
    col4.metric("Threshold", "40% Medical Std")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #7f8c8d; padding: 2rem;'>
    <h3>🩺 Built for Medical Research</h3>
    <p>Powered by Logistic Regression • Trained on 920 patients<br>
    <strong>NOT a substitute for professional medical advice</strong></p>
</div>
""", unsafe_allow_html=True)

# Auto-refresh for demo
if 'prediction_made' not in st.session_state:
    st.session_state.prediction_made = False
