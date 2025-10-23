# ============================================================
# 🧠 AI DOCTOR — Hybrid Predictor (SVM + Random Forest)
# ============================================================

import streamlit as st
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from PIL import Image
import warnings

warnings.filterwarnings('ignore')

# ============================================================
# 🎨 PAGE CONFIGURATION
# ============================================================
st.markdown("""
    <style>
    /* ===== GLOBAL PAGE ===== */
    .main {
        background-color: #0A0F16;
        color: #E6EDF3;
        font-family: "Segoe UI", sans-serif;
    }

    /* ===== BUTTON ===== */
    .stButton>button {
        background: linear-gradient(90deg, #1E90FF, #00C896);
        color: white;
        font-size: 18px;
        padding: 12px 28px;
        border-radius: 10px;
        border: none;
        width: 100%;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 0 8px rgba(0,200,150,0.3);
    }
    .stButton>button:hover {
        transform: scale(1.05);
        box-shadow: 0 0 18px rgba(30,144,255,0.8);
    }

    /* ===== SYMPTOM CARD ===== */
    .symptom-card {
        background: rgba(30, 40, 60, 0.7);
        backdrop-filter: blur(6px);
        padding: 15px;
        border-radius: 12px;
        margin: 10px 0;
        border-left: 4px solid #1E90FF;
        color: #E6EDF3;
        font-weight: 500;
        box-shadow: 0 0 10px rgba(30,144,255,0.1);
        transition: all 0.3s ease;
    }
    .symptom-card:hover {
        transform: scale(1.02);
        box-shadow: 0 0 16px rgba(30,144,255,0.6);
        border-left: 4px solid #00C896;
    }

    /* ===== PREDICTION BOX ===== */
    .prediction-box {
        background: rgba(25, 35, 50, 0.8);
        backdrop-filter: blur(8px);
        padding: 22px;
        border-radius: 12px;
        margin: 10px 0;
        color: #E6EDF3;
        box-shadow: 0 0 20px rgba(0, 200, 150, 0.15);
        border: 1px solid rgba(0,200,150,0.2);
        transition: all 0.4s ease;
    }
    .prediction-box:hover {
        transform: scale(1.02);
        box-shadow: 0 0 25px rgba(0,200,150,0.4);
        border-color: #00C896;
    }

    /* ===== METRIC / CONFIDENCE CARD ===== */
    .metric-card {
        background: rgba(20, 28, 40, 0.8);
        border-radius: 12px;
        padding: 18px;
        border: 1px solid rgba(30,144,255,0.3);
        color: #E6EDF3;
        text-align: center;
        box-shadow: 0 0 8px rgba(30,144,255,0.1);
        transition: all 0.3s ease;
    }
    .metric-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 0 18px rgba(30,144,255,0.5);
        border-color: #00C896;
    }

    /* ===== INFO BOX (Detailed Info Tabs) ===== */
    .info-box {
        background: rgba(18, 25, 35, 0.9);
        border-left: 4px solid #00C896;
        padding: 18px;
        border-radius: 12px;
        color: #E6EDF3;
        margin: 10px 0;
        box-shadow: 0 0 12px rgba(0,200,150,0.2);
        transition: all 0.3s ease;
    }
    .info-box:hover {
        transform: scale(1.01);
        box-shadow: 0 0 20px rgba(0,200,150,0.4);
        border-left: 4px solid #1E90FF;
    }

    /* ===== WARNING BOX (Alternative Diagnoses) ===== */
    .warning-box {
        background: rgba(25, 30, 40, 0.85);
        border-left: 4px solid #F85149;
        padding: 18px;
        border-radius: 12px;
        margin: 10px 0;
        color: #E6EDF3;
        box-shadow: 0 0 10px rgba(248,81,73,0.15);
        transition: all 0.3s ease;
    }
    .warning-box:hover {
        transform: scale(1.03);
        box-shadow: 0 0 20px rgba(248,81,73,0.6);
        border-left: 4px solid #FF6F61;
    }

    /* ===== HEADINGS ===== */
    h1 {
        color: #00C896;
        text-align: center;
        font-weight: 700;
        text-shadow: 0 0 12px rgba(0,200,150,0.4);
    }
    h2 { color: #1E90FF; }
    h3 { color: #E3B341; }
    p, li { color: #C9D1D9; }

    /* ===== SIDEBAR ===== */
    section[data-testid="stSidebar"] {
        background-color: #0A0F16 !important;
        border-right: 1px solid rgba(255,255,255,0.05);
    }

    /* ===== FOOTER ===== */
    footer, .footer {
        color: #8B949E;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)




# ============================================================
# 📦 LOAD MODELS AND DATA
# ============================================================
@st.cache_resource
def load_models():
    """Load SVM and Random Forest models + metadata"""
    try:
        svm_model = joblib.load('saved_models/svm_model.pkl')
        rf_model = joblib.load('saved_models/ai_doctor_random_forest.pkl')
        metadata = joblib.load('saved_models/rf_model_metadata.pkl')  # same structure for both
        return {'svm': svm_model, 'rf': rf_model, 'metadata': metadata}
    except Exception as e:
        st.error(f"❌ Error loading models: {e}")
        return None


@st.cache_data
def load_disease_data():
    """Load all disease information CSVs"""
    try:
        return {
            'description': pd.read_csv('description.csv'),
            'medications': pd.read_csv('medications.csv'),
            'diets': pd.read_csv('diets.csv'),
            'precautions': pd.read_csv('precautions_df.csv'),
            'workouts': pd.read_csv('workout_df.csv')
        }
    except Exception as e:
        st.error(f"❌ Error loading disease data: {e}")
        return None


models = load_models()
disease_data = load_disease_data()
if models is None or disease_data is None:
    st.stop()

# ============================================================
# 🔍 HELPER FUNCTIONS
# ============================================================
def get_disease_details(disease_name):
    """Extract full details from CSVs"""
    disease_lower = disease_name.strip().lower()
    details = {}

    def extract_info(df):
        disease_col = [c for c in df.columns if 'disease' in c.lower()]
        if not disease_col:
            return []
        col = disease_col[0]
        row = df[df[col].astype(str).str.lower().str.strip() == disease_lower]
        if row.empty:
            return []
        return [str(x) for x in row.drop(columns=col).values.flatten()
                if str(x).strip() and str(x).lower() != 'nan']

    try:
        desc_col = [c for c in disease_data['description'].columns if 'disease' in c.lower()]
        if desc_col:
            desc_row = disease_data['description'][
                disease_data['description'][desc_col[0]].astype(str).str.lower().str.strip() == disease_lower
            ]
            details['description'] = (
                desc_row['Description'].iloc[0]
                if not desc_row.empty and 'Description' in desc_row.columns
                else "No description available."
            )
        else:
            details['description'] = "No description available."
    except:
        details['description'] = "No description available."

    details['medications'] = extract_info(disease_data['medications']) or ["Consult a doctor"]
    details['diet'] = extract_info(disease_data['diets']) or ["Balanced diet recommended"]
    details['precautions'] = extract_info(disease_data['precautions']) or ["General precautions"]
    details['workouts'] = extract_info(disease_data['workouts']) or ["Light exercise recommended"]

    return details


def build_feature_vector(symptom_severity_dict, feature_names):
    """Build numeric input vector"""
    x = np.zeros(len(feature_names))
    for i, feature in enumerate(feature_names):
        if feature in symptom_severity_dict:
            val = min(10.0, max(1.0, float(symptom_severity_dict[feature])))
            x[i] = val / 10.0
    return x.reshape(1, -1)


def model_predict(model, x, feature_names, severity_conf):
    """Predict using a model and return structured results"""
    probs = model.predict_proba(x)[0]
    top3_idx = np.argsort(probs)[-3:][::-1]
    results = []
    for idx in top3_idx:
        disease = models['metadata']['classes'][idx]
        model_conf = probs[idx]
        final_conf = 0.6 * model_conf + 0.4 * severity_conf
        results.append({
            'disease': disease,
            'model_confidence': round(model_conf * 100, 2),
            'severity_confidence': round(severity_conf * 100, 2),
            'final_confidence': round(final_conf * 100, 2),
            'details': get_disease_details(disease)
        })
    return results


def hybrid_predict(symptom_severity_dict):
    """Run both SVM and Random Forest, pick best"""
    feature_names = models['metadata']['feature_names']
    x = build_feature_vector(symptom_severity_dict, feature_names)

    # Severity confidence
    severity_conf = np.mean(list(symptom_severity_dict.values())) / 10.0 if symptom_severity_dict else 0

    # Predictions
    svm_preds = model_predict(models['svm'], x, feature_names, severity_conf)
    rf_preds = model_predict(models['rf'], x, feature_names, severity_conf)

    # Compare top confidences
    best_svm_conf = svm_preds[0]['final_confidence']
    best_rf_conf = rf_preds[0]['final_confidence']

    if best_rf_conf >= best_svm_conf:
        best_model = "Random Forest"
        return rf_preds, best_model
    else:
        best_model = "SVM"
        return svm_preds, best_model


# ============================================================
# 🎨 STREAMLIT UI
# ============================================================

st.markdown("<h1>🏥 AI Doctor - Hybrid Disease Prediction</h1>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align: center; color: gray; font-size: 18px;'>This system intelligently combines SVM & Random Forest for the most confident disease prediction.</p>",
    unsafe_allow_html=True)
st.markdown("---")

with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/doctor-male--v1.png", width=100)
    st.markdown("## 📋 About")
    st.info("""
    **AI Doctor Hybrid** combines SVM & Random Forest
    to pick the best prediction dynamically.

    **Model Accuracy (approx):**
    - SVM: 99.7%
    - Random Forest: 99.5%

    **Features:**
    - 86+ Symptoms
    - 41 Diseases
    - Smart Confidence Blending
    """)

    st.markdown("---")
    st.metric("Features", models['metadata']['n_features'])
    st.metric("Diseases", len(models['metadata']['classes']))

    st.markdown("---")
    st.warning("⚠️ For guidance only — always consult a doctor.")

col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("## 🔍 Enter Your Symptoms")

    available_symptoms = models['metadata']['feature_names']
    if 'selected_symptoms' not in st.session_state:
        st.session_state.selected_symptoms = {}

    num_symptoms = st.slider("How many symptoms do you have?", 1, 10, 3)
    selected_symptoms = {}

    for i in range(num_symptoms):
        col_sym, col_sev = st.columns([2, 1])
        with col_sym:
            symptom = st.selectbox(f"Symptom {i+1}", [''] + sorted(available_symptoms), key=f"sym_{i}")
        with col_sev:
            if symptom:
                sev = st.slider("Severity", 1, 10, 5, key=f"sev_{i}")
                selected_symptoms[symptom] = sev

    st.markdown("---")
    predict_button = st.button("🔬 Predict Disease", use_container_width=True)

with col2:
    st.markdown("## 📊 Prediction Results")

    if predict_button:
        if not selected_symptoms:
            st.error("⚠️ Please select at least one symptom.")
        else:
            with st.spinner("🧠 Evaluating SVM & Random Forest..."):
                predictions, model_used = hybrid_predict(selected_symptoms)

                st.markdown(f"### 🧩 Using **{model_used}** model for final prediction")

                st.markdown("### 🩺 Selected Symptoms:")
                for s, sev in selected_symptoms.items():
                    st.markdown(f"<div class='symptom-card'><b>{s.replace('_', ' ').title()}</b><br>Severity: {sev}/10</div>",
                                unsafe_allow_html=True)

                st.markdown("---")

                best = predictions[0]
                st.markdown(f"""
                <div class='prediction-box'>
                    <h2 style='color: #4CAF50; text-align: center;'>🎯 Primary Prediction</h2>
                    <h1 style='text-align: center; color: #1976D2;'>{best['disease']}</h1>
                    <p style='text-align: center; font-size: 20px; color: gray;'>
                        Confidence: <b style='color: #4CAF50;'>{best['final_confidence']}%</b>
                    </p>
                </div>
                """, unsafe_allow_html=True)

                st.markdown("### 📈 Confidence Breakdown")
                cols = st.columns(3)
                with cols[0]:
                    st.markdown(f"<div class='metric-card'><h4>Model</h4><h2>{best['model_confidence']}%</h2></div>", unsafe_allow_html=True)
                with cols[1]:
                    st.markdown(f"<div class='metric-card'><h4>Severity</h4><h2>{best['severity_confidence']}%</h2></div>", unsafe_allow_html=True)
                with cols[2]:
                    st.markdown(f"<div class='metric-card' style='border-color: #4CAF50;'><h4>Final</h4><h2 style='color: #4CAF50;'>{best['final_confidence']}%</h2></div>", unsafe_allow_html=True)

                fig = go.Figure(data=[go.Bar(
                    x=[p['disease'] for p in predictions],
                    y=[p['final_confidence'] for p in predictions],
                    marker_color=['#4CAF50', '#2196F3', '#FF9800'],
                    text=[f"{p['final_confidence']}%" for p in predictions],
                    textposition='auto'
                )])
                fig.update_layout(title="Top 3 Predictions", xaxis_title="Disease", yaxis_title="Confidence (%)", height=400)
                st.plotly_chart(fig, use_container_width=True)

# ============================================================
# 🩺 Detailed Info + Alternatives
# ============================================================
if predict_button and selected_symptoms:
    st.markdown("---")
    st.markdown("## 📚 Detailed Medical Information")
    best = predictions[0]
    details = best['details']

    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📖 Description", "💊 Medications", "🥗 Diet", "⚠️ Precautions", "💪 Workout"])

    with tab1:
        st.markdown(f"<div class='info-box'><h3>{best['disease']}</h3><p>{details['description']}</p></div>", unsafe_allow_html=True)
    with tab2:
        for i, m in enumerate(details['medications'], 1): st.markdown(f"{i}. {m}")
        st.warning("⚠️ Consult your doctor before taking medication.")
    with tab3:
        for i, d in enumerate(details['diet'], 1): st.markdown(f"{i}. {d}")
    with tab4:
        for i, p in enumerate(details['precautions'], 1): st.markdown(f"{i}. {p}")
    with tab5:
        for i, w in enumerate(details['workouts'], 1): st.markdown(f"{i}. {w}")

    st.markdown("---")
    st.markdown("## 🔄 Alternative Diagnoses")
    alt_cols = st.columns(2)
    for i, alt in enumerate(predictions[1:], 1):
        with alt_cols[i - 1]:
            st.markdown(f"<div class='warning-box'><h4>{i + 1}. {alt['disease']}</h4><p>Confidence: <b>{alt['final_confidence']}%</b></p></div>", unsafe_allow_html=True)

# ============================================================
# Footer
# ============================================================
st.markdown("---")
st.markdown("""
<div style='text-align:center; color:gray; padding:20px;'>
    <p>🏥 <b>AI Doctor - Hybrid Model</b></p>
    <p>Made with ❤️ using Streamlit | © 2025</p>
    <p style='font-size:12px;'>Informational use only — not a substitute for professional medical advice.</p>
</div>
""", unsafe_allow_html=True)
