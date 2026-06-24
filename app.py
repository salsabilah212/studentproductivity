import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL']  = '3'

import sys
import streamlit as st
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from views import ann, compare_models, data_exploration, home, prediction

st.set_page_config(
    page_title = "Student Productivity Prediction System",
    page_icon  = "🎓",
    layout     = "wide"
)

# Custom Sidebar
st.markdown("""
<style>
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #451854 0%, #1a2f4a 100%);
}
[data-testid="stSidebar"] * {
    color: white !important;
}
div[role="radiogroup"] > label {
    background: transparent;
    padding: 8px 12px;
    border-radius: 8px;
    margin: 3px 0;
    cursor: pointer;
    transition: all 0.2s;
    display: flex;
    align-items: center;
    gap: 8px;
}
div[role="radiogroup"] > label:hover {
    background: rgba(53, 122, 189, 0.3) !important;
}
div[role="radiogroup"] > label[data-checked="true"] {
    background: rgba(53, 122, 189, 0.5) !important;
    border-left: 3px solid #357abd;
}
</style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    # Logo
    st.markdown("""
    <div style='text-align:center; padding:20px 10px 10px 10px;'>
        <div style='font-size:3rem;'>🎓</div>
        <h2 style='color:white; margin:8px 0 4px 0; font-size:1.1rem; font-weight:700;'>
            Student Academy
        </h2>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<hr style='border-color:#2a4a8a; margin:10px 0;'>", unsafe_allow_html=True)

    # Menu navigasi
    st.markdown("<p style='color:#8fb3e8; font-size:0.75rem; letter-spacing:2px; text-transform:uppercase; margin:8px 0;'>Menu</p>", unsafe_allow_html=True)

    page = st.radio("", [
        "🏠  Home",
        "📊  Data Exploration",
        "🧠  Model Structure",
        "📈  ANN vs ML Comparison",
        "🔮  Make Prediction",
    ], label_visibility="collapsed")

    st.markdown("<hr style='border-color:#2a4a8a; margin:10px 0;'>", unsafe_allow_html=True)

    # Info card bawah
    st.markdown("""
    <div style='background:linear-gradient(135deg, #b915c2, #f051ea);
                padding:15px; border-radius:12px; text-align:center;'>
        <p style='color:white; font-size:0.8rem; font-weight:700; margin:0 0 4px 0;'>
            Mini Project <br> Artificial Neural Network Model
        </p>
        <p style='color:#fac5fc; font-size:0.72rem; margin:0; line-height:1.4;'>
            Created by Salsabilah
        </p>
    </div>
    """, unsafe_allow_html=True)

# Routing
if   "Home"                in page: home.show()
elif "Data Exploration"    in page: data_exploration.show()
elif "Model Structure"          in page: ann.show()
elif "ANN vs ML Comparison"           in page: compare_models.show()
elif "Prediction"     in page: prediction.show()