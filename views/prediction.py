import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import pandas as pd
from utils.model_utils import transform_input, load_ann

def predict_ann(X_transformed):
    ann   = load_ann()
    score = float(ann.predict(X_transformed, verbose=0).flatten()[0])
    return round(score, 2)

def get_category(score):
    if score >= 70:
        return 'HIGH', '🟢', '#28a745', 'linear-gradient(135deg, #1a7a4a, #28a745)'
    elif score >= 45:
        return 'MEDIUM', '🟡', '#ffc107', 'linear-gradient(135deg, #856404, #ffc107)'
    else:
        return 'LOW', '🔴', '#dc3545', 'linear-gradient(135deg, #7a1a1a, #dc3545)'

def card(title, color="#1e2a3a", border="#357abd"):
    return f"""
    <div style='background:{color}; padding:18px 20px; border-radius:12px;
                border-left:4px solid {border}; margin-bottom:12px;'>
        <p style='color:#adb5bd; font-size:0.78rem; margin:0 0 4px 0; text-transform:uppercase; letter-spacing:1px;'>{title}</p>
    """

def show():
    # Header
    st.markdown("""
    <div style='background:linear-gradient(135deg, #0d1b2a, #1f3c88);
                padding:28px 32px; border-radius:16px; margin-bottom:24px;
                border: 1px solid #2a4a8a;'>
        <h1 style='color:white; margin:0; font-size:1.9rem; font-weight:700;'>
            🔮 Student Productivity Predictor
        </h1>
        <p style='color:#8fb3e8; margin:8px 0 0 0; font-size:0.95rem;'>
            Powered by Artificial Neural Network (TensorFlow/Keras) · Mini Project 2
        </p>
    </div>
    """, unsafe_allow_html=True)

    # SECTION 1 — Basic Information
    st.markdown("""
    <div style='display:flex; align-items:center; gap:10px; margin-bottom:16px;'>
        <div style='background:#1f3c88; width:4px; height:28px; border-radius:2px;'></div>
        <h3 style='color:white; margin:0;'>Basic Information</h3>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        age = st.number_input("Age", min_value=15, max_value=30, value=20)
    with col2:
        gender = st.selectbox("Gender", ["Male", "Female"])
    with col3:
        internet_quality = st.selectbox("Internet Quality", ["Poor", "Average", "Good"])
    with col4:
        part_time_job = st.selectbox("Part Time Job", ["No", "Yes"])
    with col5:
        extracurricular = st.selectbox("Extracurricular",
                                        [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")

    st.markdown("<br>", unsafe_allow_html=True)

    # SECTION 2 — Academic Performance
    st.markdown("""
    <div style='display:flex; align-items:center; gap:10px; margin-bottom:16px;'>
        <div style='background:#28a745; width:4px; height:28px; border-radius:2px;'></div>
        <h3 style='color:white; margin:0;'>Academic Performance</h3>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        study_hours   = st.slider("Study Hours / Day",        0.0, 12.0, 6.0, 0.5)
        attendance    = st.slider("Attendance (%)",           0, 100, 80, 1)        # ← integer
    with col2:
        assignments   = st.slider("Assignments Completed (%)", 0, 100, 75, 1)       # ← max 100, integer
        participation = st.slider("Class Participation",      0.0, 10.0, 7.0, 0.1)
    with col3:
        prev_gpa      = st.slider("Previous GPA",             0.0, 4.0, 3.0, 0.1)
        ai_usage      = st.slider("AI Tool Usage (hrs/week)", 0.0, 10.0, 2.0, 0.5)
    st.markdown("<br>", unsafe_allow_html=True)

    # SECTION 3 — Lifestyle & Wellbeing
    st.markdown("""
    <div style='display:flex; align-items:center; gap:10px; margin-bottom:16px;'>
        <div style='background:#e83e8c; width:4px; height:28px; border-radius:2px;'></div>
        <h3 style='color:white; margin:0;'>Lifestyle & Wellbeing</h3>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        sleep_hours  = st.slider("Sleep Hours / Night",       4.0, 12.0, 7.0, 0.5)
        stress_level = st.slider("Stress Level (1–10)",       1, 10, 5)
    with col2:
        motivation   = st.slider("Motivation Level (1–10)",   1, 10, 7)
        screen_time  = st.slider("Screen Time (hrs/day)",     0.0, 12.0, 3.0, 0.5)
    with col3:
        social_media = st.slider("Social Media (hrs/day)",    0.0, 12.0, 2.0, 0.5)
        physical_act = st.slider("Physical Activity (hrs/week)", 0.0, 20.0, 5.0, 0.5)

    st.markdown("<br>", unsafe_allow_html=True)

    # TOMBOL PREDIKSI
    col_btn = st.columns([1, 2, 1])
    with col_btn[1]:
        predict_btn = st.button("🔮 Predict Productivity Score",
                                use_container_width=True, type="primary")

    # HASIL PREDIKSI
    if predict_btn:
        input_dict = {
            "Age":                              age,
            "Gender":                           gender,
            "Study_Hours_Per_Day":              study_hours,
            "Sleep_Hours_Per_Night":            sleep_hours,
            "Screen_Time_Hours":                screen_time,
            "Social_Media_Hours":               social_media,
            "Attendance_Percentage":            attendance,
            "Assignments_Completed":            assignments,
            "Class_Participation_Score":        participation,
            "Physical_Activity_Hours_Per_Week": physical_act,
            "Stress_Level":                     stress_level,
            "Motivation_Level":                 motivation,
            "Internet_Quality":                 internet_quality,
            "Part_Time_Job":                    part_time_job,
            "Extracurricular_Involvement":      extracurricular,
            "AI_Tool_Usage_Hours_Per_Week":     ai_usage,
            "Previous_Semester_GPA":            prev_gpa,
        }

        with st.spinner("🧠 ANN sedang memproses data..."):
            X     = transform_input(input_dict)
            score = predict_ann(X)

        category, icon, color_hex, color_bg = get_category(score)

        st.divider()

        # Score Card
        col_score, col_gauge = st.columns([1, 1])

        with col_score:
            st.markdown(f"""
            <div style='background:{color_bg}; padding:35px; border-radius:16px;
                        text-align:center; box-shadow: 0 4px 20px rgba(0,0,0,0.3);'>
                <p style='color:rgba(255,255,255,0.8); font-size:0.9rem;
                           letter-spacing:2px; text-transform:uppercase; margin:0;'>
                    ANN Prediction Result
                </p>
                <h1 style='color:white; font-size:5rem; margin:10px 0; font-weight:800;
                            text-shadow: 0 2px 10px rgba(0,0,0,0.3);'>
                    {score}
                </h1>
                <p style='color:rgba(255,255,255,0.8); font-size:0.9rem; margin:0;'>
                    out of 100
                </p>
                <div style='background:rgba(255,255,255,0.2); height:2px;
                             margin:15px 0; border-radius:1px;'></div>
                <h3 style='color:white; margin:0; font-size:1.3rem;'>
                    {icon} Productivity: {category}
                </h3>
            </div>
            """, unsafe_allow_html=True)

        with col_gauge:
            # Mini bar chart input penting
            st.markdown(f"""
            <div style='background:#1e2a3a; padding:20px; border-radius:16px; height:100%;'>
                <p style='color:#8fb3e8; font-size:0.8rem; text-transform:uppercase;
                           letter-spacing:1px; margin:0 0 15px 0;'>Score Indicator</p>
            """, unsafe_allow_html=True)

            st.progress(min(int(score), 100) / 100,
                        text=f"Productivity Score: {score}/100")

            # Mini metrics
            m1, m2 = st.columns(2)
            m1.metric("Study Hours",  f"{study_hours}h/day",
                       delta="✅" if study_hours >= 3 else "⚠️ Kurang")
            m2.metric("Sleep Hours",  f"{sleep_hours}h/night",
                       delta="✅" if sleep_hours >= 7 else "⚠️ Kurang")
            m3, m4 = st.columns(2)
            m3.metric("Attendance",   f"{attendance}%",
                       delta="✅" if attendance >= 75 else "⚠️ Kurang")
            m4.metric("Stress Level", f"{stress_level}/10",
                       delta="✅" if stress_level <= 7 else "⚠️ Tinggi")

            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)