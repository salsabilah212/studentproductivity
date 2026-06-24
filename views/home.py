import streamlit as st

def show():
    # ── Hero ──────────────────────────────────────────────────────────────────
    st.markdown("""
    <div style='background: linear-gradient(135deg, #2d1b69, #6a0dad);
                padding: 40px; border-radius: 16px; text-align: center;
                margin-bottom: 30px;'>
        <h1 style='color: white; font-size: 2.2rem; margin: 0 0 10px 0;'>
            👩🏻‍🏫 Student Productivity Prediction System
        </h1>
        <p style='color: #e0b3ff; font-size: 1rem; margin: 0;'>
            Predicting student productivity using Artificial Neural Network
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # ── Background ────────────────────────────────────────────────────────────
    st.subheader("📌 Background")
    st.markdown("""
    Student productivity is a critical indicator of academic success. External
    factors such as social media usage, internet quality, part-time jobs, sleep
    hours, and stress levels significantly affect how productive a student can be.

    The **Student Productivity Dataset** contains behavioral and study habit
    attributes from approximately **10,000 students**, covering study hours,
    sleep duration, motivation level, attendance, and AI tool usage.
    """)

    st.divider()

    # ── Objectives & Problem Statement ───────────────────────────────────────
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🎯 Project Objectives")
        st.markdown("""
        This project is part of **Mini Project 2**, focused on building an
        **Artificial Neural Network (ANN)** using TensorFlow/Keras to predict
        student Productivity Scores. The key objectives are:

        1. Build an end-to-end ANN model and understand how it works
        2. Explain architectural decisions — number of layers, neurons,
           activation functions, and regularization techniques
        3. Compare ANN performance against the best ML model from
           Mini Project 1 (XGBoost) using R², MAE, and RMSE
        4. Analyze why one model outperforms the other, both
           quantitatively and qualitatively
        5. Provide an interactive prediction interface for real-time
           student productivity estimation
        """)

    with col2:
        st.subheader("❓ Problem Statement")
        st.markdown("""
        Educational institutions struggle to identify the dominant factors
        affecting student productivity. Without an accurate predictive model,
        it is difficult to provide data-driven recommendations for both
        students and educators.
        """)
        
    st.divider()

    # ── Page Guide 
    st.subheader("🗺️ Page Guide")

    a, b = st.columns(2)
    with a:
        st.info("""
        **📊 Data Exploration**

        Explore the Student Productivity dataset end-to-end. Covers dataset
        overview, preprocessing steps (missing values, outliers, encoding,
        scaling), target distribution, feature correlations, and productivity
        breakdown by categorical variables such as Part-Time Job,
        Internet Quality, and Gender.
        """)

        st.info("""
        **🧠 Model Structure**

        Deep dive into the ANN architecture — layer configuration, neuron
        counts, activation functions, dropout rates, and the reasoning behind
        each design decision. Also covers training configuration (optimizer,
        learning rate, loss function, callbacks) and the live model summary
        loaded directly from the saved .keras file.
        """)

    with b:
        st.info("""
        **📈 ANN vs ML Comparison**

        Side-by-side comparison of all models used across Mini Project 1 and 2:
        Linear Regression, Ridge, XGBoost, and ANN. Includes R², MAE, and RMSE
        metrics, visual charts, feature importance analysis, and an in-depth
        qualitative explanation of why one model performs better than another.
        """)

        st.info("""
        **🔮 Make Prediction**

        The main interactive page where users input student data manually
        via a structured form across three sections: Basic Information,
        Academic Performance, and Lifestyle & Wellbeing. After prediction,
        the app displays the ANN Productivity Score, performance category,
        and personalized recommendations based on the input values.
        """)