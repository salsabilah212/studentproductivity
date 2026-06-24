import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

@st.cache_data
def load_data():
    path = os.path.join(BASE_DIR, 'data', 'Student_Productivity_Dataset.csv')
    df   = pd.read_csv(path)
    df.drop(columns=['Student_ID', 'Performance_Category'], errors='ignore', inplace=True)
    return df
#beberapa kolom dihapus karena tidak dibutuhkan

def show():
    st.title("📊 Data Exploration")
    st.divider()

    df = load_data()

    # Dataset Overview 
    st.subheader("Dataset Overview")
    st.markdown("This section explores the student productivity dataset used for model training.")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Records",  f"{len(df):,}",           "After Cleaning")
    col2.metric("Features",       df.shape[1] - 1,          "Input Variables")
    col3.metric("Target",         "Continuous",              "Productivity Score")
    col4.metric("Missing Values", df.isna().sum().sum(),     "After Imputation")

    st.divider()

    # Preprocessing
    st.subheader("🔧 Data Preprocessing")
    tab1, tab2, tab3, tab4 = st.tabs(["Missing Values", "Outliers", "Encoding", "Scaling"])

    with tab1:
        st.markdown("**Missing Values Handling**")
        st.markdown("""
        - Identified columns with missing values
        - Used **mode imputation** for categorical features
        - Applied **median imputation** for numerical features in preprocessing pipeline
        """)
        st.markdown("Columns handled:")
        st.markdown("- `Internet_Quality`, `Part_Time_Job`, `Gender`")

    with tab2:
        st.markdown("**Outliers and Extreme Values Handling**")
        st.markdown("""
        - Detected outliers with **IQR (Tukey)** methode
        - Threshold: Q1 - 1.5×IQR and Q3 + 1.5×IQR
        - Rows containing outliers were removed from the dataset
        """)

    with tab3:
        st.markdown("**Feature Encoding**")
        st.markdown("""
        - `Internet_Quality` → **OrdinalEncoder** (Poor=0, Average=1, Good=2)
        - `Part_Time_Job` → **OrdinalEncoder** (No=0, Yes=1)
        - `Gender` → **OneHotEncoder** (drop first)
        """)

    with tab4:
        st.markdown("**Feature Scaling**")
        st.markdown("""
        - All numerical features are scaled using **StandardScaler**
        """)

    st.divider()

    # Target Distribution
    st.subheader("🎯 Productivity Score Distribution")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].hist(df['Productivity_Score'].dropna(), bins=40,
                 color='steelblue', edgecolor='white', alpha=0.85)
    axes[0].axvline(df['Productivity_Score'].mean(), color='red',
                    linestyle='--', linewidth=1.5,
                    label=f"Mean: {df['Productivity_Score'].mean():.1f}")
    axes[0].axvline(df['Productivity_Score'].median(), color='orange',
                    linestyle='--', linewidth=1.5,
                    label=f"Median: {df['Productivity_Score'].median():.1f}")
    axes[0].set_title('Histogram Productivity Score')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].boxplot(df['Productivity_Score'].dropna(), patch_artist=True,
                    boxprops=dict(facecolor='steelblue', alpha=0.7))
    axes[1].set_title('Boxplot Productivity Score')
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.divider()

    # Correlation
    st.subheader("📈 Correlation Between Features and Productivity Score")
    num_cols    = df.select_dtypes(include='number').columns.tolist()
    corr_target = df[num_cols].corr()['Productivity_Score'].drop('Productivity_Score').sort_values()
    colors_bar  = ['salmon' if v < 0 else 'cornflowerblue' for v in corr_target]

    fig, ax = plt.subplots(figsize=(9, 6))
    bars = ax.barh(corr_target.index, corr_target.values, color=colors_bar, alpha=0.85)
    ax.axvline(0, color='darkblue', linewidth=0.8)
    ax.bar_label(bars, fmt='%.3f', padding=4, fontsize=9)
    ax.set_title('Features vs Productivity Score', fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()