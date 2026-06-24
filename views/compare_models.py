import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils.model_utils import get_metrics_dict, get_feature_importance

def show():
    st.title("📊 Artificial Neural Network vs XGBoost Machine Learning")
    st.divider()

    st.subheader("Comparing ANN with XGBoost (Mini Project 1)")
    st.markdown("""
    This section provides a detailed analysis of the models used in this project:
    - **ANN (Artificial Neural Network)** - Mini Project 2 (current)
    - **XGBoost (Machine Learning Model)** - Mini Project 1
    """)
    st.divider()

    metrics = get_metrics_dict()

    # Performance Metrics (Evauation)
    st.subheader("📊 Performance Metrics Comparison")

    col1, col2 = st.columns([2, 1])

    with col1:
        models  = list(metrics.keys())
        colors  = ['steelblue', 'seagreen', 'darkorange', 'mediumpurple']
        metrics_keys = ['R2', 'MAE', 'RMSE']
        titles       = ['R² Score', 'MAE', 'RMSE']

        fig, axes = plt.subplots(1, 3, figsize=(14, 5))
        for ax, key, title in zip(axes, metrics_keys, titles):
            vals = [metrics[m][key] for m in models]
            bars = ax.bar(models, vals, color=colors, alpha=0.85, width=0.5)
            ax.bar_label(bars, fmt='%.4f', padding=3, fontsize=8)
            ax.set_title(title, fontweight='bold')
            ax.set_xticklabels(models, rotation=20, ha='right', fontsize=8)
            ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col2:
        comparison_df = pd.DataFrame({
            'Metric': ['R² Score', 'MAE', 'RMSE'],
            'XGBoost': [
                metrics['XGBoost']['R2'],
                metrics['XGBoost']['MAE'],
                metrics['XGBoost']['RMSE'],
            ],
            'ANN': [
                metrics['ANN']['R2'],
                metrics['ANN']['MAE'],
                metrics['ANN']['RMSE'],
            ],
        })
        st.dataframe(comparison_df, use_container_width=True, hide_index=True)

    st.divider()

    # Feature Importance 
    st.subheader("🌲 Feature Importance — XGBoost Top 15")
    try:
        imp_df = get_feature_importance().head(15).sort_values('Importance')
        fig, ax = plt.subplots(figsize=(9, 6))
        ax.barh(imp_df['Feature'], imp_df['Importance'], color='steelblue', alpha=0.85)
        ax.set_title('Top 15 Feature Importance', fontweight='bold')
        ax.set_xlabel('Importance Score')
        ax.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    except Exception as e:
        st.warning(f"Error: {e}")

    st.divider()

    # Insight
    st.subheader("💡 Insight")
    st.markdown("""
    - **XGBoost** sedikit lebih unggul dengan R²=0.7637 vs ANN R²=0.7615
    - XGBoost terbukti menjadi model yang paling efisien untuk dataset ukuran sedang (~10.000 baris). 
    - XGBoost didesain secara spesifik untuk menangani data terstruktur (tabular). Melalui algoritma Gradient Boosting, model ini mampu memetakan hubungan antar-fitur secara iteratif dengan memperbaiki kesalahan prediksi dari tahapan sebelumnya secara efisien.
    - Artificial Neural Network (ANN) merupakan model deep learning yang dirancang untuk mengeksplorasi representasi data berdimensi tinggi dan kompleks. Ketika diterapkan pada data terstruktur ukuran sedang, kapasitas parameter ANN yang besar cenderung berlebihan (overkill), sehingga risiko terjadinya overfitting menjadi lebih tinggi.
    """)