import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import pandas as pd
from utils.model_utils import load_ann, load_xgboost, load_preprocessor

def show():
    st.title("🧠 Artificial Neural Network")
    st.divider()

    st.subheader("Understanding the Artificial Neural Network")
    st.markdown("""
    This section explains the architecture, design decisions, and performance
    of the ANN model built specifically for student productivity prediction.
    """)
    st.divider()

    # Network Architecture
    st.subheader("🏗️ Network Architecture")
    st.markdown("**Layer Configuration**")

    col1, col2 = st.columns([1.5, 1])

    with col1:
        arch_df = pd.DataFrame({
            'Layer':      ['Input', 'Hidden 1', 'Hidden 2', 'Hidden 3', 'Output'],
            'Type':       ['Input', 'Dense', 'Dense', 'Dense', 'Dense'],
            'Units':      ['14', '128', '64', '32', '1'],
            'Activation': ['-', 'ReLU', 'ReLU', 'ReLU', 'Linear'],
            'Dropout':    ['-', '0.2', '0.1', '-', '-'],
        })
        st.dataframe(arch_df, use_container_width=True, hide_index=True)
        st.markdown("""
                    - Menggunakan 3 hidden Layer karena dataset ini termasuk data yang kompleks dengan memiliki 17 fitur dan 10.000 baris. 
                    - Dense berjumlah 128 → 64 → 32 agar model dapat menangkap representasi fitur yang luas dan beragam.
                    - Drop out digunakan untuk mencegah overfitting
                    - BatchNormalization digunakan untuk menstabilkan distribusi aktivasi antar layer sehingga training lebih stabil dan konvergen lebih cepat.
                    """)

    with col2:
        st.metric("Hidden Layers",   3)
        st.metric("Total Layers",    5)

    st.divider()

    # Training Config
    st.subheader("⚙️ Training Configuration")
    config_df = pd.DataFrame({
        'Parameter': ['Optimizer', 'Learning Rate', 'Loss Function',
                      'Metrics', 'Batch Size', 'Max Epochs', 'Early Stopping Patience'],
        'Nilai':     ['Adam', '0.0001', 'MSE',
                      'MAE', '32', '300', '20'],
    })
    st.dataframe(config_df, use_container_width=True, hide_index=True)

    st.divider()

    # Model Summary
    st.subheader("📋 Model Summary")
    try:
        ann          = load_ann()
        summary_list = []
        ann.summary(print_fn=lambda x: summary_list.append(x))
        st.code('\n'.join(summary_list), language='text')
    except Exception as e:
        st.warning(f"Tidak dapat load ANN: {e}")