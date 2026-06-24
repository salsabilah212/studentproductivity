import os
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
import streamlit as st

BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, 'models')

#Load Model and Prepocessor
@st.cache_resource
def load_preprocessor():
    return joblib.load(os.path.join(MODEL_DIR, 'preprocessor.pkl'))

@st.cache_resource
def load_xgboost():
    return joblib.load(os.path.join(MODEL_DIR, 'xgboost.pkl'))

@st.cache_resource
def load_ann():
    return tf.keras.models.load_model(
        os.path.join(MODEL_DIR, 'ann_best_model.keras')
    )

# Transform
def transform_input(input_dict: dict) -> pd.DataFrame:
    preprocessor  = load_preprocessor()
    input_df      = pd.DataFrame([input_dict])
    X_transformed = preprocessor.transform(input_df)
    feature_names = preprocessor.get_feature_names_out()
    return pd.DataFrame(X_transformed, columns=feature_names)

# Predict Model
def predict_all(X_transformed: pd.DataFrame) -> dict:
    xgb      = load_xgboost()
    ann      = load_ann()
    pred_xgb = float(xgb.predict(X_transformed)[0])
    pred_ann = float(ann.predict(X_transformed, verbose=0).flatten()[0])
    return {
        'XGBoost': round(pred_xgb, 2),
        'ANN':     round(pred_ann, 2),
        'Average': round((pred_xgb + pred_ann) / 2, 2)
    }

# Productivity Category
# untuk kategori ini saya ambil dari buku "Educational Measurement" karya Robert L. Linn & Norman E. Gronlund yang mengelompokkan performa/aktivitas siswa kedalam 3 kategori.
def get_category(score: float) -> tuple:
    if score >= 70:
        return 'HIGH', '✅', 'success'
    elif score >= 45:
        return 'MEDIUM', '⚠️', 'warning'
    else:
        return 'LOW', '❌', 'error'
    
# Evaluation Metrics (Test Set)
def get_metrics_dict() -> dict:
    return {
        'Linear Regression': {'R2': 0.6700, 'MAE': 5.3291, 'RMSE': 6.6933},
        'Ridge':             {'R2': 0.6700, 'MAE': 5.3294, 'RMSE': 6.6936},
        'XGBoost':           {'R2': 0.7637, 'MAE': 4.4833, 'RMSE': 5.6639},
        'ANN':               {'R2': 0.7616, 'MAE': 4.5180, 'RMSE': 5.7160},
    }
    
# Feature importance XGBoost 
def get_feature_importance() -> pd.DataFrame:
    xgb           = load_xgboost()
    preprocessor  = load_preprocessor()
    feature_names = preprocessor.get_feature_names_out()
    return pd.DataFrame({
        'Feature':    feature_names,
        'Importance': xgb.feature_importances_
    }).sort_values('Importance', ascending=False).reset_index(drop=True)
    
@st.cache_resource
def load_xgboost():
    return joblib.load(os.path.join(MODEL_DIR, 'xgboost.pkl')) 