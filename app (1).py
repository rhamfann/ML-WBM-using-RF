import streamlit as st
import pandas as pd
import pickle
import gzip
import numpy as np

# Set judul aplikasi
st.set_page_config(page_title='WBM Rheology Predictor', layout='wide')

st.title('Mud Properties Prediction Dashboard (WBM)')
st.markdown('Aplikasi ini memprediksi PV, YP, dan Gels menggunakan model Random Forest.')

# Load model yang sudah dikompresi (.gz)
@st.cache_resource
def load_model():
    with gzip.open('best_rf_model.pkl.gz', 'rb') as f:
        return pickle.load(f)

try:
    model = load_model()
except FileNotFoundError:
    st.error('File model (best_rf_model.pkl.gz) tidak ditemukan.')
    st.stop()

# Sidebar untuk Input
st.sidebar.header('Input Parameter Lumpur:')

def user_input_features():
    mw = st.sidebar.slider('MW, In (SG)', 1.0, 2.5, 1.2)
    temp = st.sidebar.slider('Temp, In (°F)', 60.0, 200.0, 100.0)
    fv = st.sidebar.slider('FV, In (sec)', 30.0, 100.0, 50.0)
    solids = st.sidebar.slider('Solids (%)', 0.0, 30.0, 10.0)
    ph = st.sidebar.slider('pH', 7.0, 14.0, 9.0)
    bent = st.sidebar.slider('Bent. (Ppb)', 0.0, 50.0, 20.0)

    data = {
        'MW, In (SG)': mw,
        'Temp, In (°F)': temp,
        'FV, In (sec)': fv,
        'Solids (%)': solids,
        'pH': ph,
        'Bent. (Ppb)': bent
    }
    return pd.DataFrame(data, index=[0])

input_df = user_input_features()

# Menampilkan Input
st.subheader('Parameter Input')
st.write(input_df)

# Prediksi
if st.button('Predict'):
    prediction = model.predict(input_df)[0]

    col1, col2, col3, col4 = st.columns(4)
    targets = ['PV (cps)', 'YP (lb/100ft²)', "Gels 10'", "Gels 10''"]

    with col1:
        st.metric(targets[0], f"{prediction[0]:.2f}")
    with col2:
        st.metric(targets[1], f"{prediction[1]:.2f}")
    with col3:
        st.metric(targets[2], f"{prediction[2]:.2f}")
    with col4:
        st.metric(targets[3], f"{prediction[3]:.2f}")

st.info('Catatan: Pastikan model dan app.py berada dalam satu folder.')
