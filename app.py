import streamlit as st
import pandas as pd
import numpy as np
import pickle
import math

# Load model
with open("models/social_media_addected.sav", "rb") as file:
    model = pickle.load(file)

addicted_score = [
    {"score": 2, "label": "Tidak Sama Sekali", "description": "Nggak pernah buka media sosial atau bahkan nggak punya akun sama sekali."},
    {"score": 3, "label": "Sangat Rendah", "description": "Punya akun, tapi jarang banget buka. Mungkin cuma sekali seminggu atau pas lagi iseng aja."},
    {"score": 4, "label": "Rendah", "description": "Kadang-kadang buka, paling cuma beberapa menit sehari. Nggak terlalu tertarik."},
    {"score": 5, "label": "Cukup Rendah", "description": "Aktif, tapi masih wajar. Biasanya buka sekitar 30 menit sehari dan nggak terlalu ngaruh ke aktivitas harian."},
    {"score": 6, "label": "Sedang", "description": "Rutin buka tiap hari, sekitar 1–2 jam. Kadang mulai susah berhenti, tapi masih bisa dikontrol."},
    {"score": 7, "label": "Cukup Tinggi", "description": "Sering scroll tanpa sadar. Udah mulai ganggu kegiatan lain seperti kerja, belajar, atau tidur."},
    {"score": 8, "label": "Tinggi", "description": "Hampir selalu online. Susah berhenti bahkan pas lagi ngobrol atau kerja. 3–5 jam sehari di medsos."},
    {"score": 9, "label": "Sangat Tinggi", "description": "Kecanduan parah. Bangun tidur langsung buka sosmed, tidur pun jadi susah karena nggak berhenti scroll. Bisa lebih dari 5 jam sehari."}
]


# Title
st.title("Prediksi Social Media Addiction Score")

# Input user
age = st.number_input("Age", min_value=10, max_value=100, value=20)
gender = st.selectbox("Gender", ["Male", "Female"])
usage_hours = st.number_input("Average Daily Usage (Hours)", min_value=0.0, max_value=24.0, value=3.0)
platform = st.selectbox("Most Used Platform", ["Instagram", "Twitter", "TikTok", "YouTube", "Facebook", "LinkedIn"])
sleep_hours = st.number_input("Sleep Hours Per Night", min_value=0.0, max_value=24.0, value=7.0)

# Prediksi
if st.button("Prediksi"):
    input_df = pd.DataFrame([{
        "Age": age,
        "Gender": gender,
        "Avg_Daily_Usage_Hours": usage_hours,
        "Most_Used_Platform": platform,
        "Sleep_Hours_Per_Night": sleep_hours
    }])

    # Prediksi dengan transformasi balik dari log
    pred_log = model.predict(input_df)[0]
    pred = np.expm1(pred_log)  # Transformasi balik dari log
    
    # Pembulatan dan batasan
    pred_rounded = round(pred)


    # Cari info berdasarkan skor
    score_info = next((item for item in addicted_score if item["score"] == pred_rounded), None)

    if score_info:
        message = f"""
    **Kecanduan:** **{score_info['label']} ({pred_rounded})**  
    **Keterangan:** {score_info['description']}
    """

        if pred_rounded == 6:
            st.success(message)
        elif pred_rounded in [7, 8]:
            st.error(message)  # st.danger aliasnya st.error
        elif pred_rounded >= 9:
            st.warning("Skor terlalu tinggi, perlu perhatian lebih!")
            st.error(message)
        else:
            st.info(message)
    else:
        st.warning("Skor di luar jangkauan yang ditentukan.")
