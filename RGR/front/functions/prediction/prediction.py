from functions.prediction.models import models
import streamlit as st
import pandas as pd
from functions.prediction.utils.neuro_result import neuro_result


def prediction():
    model_option = st.selectbox('Выберите модель для предсказания', ('KNN', 'Bagging', 'Нейронная сеть'))
    uploaded_file = st.file_uploader("Выберите файл .csv (без заголовков столбцов):")

    if uploaded_file:
        upload_data = pd.read_csv(uploaded_file, sep=";", encoding='utf-8')
        upload_data.columns = ["est_diameter_min", "est_diameter_max", "relative_velocity", "miss_distance", "absolute_magnitude"]

    if st.button("Получить предсказание"):
        if not uploaded_file:
            st.warning('Сначала нужно загрузить данные!', icon="⚠️")
        else:
            y_pred = models[model_option].predict(upload_data)
            if model_option == "Нейронная сеть":
                y_pred = neuro_result(y_pred)
            upload_data['hazardous'] = y_pred
            upload_data.to_csv("geg.csv")
            st.write(upload_data)
            st.download_button("Сохранить результат", upload_data.to_csv(index=False, encoding='utf-8'), "file.csv", "text/csv", key='download-csv')
