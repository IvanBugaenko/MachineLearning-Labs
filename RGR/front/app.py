import streamlit as st
import pandas as pd
import numpy as np
from streamlit_option_menu import option_menu
from info import INFO
from models import pipe_bagging, pipe_dence, pipe_knn


df = pd.DataFrame([{
    "est_diameter_min": "",
    "est_diameter_max": "", 
    "relative_velocity": "", 
    "miss_distance": "", 
    "absolute_magnitude": ""
}])


with st.sidebar:
    selected = option_menu("Меню", ["Описание данных", 'Визуализации', 'Предсказание'], 
        icons=['info-circle', 'bar-chart', 'tag'], menu_icon="cast", default_index=0)
    
    if selected == "Предсказание":
        model_option = st.selectbox('Выберите модель для предсказания', ('KNN', 'Bagging', 'Нейронная сеть'))


if selected == "Описание данных":
    st.markdown(INFO)


if selected == "Визуализация данных":
    ...


if selected == "Предсказание":
    uploaded_file = st.file_uploader("Выберите файл:")
    edited_df = st.experimental_data_editor(df, num_rows="dynamic")

    if uploaded_file:
        upladdata = pd.read_csv(uploaded_file, sep=";", encoding='utf-8')
        upladdata.columns = ["est_diameter_min", "est_diameter_max", "relative_velocity", "miss_distance", "absolute_magnitude"]
        print(pd.concat([edited_df, upladdata]))

    if st.button("Получить предсказание"):
        print(np.array(edited_df, dtype=np.float32))
        st.write("Ожидайте")
