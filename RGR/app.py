import streamlit as st
from streamlit_autorefresh import st_autorefresh
from streamlit_option_menu import option_menu
from RGR.info import INFO

st_autorefresh(interval=1000, key="dataframerefresh")


with st.sidebar:
    selected = option_menu("Меню", ["Описание данных", 'Визуализация данных', 'Предсказание'], 
        icons=['info-circle', 'bar-chart', 'tag'], menu_icon="cast", default_index=0)
    
    if selected == "Предсказание":
        option = st.selectbox('Выберите модель для предсказания', ('KNN', 'XGBClassifier', 'Нейронная сеть'))

    
if selected == "Описание данных":
    st.markdown(INFO)

if selected == "Визуализация данных":
    st.markdown(INFO)

if selected == "Предсказание":
    st.markdown(INFO)
