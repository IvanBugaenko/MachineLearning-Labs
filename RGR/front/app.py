import streamlit as st
from streamlit_option_menu import option_menu
from IoC import IoC


with st.sidebar:
    selected = option_menu("Меню", ["Описание данных", 'Визуализации', 'Предсказание'], 
        icons=['info-circle', 'bar-chart', 'tag'], menu_icon="cast", default_index=0)


IoC["app"][selected]()
