from functions.visualization.data import df
import plotly.express as px
import plotly.figure_factory as ff
import streamlit as st
import numpy as np


def draw_depend():
    options = st.multiselect('Выберите пару признаков, чтобы отобразить зависимость между ними', 
                             ['relative_velocity', 'est_diameter_min', 'est_diameter_max', 'miss_distance', 'absolute_magnitude', 'hazardous'], max_selections=2)
    if len(options) == 2:
        fig1 = px.scatter(df, y=options[1], x=options[0], height=550, width=650)
        fig1.update_traces(marker_size=10)
        st.plotly_chart(fig1, use_container_width=True)


@st.cache_data
def draw_corr():
    fig2 = px.imshow(df.corr(), text_auto=True, height=550, width=650)
    st.plotly_chart(fig2, use_container_width=True)


def draw_box():
    type = st.selectbox('Название признака', ['relative_velocity', 'est_diameter_min', 'est_diameter_max', 'miss_distance', 'absolute_magnitude'])
    fig3 = px.box(df, y=type, height=550, width=650)
    st.plotly_chart(fig3, use_container_width=True)


def visualization():
    tab1, tab2, tab3 = st.tabs(["График зависимостей", "Матрица корреляции", "Ящик с усами"])

    with tab1:
        st.write("""
        График зависимостей показывает точки в двумерном пространстве, координаты которых являются значениями выбранных признаков из исходного признакового пространства. Тем самым получается установить зависимости между признаками для лучшего подбора алгориотма.
        """)
        draw_depend()

    with tab2:
        st.write("""
        Матрица корреляции позволяет исследовать признаки на наличие попарной линейной зависимости (как прямой, так и обратной). Благодаря этому можно проверить мультиколлинеарность признаков.
        """)
        draw_corr()

    with tab3:
        st.write("""
        "Ящик с усами" позволяет исследовать признак с точки зрения случайной величины - найти математическое ожидание признака, его среднеквадратическое отклонение и диапозон, в котором находится большинство число объектов. Помогает искать выбросы.
        """)
        draw_box()
