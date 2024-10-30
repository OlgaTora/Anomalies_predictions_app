from datetime import datetime

import pandas as pd
import streamlit as st
from PIL import Image

from CONST import DATA_PATH, START_DATE, END_DATE
from model import preprocess_data, predict
from utils import transform_df_temps
from data_translation import translation


def set_app_config():
    """Настройки приложения и стили."""
    image = 'app/images/logo-new.png'
    image = Image.open(image)
    # image = image.resize((250, 250))
    st.set_page_config(
        layout='wide',
        initial_sidebar_state='expanded',
        page_title='Anomalies prediction',
        # page_icon=image,
    )
    st.image(image, width=300)
    st.title(':blue[Сервис для проверки аномалий в платежах]')
    st.markdown(
        """
        <style>
        .reportview-container {
            background: white;
        }
        [data-testid=stSidebar] {
            background-color: #f78f1e;
            color: #FFFFFF;
            }
        .stButton > button {
            box-shadow: inset 0px 1px 0px 0px #fff6af;
	        background: linear-gradient(to bottom, #ffec64 5%, #f79d0d 100%);
	        background-color: #f78f1e;
	        border-radius: 6px;
	        border: 1px solid #ffaa22;
        	display: inline-block;
	        cursor: pointer;
	        color: #000000;
	        font-family: Roboto;
        	font-size: 20px;
	        font-weight: bold;
	        padding: 10px 35px;
	        text-shadow: 0px 1px 0px #ffee66;
        }

        .pretty-font {
            font-size:50px !important;
            color:rgb(235, 102, 0)
        }

        .st-by {
            background:#fb9e25;
            color:#ffffff
        }

        .st-aw{
            background:#000000;            
        }  
        .StyledThumbValue{
            color:#ffffff
        }
        </style>
    """, unsafe_allow_html=True)


def sidebar() -> tuple[dict, bool]:
    """Функция для обработки боковой панели ввода."""
    st.sidebar.header('Заданные пользователем параметры:')
    division = st.sidebar.selectbox("Подразделение", ("Уфа"))
    num_odpu = st.sidebar.text_input("№ ОДПУ")
    st.sidebar.text("Период показаний")
    month = st.sidebar.selectbox("месяц", ("январь", "февраль", "март", "апрель",
                                           "октябрь", "ноябрь", "декабрь"))
    year = st.sidebar.slider("год", min_value=2021, max_value=2023, step=1)
    hot_water = st.sidebar.selectbox("ГВС", ("да", "нет"))
    address = st.sidebar.text_input("Адрес объекта")
    object_type = st.sidebar.selectbox("Тип объекта", ("Другое строение", "Дет.ясли и сады",
                                                       "Многоквартирный дом",
                                                       "Административные здания, конторы",
                                                       "Школы и ВУЗ", "Магазины",
                                                       "Спортзалы, крытые стадионы и другие спортивные сооружения",
                                                       "Нежилой дом", "Пожарное депо", "Гаражи"))
    floors = st.sidebar.slider('Этажность объекта', min_value=1, max_value=100, value=1, step=1)
    contruction_date = st.sidebar.text_input('Дата постройки', value='0', max_chars=4)
    square = st.sidebar.text_input('Общая площадь объекта')
    current_consumption = st.sidebar.text_input('Текущее потребление, Гкал')
    submit = st.sidebar.button('Отправить')
    data = {
        'division': division,
        'num_odpu': num_odpu,
        "month": translation[month],
        "year": year,
        'hot_water': translation[hot_water],
        'address': address,
        'object_type': object_type,
        'floors': floors,
        'contruction_date': int(contruction_date),
        'square': square,
        'current_consumption': current_consumption,
    }
    return data, submit


def display_results(data: dict, submit: bool) -> bool:
    """Функция для отображения результатов ввода на главной странице."""
    flag = False
    df = pd.DataFrame(data, index=[0])
    st.write('## Проверьте ваши данные перед отправкой.')
    st.write(df)
    if submit:
        # print(datetime(data['year'], data['month'], 1).date())
        if data['num_odpu'] == "":
            st.error('Пожалуйста, введите номер ОДПУ.')
        elif data['address'] == '':
            st.error('Пожалуйста, введите адрес.')
        elif data['year'] == '':
            st.error('Пожалуйста, введите год.')
        elif data['current_consumption'] == '':
            st.error('Пожалуйста, введите сумму ГКал.')
        elif ((datetime.strptime(START_DATE, "%Y-%m-%d")
              > datetime(data['year'], data['month'], 1)
                or (datetime(data['year'], data['month'], 1)
                    > datetime.strptime(END_DATE, "%Y-%m-%d")))):
            st.error(f'Дата ограничена данными: от {START_DATE} до {END_DATE}')
        else:
            st.success('Данные успешно отправлены.')
            flag = True
    return flag


def process_side_bar_inputs(data: dict, submit: bool):
    """Объединяет полученные от пользователя данные и температуры, вызывает функцию препроцессинга"""
    flag = display_results(data, submit)
    if flag:
        temps = transform_df_temps(DATA_PATH)
        df = pd.DataFrame(data, index=[0])
        df = df.merge(temps, on=['month', 'year'])
        df, object_type = preprocess_data(df)
        if object_type == 'Многоквартирный дом':
            write_prediction(df, True)
        else:
            write_prediction(df, False)


def write_prediction(test: pd.DataFrame, mkd: bool):
    """Функция для вывода результатов предсказания на экран, вызывает функцию расчета предсказания"""
    prediction = predict(test, mkd)
    if prediction > 0:
        st.markdown(
            '<p class="pretty-font">Обнаружена аномалия в данных:</p>', unsafe_allow_html=True)
        st.header(prediction)
    else:
        st.header(prediction)
        st.write('## Введите запрос!')


def main():
    """Основная функция для запуска приложения."""
    set_app_config()
    data, submit = sidebar()
    process_side_bar_inputs(data, submit)


if __name__ == "__main__":
    main()
