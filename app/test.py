import pandas as pd
import streamlit as st
from PIL import Image

from data_translation import translation


def set_app_config():
    """Настройки приложения и стили."""
    image = 'app/images/logo-new.png'
    image = Image.open(image)
    # image = image.resize((250, 250))
    st.set_page_config(
        layout='centered',
        initial_sidebar_state='expanded',
        page_title='Anomalies prediction',
        # page_icon=image,
    )
    st.image(image, width=400)
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

def sidebar():
    """Функция для обработки боковой панели ввода."""
    st.sidebar.header('Заданные пользователем параметры:')
    division = st.sidebar.selectbox("Подразделение", ("Уфа"))
    num_odpu = (st.sidebar.text_input("№ ОДПУ"))
    st.sidebar.text("Период показаний")
    month = st.sidebar.selectbox("месяц", ("январь", "февраль", "март", "апрель",
                                           "октябрь", "ноябрь", "декабрь"))
    year = st.sidebar.slider("год", min_value=2022, max_value=2050, step=1)
    hot_water = st.sidebar.selectbox("ГВС", ("да", "нет"))
    address = (st.sidebar.text_input("Адрес объекта"))
    object_type = st.sidebar.selectbox("Продавец", ("Другое строение", "Дет.ясли и сады",
                                                    "Многоквартирный дом",
                                                    "Административные здания, конторы",
                                                    "Школы и ВУЗ", "Магазины",
                                                    "Спортзалы, крытые стадионы и другие спортивные сооружения",
                                                    "Нежилой дом", "Пожарное депо", "Гаражи"))
    floors = st.sidebar.slider("Этажность объекта", min_value=0, max_value=100, value=0, step=1)
    contruction_date = st.sidebar.slider("Дата постройки", min_value=1800, max_value=2050, value=0, step=1)
    square = st.sidebar.slider("Общая площадь объекта", min_value=0, max_value=20000, value=0, step=1)
    current_consumption = st.sidebar.slider("Текущее потребление, Гкал", min_value=0, max_value=100, value=0, step=1)
    submit = st.sidebar.button("Отправить")
    data = {
        'division': division,
        'num_odpu': num_odpu,
        "month": translation[month],
        "year": year,
        'hot_water': translation[hot_water],
        'address': address,
        'object_type': object_type,
        'floors': floors,
        'contruction_date': contruction_date,
        'square': square,
        'current_consumption': current_consumption,
    }
    return data, submit


def display_results(data: dict, submit):
    """Функция для отображения результатов ввода на главной странице."""
    if submit:
        if data['num_odpu'] == "":
            st.error("Пожалуйста, введите номер ОДПУ.")
        else:
            df = pd.DataFrame(data, index=[0])
            write_user_data()
            st.write('## Ваши данные')
            st.write(df)


def write_user_data():
    click = st.button('Проверьте ваши данные')
    if click:
        st.success("Данные успешно отправлены.")
        print('ddd')

def main():
    """Основная функция для запуска приложения."""
    set_app_config()
    data, submit = sidebar()
    display_results(data, submit)


if __name__ == "__main__":
    main()
