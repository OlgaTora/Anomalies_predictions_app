from datetime import datetime

import pandas as pd
import streamlit as st
from PIL import Image

pd.set_option('future.no_silent_downcasting', True)

from CONST import START_DATE, END_DATE, DATA_TEMP_PATH, DATA_OBJECTS_PATH, MONTHS
from model import check_anomalies, check_anomalies_mkd
from utils import transform_df_temps, read_csv_file
from data_translation import translation


def set_app_config():
    """Настройки приложения и стили."""
    image = 'app/images/logo-new.png'
    image = Image.open(image)
    st.set_page_config(
        layout='wide',
        initial_sidebar_state='expanded',
        page_title='Anomalies prediction',
        # page_icon=image,
    )
    st.image(image, width=300)
    st.title(':blue[Сервис для проверки аномалий в платежах]')
    st.header(' :blue[Загрузка csv-файла]')

    file = st.file_uploader("Выберите файл:", type="csv")
    st.markdown(
        """
        <style>
        [data-testid=stAlert]{
            background-color: #f78f1e;
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
            font-size:45px !important;
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
    return file


def process_file(file):
    """Обработка загруженного файла."""
    if file is not None:
        try:
            df = pd.read_csv(file)
            cols_dict = {
                "Подразделение": "division",
                "№ ОДПУ": "num_odpu",
                "Вид энерг-а ГВС": "hot_water",
                "Адрес объекта": "address",
                "Тип объекта": "object_type",
                "Дата текущего показания": "date",
                "Текущее потребление, Гкал": "current_consumption",
            }
            df_ = df.copy()
            df_.rename(columns=cols_dict, inplace=True)
            # Список столбцов, которые нужно оставить
            columns_to_keep = [
                'division',
                'num_odpu',
                'hot_water',
                'address',
                'object_type',
                'date',
                'current_consumption']
            df_ = df_.loc[:, columns_to_keep]
            df_ = df_[df_.hot_water != 'ГВС (централ)']
            df_ = df_.dropna(subset=['date', 'current_consumption'])
            df_info = read_csv_file(DATA_OBJECTS_PATH)
            df_ = pd.merge(df_, df_info, how="left", on=['address', 'object_type'])
            df_.date = pd.to_datetime(df_.date)
            df_['year'] = df_['date'].dt.year
            df_['month'] = df_['date'].dt.month
            df_.date = df_.date.dt.date
            df_.hot_water = df_.hot_water.fillna(0)
            df_.hot_water = df_.hot_water.replace('ГВС-ИТП', 1)
            df_ = df_.fillna(0)
            for index in df_.index:
                row_df = df_.loc[[index]]
                anomaly, reason, consumption = process_data_inputs(row_df)
                if anomaly:
                    st.write(anomaly)
                    # df.loc[index].anomaly = anomaly
        except Exception as e:
            st.error(f"Произошла ошибка при обработке файла: {e}")


def sidebar() -> tuple[dict, bool] | None:
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
    floors = st.sidebar.slider('Этажность объекта', min_value=1, max_value=40, value=1, step=1)
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
    return data, submit if data else None


def display_results(data: dict, submit: bool) -> bool:
    """Функция для отображения результатов ввода на главной странице."""
    flag = False
    df = pd.DataFrame(data, index=[0])
    st.write('## :blue[Проверьте ваши данные перед отправкой.]')
    st.write(df)
    if submit:
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


def to_dataframe(data: dict) -> pd.DataFrame:
    return pd.DataFrame(data, index=[0])


def process_data_inputs(data: pd.DataFrame) -> tuple:
    """Объединяет полученные от пользователя данные и температуры"""
    anomaly = False
    consumption = None
    temps = transform_df_temps(DATA_TEMP_PATH)
    df = data.merge(temps, on=['month', 'year'])
    anomaly_check, reason = check_anomalies(df)
    if anomaly_check:
        anomaly = True
        consumption = df.loc[0].current_consumption
    else:
        if (df.loc[0].object_type == 'Многоквартирный дом') \
                and ('Подобъект' not in df.loc[0].address) \
                and ('Подъезд' not in df.loc[0].address):
            if check_anomalies_mkd(df):
                anomaly = True
                reason = 'Аномалия в сравнении с похожими домами'
                consumption = df.loc[0].current_consumption
    return anomaly, reason, consumption


def write_prediction(prediction: bool, reason: str = None, consumption=None):
    """Функция для вывода результатов предсказания на экран, вызывает функцию расчета предсказания"""
    if prediction:
        st.markdown(
            '<p class="pretty-font">Обнаружена аномалия в данных:</p>', unsafe_allow_html=True)
        st.header(f'Сумма показаний: {consumption}')
        st.header(reason)
    else:
        st.write(f'## Аномалии нет\nВведите новый запрос!')


def main():
    """Основная функция для запуска приложения."""
    file = set_app_config()
    data, submit = sidebar()
    process_file(file)
    if data:
        if display_results(data, submit):
            data = to_dataframe(data)
            anomaly, reason, consumption = process_data_inputs(data)
            write_prediction(anomaly, reason, consumption)


if __name__ == "__main__":
    main()
