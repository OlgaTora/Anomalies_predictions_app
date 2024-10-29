import pandas as pd
import streamlit as st
from PIL import Image
from model import preprocess_data, predict, open_data


def process_main_page():
    show_main_page()
    process_side_bar_inputs()


def show_main_page():
    image = 'app/images/logo-new.png'
    image = Image.open(image)
    # image = image.resize((250, 250))
    st.set_page_config(
        layout='wide',
        initial_sidebar_state='auto',
        page_title='Anomalies prediction',
        # page_icon=image,
    )

    st.image(image, width=400)
    st.title(':blue[Сервис для проверки аномалий в платежах]:')
    # st.header('Попробуй - убедись!')

    st.markdown("""
        <style>
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


def write_user_data(df):
    click = st.button('Проверьте ваши данные')
    if click:
        st.write('## Ваши данные')
        st.write(df)


def write_prediction(prediction):
    prediction = round(prediction[0])
    if prediction > 0:
        st.markdown('<p class="pretty-font">Рекомендуемая цена на данный автомобиль:</p>', unsafe_allow_html=True)
        st.header(prediction)
    else:
        st.write('## Введите запрос!')


def process_side_bar_inputs():
    st.sidebar.header('Заданные пользователем параметры:')
    user_input_df = sidebar_input_features()
    write_user_data(user_input_df)
    df = open_data()
    df = pd.concat((user_input_df, df), axis=0)
    print('----')
    # preprocessed_df = preprocess_data(df)
    # print(preprocessed_df)
    # user_df = preprocessed_df[:1]
    # prediction = predict(user_df)
    # write_prediction(prediction)


def sidebar_input_features():
    with st.sidebar.form(key='my_form'):
        submit_button = st.form_submit_button(label='Отправить')
        division = st.sidebar.selectbox("Подразделение", ("Уфа"))
        num_odpu = (st.sidebar.text_input("№ ОДПУ"))
        st.sidebar.text("Период показаний")
        month = st.sidebar.selectbox("месяц", ("январь", "февраль", "март", "апрель",
                                             "октябрь", "ноябрь", "декабрь"))
        year = st.sidebar.slider("год", min_value=2022, max_value=2050, value=0, step=1)
        hot_water = st.sidebar.selectbox("ГВС", ("да", "нет"))
        address = (st.sidebar.text_input("Адрес объекта"))
        object_type = st.sidebar.selectbox("Продавец", ("Другое строение","Дет.ясли и сады",
                                                        "Многоквартирный дом",
                                                        "Административные здания, конторы",
                                                        "Школы и ВУЗ", "Магазины",
                                                        "Спортзалы, крытые стадионы и другие спортивные сооружения",
                                                        "Нежилой дом", "Пожарное депо", "Гаражи"))
        floors = st.sidebar.slider("Этажность объекта", min_value=0, max_value=100, value=0, step=1)
        contruction_date = st.sidebar.slider("Дата постройки", min_value=1800, max_value=2050, value=0, step=1)
        square = st.sidebar.slider("Общая площадь объекта", min_value=0, max_value=20000, value=0, step=1)
        current_consumption = st.sidebar.slider("Текущее потребление, Гкал", min_value=0, max_value=100, value=0, step=1)

    translation = {
        "да": 1,
        "нет": 0,
        "январь": 1,
        "февраль": 2,
        "март": 3,
        "апрель": 4,
        "октябрь": 10,
        "ноябрь": 11,
        "декабрь": 12
    }

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

    df = pd.DataFrame(data, index=[0])
    print("ф")
    return df


if __name__ == "__main__":
    process_main_page()
