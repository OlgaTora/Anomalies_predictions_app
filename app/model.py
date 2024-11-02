import numpy as np
import pandas as pd
from keras._tf_keras import keras

from sklearn.preprocessing import RobustScaler, LabelEncoder
from tensorflow.keras.models import load_model

from CONST import DF_PATH, WEIGHTS_PATH, WEIGHTS_PATH_MKD
from utils import calculate_deviation, read_xls_file, create_sequences, read_csv_file


def encoder_cat(data: pd.DataFrame) -> pd.DataFrame:
    """Кодировние категориальных признаков, нормализация данных"""
    categorical_features = (data
                            .select_dtypes(include=['object', 'category']).columns.tolist())
    label_encoders = {feature: LabelEncoder() for feature in categorical_features}

    for feature, le in label_encoders.items():
        data[feature] = le.fit_transform(data[feature])

    scaler = RobustScaler()
    data = pd.DataFrame(
        scaler.fit_transform(data), columns=data.columns)
    return data


def preprocess_input_data(data_df: pd.DataFrame) -> pd.DataFrame:
    """Препроцессинг введенных данных: преобразование типов, заполнение пропусков"""
    data_df['current_consumption'] = data_df['current_consumption'].astype(float)
    data_df['square'] = data_df['square'].astype(float)

    # Если нет данных
    data_df['square'].replace(0, -1)
    data_df['contruction_date'].replace(0, -1)
    data_df['date'] = pd.to_datetime(data_df[['year', 'month']].assign(day=1)).dt.strftime('%Y-%m-%d')
    return data_df


def generate_new_features(data_df: pd.DataFrame) -> pd.DataFrame:
    """Генерация признаков для модели"""
    # группировка по этажности и дате постройки
    data_df['const_date_group'] = pd.cut(data_df['contruction_date'],
                                         bins=[-1, 0, 1958, 1989, 2000, 2010, 2024],
                                         labels=['-1', '<=1958', '1959-1989', '1990-2000', '2001-2010', '2011-2024'],
                                         include_lowest=True)
    data_df['floors_group'] = pd.cut(data_df['floors'],
                                     bins=[0, 2, 4, 9, 12, float('inf')],
                                     labels=['1-2', '3-4', '5-9', '10-12', '>=13'],
                                     include_lowest=True)
    # разница в потреблении объекта в отношении к средней потребления по группе
    data_df['cons_deviation'] = 0
    data_df = (
        data_df.groupby(["year", "month", "object_type", "const_date_group", "floors_group"], observed=False)
        .apply(calculate_deviation)
        .reset_index(drop=True)
    )
    data_df.cons_deviation = round(data_df.cons_deviation, 0)

    # изменение потребления относительно того же месяца предшествующего периода
    data_df = data_df.sort_values(by=['address', 'num_odpu', 'year', 'month'])
    data_df['year_per_year_cons_devi'] = round(
        abs(data_df.groupby('address')['current_consumption'].pct_change(periods=7) * 100))
    if data_df['address'].nunique() > 1:
        data_df.loc[data_df['year_per_year_cons_devi'].isna(), 'year_per_year_cons_devi'] = 0
    data_df['year_per_year_cons_devi'] = (
        data_df['year_per_year_cons_devi'].replace([float('inf'), -float('inf')], 0))
    data_df['year_per_year_cons_devi'] = data_df['year_per_year_cons_devi']  # .astype(int)
    data_df['is_same_as_previous'] = (
            data_df['current_consumption'] == data_df['current_consumption'].shift(1)).astype(int)
    # среднее потребление за 3 месяца
    data_df['avg_cons'] = data_df['current_consumption'].rolling(window=3).mean()
    # отклонение от среднего потребления
    data_df['cons_dev'] = round(
        abs(((data_df['current_consumption'] - data_df['avg_cons']) / data_df['avg_cons']) * 100), 2)
    data_df['cons_dev'] = data_df['cons_dev'].replace([float('inf'), -float('inf')], 0)
    data_df.cons_dev = data_df.cons_dev.fillna(0)
    # столбец с предыдущим потреблением
    data_df['prev_consumption'] = data_df.groupby(['address', 'num_odpu', 'year'])['current_consumption'].shift(1)
    # лаг потребления Гкал за предыдущий месяц
    data_df['consumption_diff'] = abs(
        (data_df['current_consumption'] - data_df['prev_consumption']) / data_df['prev_consumption']) * 100
    data_df['consumption_diff'] = data_df['consumption_diff'].replace([float('inf'), -float('inf')], 0)
    data_df = data_df.drop('prev_consumption', axis=1)
    data_df.consumption_diff = data_df.consumption_diff.fillna(0)
    # потребление на градус
    data_df['cons_per_deg'] = abs(data_df['current_consumption'] / data_df['temperature'])
    # температурный коэффициент
    data_df['temp_coef'] = (
            data_df['current_consumption'] / (data_df['temperature'] - data_df['temperature'].mean()))
    # лаг температуры за предыдущий месяц;
    data_df['prev_temperature'] = data_df.groupby(['address', 'num_odpu', 'year'])['temperature'].shift(1)
    data_df['temperature_diff'] = (
            (data_df['temperature'] - data_df['prev_temperature']) / data_df['prev_temperature'])
    data_df['temperature_diff'] = data_df['temperature_diff'].replace([float('inf'), -float('inf')], 0)
    data_df.drop('prev_temperature', axis=1, inplace=True)
    data_df.temperature_diff = data_df.temperature_diff.fillna(0)
    # произведение потребления Гкал и температуры
    data_df['consumption_times_temperature'] = data_df['current_consumption'] * data_df['temperature']
    # разница между потреблением Гкал и температурой
    data_df['consumption_minus_temperature'] = data_df['current_consumption'] - data_df['temperature']
    # выделение улицы и города
    data_df[['city', 'street', 'house', 'building']] = data_df['address'].str.split(', ', expand=True)
    data_df = data_df.drop(columns=["house", "building"], axis=1)#, errors='ignore')
    data_df = data_df.drop('address', axis=1)
    # таргеты
    data_df['cons_dev_anom'] = data_df['cons_deviation'].apply(lambda x: 1 if x > 25 else 0)
    data_df['ypy_cons_devi_anom'] = data_df['year_per_year_cons_devi'].apply(lambda x: 1 if x > 25 else 0)
    data_df['cons_dev_anom'] = data_df['cons_dev'].apply(lambda x: 1 if x > 50 else 0)
    data_df['anom_sum'] = \
        (data_df.is_same_as_previous + data_df.cons_dev_anom
         + data_df.ypy_cons_devi_anom + data_df.cons_dev_anom)

    data_df.insert(5, 'city', data_df.pop('city'))
    data_df.insert(6, 'street', data_df.pop('street'))
    data_df.insert(8, 'date', data_df.pop('date'))
    data_df.insert(9, 'current_consumption', data_df.pop('current_consumption'))
    data_df.insert(10, 'floors', data_df.pop('floors'))

    data_df.insert(17, 'month', data_df.pop('month'))
    data_df.insert(16, 'year', data_df.pop('year'))
    data_df.insert(23, 'cons_per_deg', data_df.pop('cons_per_deg'))
    data_df.insert(24, 'temp_coef', data_df.pop('temp_coef'))
    data_df.insert(25, 'avg_cons', data_df.pop('avg_cons'))
    data_df.insert(26, 'cons_dev', data_df.pop('cons_dev'))
    data_df.insert(27, 'consumption_diff', data_df.pop('consumption_diff'))
    return data_df


def preprocess_data(df: pd.DataFrame):
    """
    Подготовка данных к тестированию (делается на всем массиве, так как нужны исторические
    данные для выявления аномалий.
    Возращает только строку с таргетом.
    """
    target = preprocess_input_data(df)
    print(target.current_consumption)
    target_date = target.iloc[0].date
    target_consumption = target.iloc[0].current_consumption
    num_odpu = target.iloc[0].num_odpu
    all_data = read_csv_file(DF_PATH)
    df = pd.concat((target, all_data), ignore_index=True)
    df = generate_new_features(df)
    target = df[(df.num_odpu == num_odpu) & (df.date == target_date) & (df.current_consumption == target_consumption)]
    print(df.loc[target.index])
    df = encoder_cat(df)
    # Оставим только таргет
    df = df.loc[target.index]
    df = df.fillna(0)
    return df, target.object_type.to_list()


def generate_new_features_1(data_df):
    data_df = data_df.sort_values(by=['num_odpu', 'address', 'date'])
    data_df['zero_consumption'] = (data_df.current_consumption == 0) & (data_df.hot_water != 1)
    data_df['prev_months_change'] = (data_df.groupby(['num_odpu', 'address', 'year'])[
                                         'current_consumption'].pct_change() == 0) | \
                                    (data_df.groupby(['num_odpu', 'address', 'year'])['current_consumption'].pct_change(
                                        2) == 0)
    data_df['coef_per_day'] = round(data_df.current_consumption / data_df.ozp / data_df.temp_K, 6)
    data_df['prev_coef_per_day'] = data_df.groupby(['num_odpu', 'address'])['coef_per_day'].shift(1)
    data_df['consumption_change_per_day'] = (
        data_df.groupby(['num_odpu', 'address'])['coef_per_day'].pct_change())  # > 0.25
    data_df.consumption_change_per_day = data_df.consumption_change_per_day.fillna(0)
    data_df.consumption_change_per_day = data_df.consumption_change_per_day.replace([float('inf'), -float('inf')], 0)

    data_df.insert(16, 'month', data_df.pop('month'))
    data_df.insert(5, 'object_type', data_df.pop('object_type'))

    data_df.insert(6, 'date', data_df.pop('date'))
    data_df.insert(7, 'current_consumption', data_df.pop('current_consumption'))
    data_df.insert(15, 'year', data_df.pop('year'))
    return data_df


def predict(test: pd.DataFrame, mkd: bool) -> int:
    """Получение предсказания"""
    print(test.current_consumption)

    seq_length = 50  # Длина временного окна
    X = create_sequences(test.values, seq_length)
    if mkd:
        model = keras.models.load_model(WEIGHTS_PATH_MKD)
    else:
        model = keras.models.load_model(WEIGHTS_PATH)
    y_pred = model.predict(X)#[0][0]
    # Преобразуем вероятности в метки классов
    # y_pred = (y_pred > 0.5).astype(int)
    return y_pred
    # import pickle
    # print(test.current_consumption)
    # with open('app/models/data.pickle', "rb") as openfile:
    #     while True:
    #         try:
    #             model = pickle.load(openfile)
    #             y_pred = model.predict(test)
    #             print(model.predict_proba(test))
    #             # y_pred = model.predict_proba(test)[:, 1]
    #         except EOFError:
    #             break
    # return y_pred

