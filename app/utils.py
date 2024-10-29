import numpy as np
import pandas as pd

from CONST import MONTHS, DF_PATH, DATA_PATH


def calculate_deviation(group):
    """Функция для расчета отклонения в группе и записи в новый столбец"""
    group["cons_deviation"] = (
        abs(
            (group["current_consumption"] - group["current_consumption"].mean())
            / group["current_consumption"].mean()
        )
        * 100
    )
    return group


def get_temperatures(data_dir: str):
    df = pd.read_excel(data_dir, skiprows=1)
    df.drop(columns=['Unnamed: 0'], inplace=True)
    df = df.set_index('Период').T
    df = df.reset_index()
    df = df[~df['index'].dt.month.isin(MONTHS)]
    df.rename(
        columns={"Тн.в, град.С": "temperature", "Продолжительность ОЗП, сут.": "ozp"},
        inplace=True,
    )
    df.temperature = round(df.temperature, 2)
    return df


def transform_df_temps(data_dir):
    temps = get_temperatures(data_dir)
    temps['month'] = temps['index'].apply(lambda x: str(x.month)).astype(int)
    temps['year'] = temps['index'].apply(lambda x: str(x.year)).astype(int)
    temps = temps.drop(columns=['index'])
    return temps


def get_all_data(data_dir):
    df = pd.read_csv(data_dir)
    return df


def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:i+seq_length, :-1]  # Исключаем последний столбец (метку аномалии)
        y = data[i+seq_length, -1]     # Метка аномалии
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

