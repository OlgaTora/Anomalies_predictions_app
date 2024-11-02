import numpy as np
import pandas as pd

from CONST import MONTHS


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


def read_xls_file(data_dir: str) -> pd.DataFrame:
    """Чтение xls файла"""
    df = pd.read_excel(data_dir, skiprows=1)

    return df


def transform_df_temps(data_dir: str) -> pd.DataFrame:
    """Функция для преобразования файла температур"""
    df = read_xls_file(data_dir)
    df.drop(columns=['Unnamed: 0'], inplace=True)
    df = df.set_index('Период').T
    df = df.reset_index()
    df = df[~df['index'].dt.month.isin(MONTHS)]
    df.rename(
        columns={"Тн.в, град.С": "temperature", "Продолжительность ОЗП, сут.": "ozp"},
        inplace=True,
    )
    df['temp_K'] = df['temperature'] + 273.15
    df = df.sort_values(by=['index'])
    df['prev_temp_K'] = df['temp_K'].shift(1)
    df['temp_change_K'] = (df['temp_K'] - df['prev_temp_K']) / df['prev_temp_K']

    df.temperature = round(df.temperature, 6)
    df.temp_change_K = round(df.temperature, 6)
    df.prev_temp_K = round(df.temperature, 6)
    df.temp_K = round(df.temperature, 6)

    df['month'] = df['index'].apply(lambda x: str(x.month)).astype(int)
    df['year'] = df['index'].apply(lambda x: str(x.year)).astype(int)
    df = df.drop(columns=['index'])
    return df


def read_csv_file(data_dir: str) -> pd.DataFrame:
    """Чтение csv файла"""
    df = pd.read_csv(data_dir)
    return df


def create_sequences(data, seq_length):
    X_test = data[:, -1]
    X_test_padded = np.zeros((X_test.shape[0], seq_length, 33))
    X_test_padded[:, :1, :33] = X_test
    return X_test_padded

