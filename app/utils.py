import numpy as np
import pandas as pd


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
    # df = df[~df['index'].dt.month.isin(MONTHS)]
    df.rename(
        columns={"Тн.в, град.С": "temperature", "Продолжительность ОЗП, сут.": "ozp"},
        inplace=True,
    )
    df['temp_K'] = df['temperature'] + 273.15
    df.temperature = round(df.temperature, 6)
    df.temp_K = round(df.temp_K, 6)
    df['month'] = df['index'].apply(lambda x: str(x.month)).astype(int)
    df['year'] = df['index'].apply(lambda x: str(x.year)).astype(int)
    df = df.drop(columns=['index'])
    return df


def read_csv_file(data_dir: str) -> pd.DataFrame:
    """Чтение csv файла"""
    df = pd.read_csv(data_dir)
    return df


def group_year(year):
    if year <= 1958:
        return 'до 1958 г.'
    elif 1959 <= year <= 1989:
        return '1959-1989 гг.'
    elif 1990 <= year <= 2000:
        return '1990-2000 гг.'
    elif 2001 <= year <= 2010:
        return '2001-2010 гг.'
    else:
        return '2011-2024 гг.'


def group_floors(floors):
    if floors in range(1, 3):
        return '1-2 этажа'
    elif floors in range(3, 5):
        return '3-4 этажа'
    elif floors in range(5, 10):
        return '5-9 этажей'
    elif floors in range(10, 13):
        return '10-12 этажей'
    else:
        return '13 и более этажей'
