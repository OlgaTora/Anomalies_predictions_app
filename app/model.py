import numpy as np
import pandas as pd

# from keras._tf_keras import keras
# from sklearn.preprocessing import RobustScaler, LabelEncoder
# from tensorflow.keras.models import load_model

from CONST import DF_PATH
from utils import read_csv_file, group_floors, group_year


def preprocess_input_data(data_df: pd.DataFrame) -> pd.DataFrame:
    """Препроцессинг введенных данных: преобразование типов, заполнение пропусков"""
    data_df['current_consumption'] = data_df['current_consumption'].astype(float)
    # Если нет данных
    data_df = data_df.fillna(0)
    # data_df['square'].fillna(-1)
    # data_df['floors'].fillna(-1)
    # data_df['contruction_date'].fillna(-1)
    data_df['date'] = pd.to_datetime(data_df[['year', 'month']].assign(day=1)).dt.strftime('%Y-%m-%d')
    return data_df


def get_mkd_data(data_df: pd.DataFrame) -> pd.DataFrame:
    mask = (~data_df.address.str.contains('Подобъект|Подъезд')) & (data_df.object_type == 'Многоквартирный дом')
    return data_df[mask]


def get_target_data(target: pd.DataFrame) -> tuple:
    target_date = target.iloc[0].date
    target_num_odpu = target.iloc[0].num_odpu
    target_consumption = float(target.iloc[0].current_consumption)
    return target_num_odpu, target_date, target_consumption


def concat_data(df: pd.DataFrame):
    target = preprocess_input_data(df)
    target_num_odpu, target_date, target_consumption = get_target_data(target)
    all_data = read_csv_file(DF_PATH)
    condition = ((all_data.num_odpu == target_num_odpu) & (all_data.date >= target_date))
    all_data = all_data[~condition]
    df = pd.concat((target, all_data), ignore_index=True)
    return df, target


def check_simple_anomalies(target: pd.DataFrame) -> bool:
    if (float(target.iloc[0].current_consumption) == 0) & (int(target.iloc[0].hot_water) != 1):
        return True


def check_same_values(df: pd.DataFrame) -> bool:
    df['anom'] = (df.current_consumption.pct_change()) == 0 | (df.current_consumption.pct_change(2) == 0)
    return df.loc[0].anom == 1


def check_anomalies_same_odpu(df: pd.DataFrame) -> bool:
    df['coef_per_day'] = round(df.current_consumption / df.ozp / df.temp_K, 6)
    df['prev_coef_per_day'] = df['coef_per_day'].shift(1)
    df['consumption_change_per_day'] = df['coef_per_day'].pct_change(fill_method=None)
    df.consumption_change_per_day = df.consumption_change_per_day.fillna(0)
    # data_df.consumption_change_per_day = data_df.consumption_change_per_day.replace([float('inf'), -float('inf')], 0)
    q1 = df['consumption_change_per_day'].quantile(0.25)
    q3 = df['consumption_change_per_day'].quantile(0.75)
    iqr = q3 - q1
    df['anom'] = np.where((df['consumption_change_per_day'] > q1 - iqr * 1.5) \
                          & (df['consumption_change_per_day'] < q3 + iqr * 1.5), False, True)
    return df.loc[0].anom == 1


def check_anomalies_mkd(data: pd.DataFrame):
    df, target = concat_data(data)
    df = get_mkd_data(df)
    df['const_date_group'] = df['contruction_date'].apply(group_year)
    df['floors_group'] = df['floors'].apply(group_floors)
    grouped_df = df.groupby(['const_date_group', 'floors_group', 'hot_water'])
    grouped_df = grouped_df['current_consumption'].mean().reset_index()
    df = df.merge(grouped_df, on=['const_date_group', 'floors_group', 'hot_water'], how='left')
    df['anom'] = ((df['current_consumption_x'] - df['current_consumption_y']) / df[
        'current_consumption_y']) * 100
    df['anom'] = np.where(df['anom'].abs() > 25, True, False)
    df = df.loc[target.index]
    return df.loc[0].anom == 1


def check_anomalies(data: pd.DataFrame):
    msg = ''
    flag = False
    df, target = concat_data(data)
    condition = (df.num_odpu == target.iloc[0].num_odpu)
    df = df[condition]
    df = df.sort_values(by='date')
    if check_simple_anomalies(data):
        msg = 'Нулевые показания'
        flag = True
    elif check_same_values(df):
        msg = 'Одинаковые показания за прошлые периоды'
        flag = True
    elif check_anomalies_same_odpu(df):
        msg = 'Аномальные показания в сравнении с прошлыми периодами'
        flag = True
    return flag, msg




