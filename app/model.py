import pandas as pd
from sklearn.metrics import classification_report,accuracy_score,f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, RobustScaler, LabelEncoder
from utils import calculate_deviation


def encoder_cat(data_normalized: pd.DataFrame) -> pd.DataFrame:
    categorical_features = data_normalized.select_dtypes(include=['object', 'category']).columns.tolist()
    label_encoders = {feature: LabelEncoder() for feature in categorical_features}

    for feature, le in label_encoders.items():
        data_normalized[feature] = le.fit_transform(data_normalized[feature])

    scaler = RobustScaler()
    data_normalized = pd.DataFrame(scaler.fit_transform(data_normalized), columns=data_normalized.columns)
    return data_normalized


def generate_new_features(data_df: pd.DataFrame) -> pd.DataFrame:
    data_df['const_date_group'] = pd.cut(data_df['contruction_date'],
                                         bins=[-1, 0, 1958, 1989, 2000, 2010, 2024],
                                         labels=['-1', '<=1958', '1959-1989', '1990-2000', '2001-2010', '2011-2024'],
                                         include_lowest=True)

    data_df['floors_group'] = pd.cut(data_df['floors'],
                                     bins=[0, 2, 4, 9, 12, float('inf')],
                                     labels=['1-2', '3-4', '5-9', '10-12', '>=13'],
                                     include_lowest=True)
    data_df = (
        data_df.groupby(["year", "month", "object_type", "const_date_group", "floors_group"])
        .apply(calculate_deviation)
        .reset_index(drop=True)
    )
    data_df.cons_deviation = round(data_df.cons_deviation, 0)
    data_df.cons_deviation.fillna(0, inplace=True)
    data_df.cons_deviation = data_df.cons_deviation.astype(int)

    data_df['area_deviation'] = data_df.groupby(['address', 'object_type'])['square'].pct_change()
    # т.к. метод pct_change() сравнивает с предыдущей строкой, то у первого объекта в группе будет значение NaN. Заменим на 0
    if data_df['address'].nunique() > 1:
        data_df.loc[data_df['area_deviation'].isna(), 'area_deviation'] = 0

    # т.к. в ряде случаем площадь объекта указана как 0, то в ряде случаев (~300) возвращает inf. Заменим на 0
    data_df['area_deviation'] = data_df['area_deviation'].replace([float('inf'), -float('inf')], -1)

    # округлим до сотых
    data_df['area_deviation'] = round(data_df['area_deviation'], 2)
    data_df = data_df.sort_values(by=['address', 'num_odpu', 'year', 'month'])
    data_df['year_per_year_cons_devi'] = round(
        abs(data_df.groupby('address')['current_consumption'].pct_change(periods=7) * 100))
    # т.к. метод pct_change() сравнивает с предыдущей строкой, то у первого объекта в группе будет значение NaN. Заменим на 0
    if data_df['address'].nunique() > 1:
        data_df.loc[data_df['year_per_year_cons_devi'].isna(), 'year_per_year_cons_devi'] = 0
    # т.к. в ряде случаем площадь объекта указана как 0, то в ряде случаев (~300) возвращает inf. Заменим на 0
    data_df['year_per_year_cons_devi'] = data_df['year_per_year_cons_devi'].replace([float('inf'), -float('inf')], 0)
    data_df['year_per_year_cons_devi'] = data_df['year_per_year_cons_devi'].astype(int)
    data_df['cons_per_deg'] = abs(data_df['current_consumption'] / data_df['temperature'])
    data_df['temp_coef'] = data_df['current_consumption'] / (data_df['temperature'] - data_df['temperature'].mean())

    #2. равные значения показаний в течение нескольких расчетных периодов
    data_df['is_same_as_previous'] = int((data_df['current_consumption'] == data_df['current_consumption'].shift(1)))
    data_df['avg_cons'] = data_df['current_consumption'].rolling(window=3).mean()
    data_df['cons_dev'] = round(
        abs(((data_df['current_consumption'] - data_df['avg_cons']) / data_df['avg_cons']) * 100), 2)
    data_df['cons_dev'] = data_df['cons_dev'].replace([float('inf'), -float('inf')], 0)
    # Создаем столбец с предыдущим потреблением
    data_df['prev_consumption'] = data_df.groupby(['address', 'num_odpu', 'year'])['current_consumption'].shift(1)
    # Вычисляем разницу между текущим и предыдущим потреблением
    data_df['consumption_diff'] = abs(
        (data_df['current_consumption'] - data_df['prev_consumption']) / data_df['prev_consumption']) * 100
    data_df['consumption_diff'] = data_df['consumption_diff'].replace([float('inf'), -float('inf')], 0)
    data_df.drop('prev_consumption', axis=1, inplace=True)


    data_df['prev_temperature'] = data_df.groupby(['address', 'num_odpu', 'year'])['temperature'].shift(1)
    data_df['temperature_diff'] = (data_df['temperature'] - data_df['prev_temperature']) / data_df['prev_temperature']
    data_df['temperature_diff'] = data_df['temperature_diff'].replace([float('inf'), -float('inf')], 0)
    data_df.drop('prev_temperature', axis=1, inplace=True)
    data_df['consumption_times_temperature'] = data_df['current_consumption'] * data_df['temperature']
    data_df['consumption_minus_temperature'] = data_df['current_consumption'] - data_df['temperature']
    new_cols = (
        data_df["address"]
        .str.split(", ", expand=True)
        .rename(columns={0: "city", 1: "street", 2: "house", 3: "building"})
    )
    new_cols.drop(columns=["house", "building"], axis=1, inplace=True)
    data_df = data_df.join(new_cols)

    data_df.insert(3, 'city', data_df.pop('city'))
    data_df.insert(4, 'street', data_df.pop('street'))
    data_df['cons_dev_anom'] = data_df['cons_deviation'].apply(lambda x: 1 if x > 25 else 0)
    data_df['area_dev_anom'] = data_df['area_deviation'].apply(lambda x: 1 if x > 10 else 0)
    data_df['ypy_cons_devi_anom'] = data_df['year_per_year_cons_devi'].apply(lambda x: 1 if x > 25 else 0)
    data_df['cons_dev_anom'] = data_df['cons_dev'].apply(lambda x: 1 if x > 50 else 0)

    data_df[
        'anom_sum'] = data_df.is_same_as_previous + data_df.cons_dev_anom + data_df.area_dev_anom + data_df.ypy_cons_devi_anom + data_df.cons_dev_anom

    data_df['anom'] = data_df['anom_sum'].apply(lambda x: 1 if x != 0 else 0)
    return data_df

def load_model(path):
    pass
    # with open(path, 'rb') as file:
    #     return pickle.load(file)


def preprocess_data(df: pd.DataFrame):
    df = generate_new_features(df)
    df = encoder_cat(df)
    return df


def predict():
    pass


def open_data():
    pass

