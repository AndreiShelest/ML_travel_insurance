import pandas as pd
import numpy as np
from category_encoders.cat_boost import CatBoostEncoder
from sklearn.model_selection import train_test_split


def split_data(data: pd.DataFrame, model_options: dict, data_params: dict) -> tuple:
    target_col = data_params['target_col']

    X = data.drop(columns=target_col)
    y = data[target_col]
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=model_options['test_size'],
        random_state=model_options['random_state'],
        stratify=y,
    )

    return X_train, X_test, y_train, y_test


def impute_data(data: pd.DataFrame) -> pd.DataFrame:
    data.loc[data['gender'].isna(), 'gender'] = 'Missing'

    return data


def encode_categorical_train(
    data: pd.DataFrame, target: pd.DataFrame, data_params: dict
) -> tuple[pd.DataFrame, pd.DataFrame, CatBoostEncoder]:
    # encode target
    target_encoded = target == 'Yes'

    # One-Hot encoder
    oh_cols = data_params['one_hot_cols']
    data_encoded = pd.get_dummies(data, columns=oh_cols)

    # catboost encoder
    catboost_cols = data_params['catboost']['cols']

    catboost_encoder = CatBoostEncoder(cols=catboost_cols)
    data_encoded = catboost_encoder.fit_transform(data_encoded, target_encoded)

    return data_encoded, target_encoded, catboost_encoder


def encode_categorical_test(
    data: pd.DataFrame,
    target: pd.DataFrame,
    data_params: dict,
    catboost_encoder: CatBoostEncoder,
) -> pd.DataFrame:
    # encode target
    target_encoded = target == 'Yes'

    # One-Hot encoder
    oh_cols = data_params['one_hot_cols']
    data_encoded = pd.get_dummies(data, columns=oh_cols)

    # catboost encoder
    data_encoded = catboost_encoder.transform(data_encoded)

    return data_encoded, target_encoded


def create_features(data: pd.DataFrame) -> pd.DataFrame:
    data['duration'] = np.where(data['duration'] < 0, 0, data['duration'])
    data['duration_sqrt'] = np.sqrt(data['duration'])

    data['net_sales_3'] = data['net_sales'] ** (1 / 3)
    data['commision_sqrt'] = data['commision'] ** 0.5

    return data
