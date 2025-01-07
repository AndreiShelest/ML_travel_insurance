import pandas as pd
import numpy as np
from category_encoders.cat_boost import CatBoostEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    average_precision_score,
)
from imblearn.under_sampling import RandomUnderSampler


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


def std_scale_data_train(
    data: pd.DataFrame, scaler_cols: list[str]
) -> tuple[pd.DataFrame, StandardScaler]:
    std_scaler = StandardScaler()

    scaler_cols_std = [f'{col}_std' for col in scaler_cols]
    data[scaler_cols_std] = std_scaler.fit_transform(data.loc[:, scaler_cols])

    return data, std_scaler


def std_scale_data_test(
    data: pd.DataFrame, scaler_cols: list[str], std_scaler: StandardScaler
) -> pd.DataFrame:
    scaler_cols_std = [f'{col}_std' for col in scaler_cols]
    data[scaler_cols_std] = std_scaler.transform(data.loc[:, scaler_cols])
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
    data['commision_sqrt'] = np.sqrt(data['commision'])

    return data


def train_logistic_regression(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int,
) -> list:
    y_train = pd.DataFrame(y_train).T.iloc[0]

    best_estimators = []

    for sample_size in np.arange(0.1, 1.01, 0.1):
        X_train_s, y_train_s = RandomUnderSampler(
            random_state=random_state, sampling_strategy=sample_size
        ).fit_resample(X_train, y_train)

        regr_cv = GridSearchCV(
            estimator=LogisticRegression(max_iter=10000),
            param_grid=[
                {
                    'solver': ['saga'],
                    'penalty': [None],
                    'class_weight': ['balanced'],
                },
                {
                    'solver': ['saga'],
                    'penalty': ['l1', 'l2'],
                    'C': [0.01, 0.25, 0.5, 1.0, 2.0],
                    'class_weight': ['balanced'],
                },
            ],
            scoring='average_precision',
            verbose=3,
            n_jobs=8,
            refit=True,
        )

        best_regr_fit = regr_cv.fit(X_train_s, y_train_s)
        best_estimators.append(best_regr_fit)

    return best_estimators
