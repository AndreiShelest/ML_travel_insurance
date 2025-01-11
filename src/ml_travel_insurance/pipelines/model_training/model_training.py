import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV


def train_logistic_regression(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int,
) -> list:
    y_train = pd.DataFrame(y_train).T.iloc[0]

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
            {
                'solver': ['saga'],
                'penalty': ['elasticnet'],
                'C': [0.01, 0.25, 0.5, 1.0, 2.0],
                'class_weight': ['balanced'],
                'l1_ratio': [0.25, 0.5, 0.75],
            },
        ],
        scoring='average_precision',
        verbose=3,
        n_jobs=8,
        refit=True,
    )

    best_regr_fit = regr_cv.fit(X_train, y_train)

    return [best_regr_fit]
