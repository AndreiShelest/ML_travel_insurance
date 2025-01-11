import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegressionCV

import logging

logger = logging.getLogger(__name__)


def train_logistic_regression(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int,
) -> list:
    y_train = pd.DataFrame(y_train).T.iloc[0]

    cs = [0.01, 0.25, 0.5, 1.0, 2.0]

    l1_cv = LogisticRegressionCV(
        Cs=cs,
        penalty='l1',
        scoring='average_precision',
        max_iter=10000,
        class_weight='balanced',
        n_jobs=8,
        random_state=random_state,
        solver='saga',
    )
    l2_cv = LogisticRegressionCV(
        Cs=cs,
        penalty='l2',
        scoring='average_precision',
        max_iter=10000,
        class_weight='balanced',
        n_jobs=8,
        random_state=random_state,
        solver='saga',
    )
    elastic_cv = LogisticRegressionCV(
        Cs=cs,
        penalty='elasticnet',
        scoring='average_precision',
        max_iter=10000,
        class_weight='balanced',
        n_jobs=8,
        random_state=random_state,
        l1_ratios=[0.25, 0.5, 0.75],
        solver='saga',
    )

    logger.info('CV: L1')
    l1_fitted = l1_cv.fit(X_train, y_train)

    logger.info('CV: L2')
    l2_fitted = l2_cv.fit(X_train, y_train)

    logger.info('CV: elasticnet')
    elnet_fitted = elastic_cv.fit(X_train, y_train)

    best_models = [('L1', l1_fitted), ('L2', l2_fitted), ('elasticnet', elnet_fitted)]
    best_scores = [model.score(X_train, y_train) for (_, model) in best_models]
    grand_best_model = best_models[np.argmax(best_scores)]

    return {
        'grand_best_model': grand_best_model,
        'best_models': best_models,
    }
