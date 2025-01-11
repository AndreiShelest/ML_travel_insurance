import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import ParameterGrid, StratifiedKFold
import tensorflow as tf

import logging

os.environ['KERAS_BACKEND'] = 'tensorflow'

tf.config.threading.set_intra_op_parallelism_threads(8)
tf.config.threading.set_inter_op_parallelism_threads(8)

# Note that Keras should only be imported after the backend
# has been configured. The backend cannot be changed once the
# package is imported.
import keras


logger = logging.getLogger(__name__)


def train_logistic_regression(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int,
) -> dict:
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


def train_neural_network(
    X_train: pd.DataFrame, y_train: pd.DataFrame, random_state: int
) -> dict:
    keras.utils.set_random_seed(random_state)

    kX_train = X_train.astype(np.float64).values
    kY_train = y_train.astype(np.int64).values

    counts = np.bincount(kY_train[:, 0])
    class_weights = {0: 1 / counts[0], 1: 1 / counts[1]}

    params_grid = ParameterGrid(
        {
            'dense_output_size': np.arange(10, 331, 40),
            'dropout_size': np.arange(0, 0.61, 0.15),
        }
    )
    # params_grid = ParameterGrid({'dense_output_size': [256], 'dropout_size': [0.3]})
    all_params = list(params_grid)
    n_folds = 5

    params_evals = []

    logger.info('Starting NN CV.')

    for params in all_params:
        logger.info(f'Params set for CV: {params}.')

        metric_values = []

        skf = StratifiedKFold(n_splits=n_folds)
        for fold_idx, (train_index, val_index) in enumerate(
            skf.split(kX_train, kY_train)
        ):
            nn_model = _get_neural_network(
                kX_train.shape, params['dense_output_size'], params['dropout_size']
            )

            nn_model.fit(
                kX_train[train_index],
                kY_train[train_index],
                batch_size=4096,
                epochs=50,
                validation_data=(kX_train[val_index], kY_train[val_index]),
                class_weight=class_weights,
                verbose=2,
            )

            nn_model_valuation = nn_model.evaluate(
                kX_train[val_index],
                kY_train[val_index],
                batch_size=4096,
                return_dict=True,
            )

            metric_values.append(nn_model_valuation['pr_auc'])

        final_metric_value = np.mean(metric_values)
        params_evals.append((final_metric_value, params))

        logger.info(f'Set metric value: {final_metric_value}.')

    best_params_idx = np.argmax([pe[0] for pe in params_evals])
    best_params = params_evals[best_params_idx][1]

    # train model again on best parameters
    best_model = _get_neural_network(
        kX_train.shape, best_params['dense_output_size'], best_params['dropout_size']
    )
    best_model.fit(
        kX_train[train_index],
        kY_train[train_index],
        batch_size=4096,
        epochs=50,
        validation_split=0.2,
        class_weight=class_weights,
        verbose=2,
    )

    logger.info(
        f'Best params: {best_params}, best model metric: {params_evals[best_params_idx][0]}.'
    )

    return {
        'best_model': best_model,
        'best_params': best_params,
        'params_evals': params_evals,
    }


def _get_neural_network(
    kX_train_shape: tuple, dense_output_size: int, dropout_size: float
) -> keras.Model:
    metrics = [
        keras.metrics.FalseNegatives(name='fn'),
        keras.metrics.FalsePositives(name='fp'),
        keras.metrics.TrueNegatives(name='tn'),
        keras.metrics.TruePositives(name='tp'),
        keras.metrics.Precision(name='precision'),
        keras.metrics.Recall(name='recall'),
        keras.metrics.F1Score(threshold=0.5, name='f1'),
        keras.metrics.AUC(name='pr_auc', curve='PR'),
    ]

    model = keras.Sequential(
        [
            keras.Input(shape=kX_train_shape[1:]),
            keras.layers.Dense(dense_output_size, activation='relu'),
            keras.layers.Dense(dense_output_size, activation='relu'),
            keras.layers.Dropout(dropout_size),
            keras.layers.Dense(dense_output_size, activation='relu'),
            keras.layers.Dropout(dropout_size),
            keras.layers.Dense(1, activation='sigmoid'),
        ]
    )

    model.compile(
        optimizer=keras.optimizers.Adam(1e-2),
        loss='binary_crossentropy',
        metrics=metrics,
    )

    return model
