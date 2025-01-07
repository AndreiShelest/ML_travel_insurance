from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (
    split_data,
    encode_categorical_train,
    encode_categorical_test,
    impute_data,
    create_features,
    std_scale_data_train,
    std_scale_data_test,
)


def create_data_science_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=split_data,
                inputs=[
                    'travel_insurance',
                    'params:model_options',
                    'params:travel_insurance',
                ],
                outputs=[
                    'TI_train',
                    'TI_test',
                    'TI_y_train',
                    'TI_y_test',
                ],
                name='split_data_node',
            ),
            # train part
            node(
                func=impute_data,
                inputs='TI_train',
                outputs='TI_train_imp',
                name='impute_data_train',
            ),
            node(
                func=encode_categorical_train,
                inputs=[
                    'TI_train_imp',
                    'TI_y_train',
                    'params:travel_insurance.encoding',
                ],
                outputs=['TI_train_enc', 'TI_y_train_feature', 'catboost_encoder'],
                name='encode_categorical_train',
            ),
            node(
                func=std_scale_data_train,
                inputs=['TI_train_enc', 'params:travel_insurance.scaler.std'],
                outputs=['TI_train_scaled', 'std_scaler'],
                name='std_scale_data_train',
            ),
            node(
                func=create_features,
                inputs='TI_train_scaled',
                outputs='TI_train_feature',
                name='create_features_train',
            ),
            # test part
            node(
                func=impute_data,
                inputs='TI_test',
                outputs='TI_test_imp',
                name='impute_data_test',
            ),
            node(
                func=encode_categorical_test,
                inputs=[
                    'TI_test_imp',
                    'TI_y_test',
                    'params:travel_insurance.encoding',
                    'catboost_encoder',
                ],
                outputs=[
                    'TI_test_enc',
                    'TI_y_test_feature',
                ],
                name='encode_categorical_test',
            ),
            node(
                func=std_scale_data_test,
                inputs=[
                    'TI_test_enc',
                    'params:travel_insurance.scaler.std',
                    'std_scaler',
                ],
                outputs='TI_test_scaled',
                name='std_scale_data_test',
            ),
            node(
                func=create_features,
                inputs='TI_test_scaled',
                outputs='TI_test_feature',
                name='create_features_test',
            ),
        ]
    )
