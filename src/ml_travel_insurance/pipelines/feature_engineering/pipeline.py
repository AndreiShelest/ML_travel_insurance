from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (
    split_data,
    encode_categorical_train,
    encode_categorical_test,
    std_scale_data_train,
    std_scale_data_test,
    other_feature_transformation_train,
    other_feature_transformation_test,
)


def create_pipeline(**kwargs) -> Pipeline:
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
                func=other_feature_transformation_train,
                inputs='TI_train',
                outputs=['TI_train_other_feature', 'other_train_params'],
                name='other_features_train',
            ),
            node(
                func=encode_categorical_train,
                inputs=[
                    'TI_train_other_feature',
                    'TI_y_train',
                    'params:travel_insurance.encoding',
                ],
                outputs=['TI_train_enc', 'TI_y_train_feature', 'catboost_encoder'],
                name='encode_categorical_train',
            ),
            node(
                func=std_scale_data_train,
                inputs=['TI_train_enc', 'params:travel_insurance.scaler.std'],
                outputs=['TI_train_feature', 'std_scaler'],
                name='std_scale_data_train',
            ),
            # test part
            node(
                func=other_feature_transformation_test,
                inputs=['TI_test', 'other_train_params'],
                outputs='TI_test_other_feature',
                name='other_features_test',
            ),
            node(
                func=encode_categorical_test,
                inputs=[
                    'TI_test_other_feature',
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
                outputs='TI_test_feature',
                name='std_scale_data_test',
            ),
        ]
    )
