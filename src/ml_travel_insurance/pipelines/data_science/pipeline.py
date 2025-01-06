from kedro.pipeline import Pipeline, node, pipeline

from .nodes import split_data, encode_categorical_train, impute_data, create_features


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
                name='impute_data',
            ),
            node(
                func=encode_categorical_train,
                inputs=[
                    'TI_train_imp',
                    'TI_y_train',
                    'params:travel_insurance.encoding',
                ],
                outputs=['TI_train_enc', 'TI_y_train_enc', 'catboost_encoder'],
                name='encode_categorical_train',
            ),
            node(
                func=create_features,
                inputs='TI_train_enc',
                outputs='TI_train_feature',
                name='create_features',
            ),
        ]
    )
