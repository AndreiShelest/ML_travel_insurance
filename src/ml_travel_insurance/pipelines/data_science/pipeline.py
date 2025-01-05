from kedro.pipeline import Pipeline, node, pipeline

from .nodes import split_data, encode_categorical_train, impute_data


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
                    'TI_train_raw',
                    'TI_test_raw',
                    'TI_y_train_raw',
                    'TI_y_test_raw',
                ],
                name='split_data_node',
            ),
            node(
                func=impute_data,
                inputs='TI_train_raw',
                outputs='TI_train_imp',
                name='impute_data',
            ),
            node(
                func=encode_categorical_train,
                inputs=[
                    'TI_train_imp',
                    'TI_y_train_raw',
                    'params:travel_insurance.encoding',
                ],
                outputs=['TI_train_enc', 'TI_y_train', 'catboost_encoder'],
                name='encode_categorical_train',
            ),
        ]
    )
