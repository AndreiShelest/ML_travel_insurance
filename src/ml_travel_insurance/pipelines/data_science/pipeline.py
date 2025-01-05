from kedro.pipeline import Pipeline, node, pipeline

from .nodes import split_data


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
        ]
    )
