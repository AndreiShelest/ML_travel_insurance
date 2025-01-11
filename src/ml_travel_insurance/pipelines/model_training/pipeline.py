from kedro.pipeline import Pipeline, pipeline, node
from .model_training import train_logistic_regression, train_neural_network


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=train_logistic_regression,
                inputs=[
                    'TI_train_feature',
                    'TI_y_train_feature',
                    'params:model_options.random_state',
                ],
                outputs='logit_trained',
                name='train_logit',
            ),
            node(
                func=train_neural_network,
                inputs=[
                    'TI_train_feature',
                    'TI_y_train_feature',
                    'params:model_options.random_state',
                ],
                outputs='nn_trained',
                name='train_nn',
            ),
        ]
    )
