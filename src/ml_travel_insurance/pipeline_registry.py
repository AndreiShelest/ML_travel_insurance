"""Project pipelines."""

from kedro.framework.project import find_pipelines
from kedro.pipeline import Pipeline
from .pipelines.feature_engineering.pipeline import (
    create_pipeline as create_fe_pipeline,
)
from .pipelines.data_processing.pipeline import create_pipeline
from .pipelines.model_training.pipeline import (
    create_pipeline as create_training_pipeline,
)


def register_pipelines() -> dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from pipeline names to ``Pipeline`` objects.
    """
    pipelines = find_pipelines()

    pipelines['data_processing_pipeline'] = create_pipeline()
    pipelines['feature_engineering_pipeline'] = create_fe_pipeline()
    pipelines['model_training_pipeline'] = create_training_pipeline()

    return pipelines
