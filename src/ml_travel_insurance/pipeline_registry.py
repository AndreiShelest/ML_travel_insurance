"""Project pipelines."""

from kedro.framework.project import find_pipelines
from kedro.pipeline import Pipeline
from .pipelines.data_science.pipeline import create_data_science_pipeline
from .pipelines.data_processing.pipeline import create_data_processing_pipeline


def register_pipelines() -> dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from pipeline names to ``Pipeline`` objects.
    """
    pipelines = find_pipelines()
    pipelines["__default__"] = sum(pipelines.values())

    pipelines['data_processing_pipeline'] = create_data_processing_pipeline()
    pipelines['data_science_pipeline'] = create_data_science_pipeline()

    return pipelines
