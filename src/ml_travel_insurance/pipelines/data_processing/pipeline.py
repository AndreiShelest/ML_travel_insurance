import pandas as pd
from kedro.pipeline import Pipeline, node, pipeline


def rename_cols(data: pd.DataFrame) -> pd.DataFrame:
    cols = data.columns.values
    rename_map = {col: col.replace(' ', '_').lower() for col in cols}
    rename_map['Commision (in value)'] = 'commision'

    return data.rename(columns=rename_map)


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=rename_cols,
                inputs='travel_insurance_raw',
                outputs='travel_insurance',
                name='rename_cols',
            ),
        ]
    )
