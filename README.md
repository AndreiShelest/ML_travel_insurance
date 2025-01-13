# Travel Insurance

[![Powered by Kedro](https://img.shields.io/badge/powered_by-kedro-ffc900?logo=kedro)](https://kedro.org)

## Overview

Project is based on dataset https://www.kaggle.com/datasets/mhdzahier/travel-insurance/data

To see the report, check the notebook `notebooks/report.ipynb`. It is an aggregation of other notebooks, extended with comments and other information.

Pipelines:

- feature_engineering
- model_training, with nodes:
    - train_logit
    - train_xgboost
    - train_lgbm
    - train_nn