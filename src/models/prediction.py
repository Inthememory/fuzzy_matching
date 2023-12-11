import polars as pl
from typing import Union
import xgboost as xgb
import yaml


def get_predictions(model, dataset: pl.DataFrame, config: yaml):
    X = dataset.select(config["indicators_var"])
    inputs = [X]
    if (
        config["label_var"][0] in dataset.columns
        and config["label_var"][1] in dataset.columns
    ):
        inputs.append(dataset.select(config["label_var"]))
    if config["target_var"] in dataset.columns:
        inputs.append(dataset.select(config["target_var"]))

    df_prediction = pl.concat(
        inputs
        + [
            pl.DataFrame(model.predict(X), schema={"prediction": pl.Int64}),
            pl.DataFrame(
                model.predict_proba(X),
                schema={"proba_0": pl.Float64, "proba_1": pl.Float64},
            ),
        ],
        how="horizontal",
    )
    return df_prediction
