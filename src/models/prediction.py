import polars as pl


def export_prediction(inputs, target_prediction, proba_prediction, set_name):
    df_prediction = pl.concat(
        inputs
        + [
            pl.DataFrame(target_prediction, schema={"prediction": pl.Int64}),
            pl.DataFrame(
                proba_prediction,
                schema={"proba_0": pl.Float64, "proba_1": pl.Float64},
            ),
        ],
        how="horizontal",
    )

    df_prediction.write_csv(
        f"data/processed/xgb_model_prediction_{set_name}.csv", separator=";"
    )

    return df_prediction
