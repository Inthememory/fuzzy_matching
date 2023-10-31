import polars as pl


def save_prediction(label, X, Y, model, model_name, path):
    df_prediction = pl.concat(
        [
            label,
            X,
            Y,
            pl.DataFrame(model.predict(X), schema={"prediction": pl.Int64}),
            pl.DataFrame(
                model.predict_proba(X),
                schema={"proba_0": pl.Float64, "proba_1": pl.Float64},
            ),
        ],
        how="horizontal",
    )
    df_prediction.write_csv(f"{path}{model_name}_prediction.csv", separator=";")
