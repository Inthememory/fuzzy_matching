import polars as pl


def get_prediction(df_for_prediction, indicators_var, label_var, target_var, model):
    label = df_for_prediction.select(label_var)
    X = df_for_prediction.select(indicators_var)
    inputs = [label, X]
    if target_var in df_for_prediction.columns:
        Y = df_for_prediction.select(target_var)
        inputs.append(Y)

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
