import polars as pl

from fuzzywuzzy import fuzz


def get_token_set_ratio(sdf):
    return (
        sdf.with_columns(pl.struct(pl.col(["left_side", "right_side"])).alias("comb"))
        .with_columns(
            pl.col("comb")
            .apply(lambda df: fuzz.token_set_ratio(df["left_side"], df["right_side"]))
            .alias("token_set_ratio")
        )
        .drop("comb")
    )
