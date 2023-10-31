import polars as pl
from itertools import combinations


def pair_same_products(datasets):
    # All possible pairs in List using combinations()
    pairs = list(combinations(datasets, 2))

    datasets_paired = []
    for pair in pairs:
        datasets_paired.append(
            pl.concat(
                [
                    pair[0].select(
                        pl.col("product_id"),
                        pl.col("brand_desc_slug").alias("brand_desc_left"),
                    ),
                    pair[1].select(
                        pl.col("product_id"),
                        pl.col("brand_desc_slug").alias("brand_desc_right"),
                    ),
                ],
                how="align",
            )
            .filter(pl.col("brand_desc_left").is_not_null())
            .filter(pl.col("brand_desc_right").is_not_null())
            .filter(pl.col("brand_desc_left") != pl.col("brand_desc_right"))
            .groupby("brand_desc_left", "brand_desc_right")
            .count()
            .filter(pl.col("count") > 1)
            .select("brand_desc_left", "brand_desc_right")
        )

    datasets_paired_concat = pl.concat(datasets_paired, how="vertical")

    datasets_paired_concat_invert = (
        datasets_paired_concat.with_columns(pl.col("brand_desc_left").alias("tmp"))
        .with_columns(pl.col("brand_desc_right").alias("brand_desc_left"))
        .with_columns(pl.col("tmp").alias("brand_desc_right"))
        .select(pl.col("brand_desc_left"), pl.col("brand_desc_right"))
    )

    return (
        pl.concat(
            [datasets_paired_concat, datasets_paired_concat_invert], how="vertical"
        )
        .unique()
        .select(
            pl.col("brand_desc_left").alias("left_side"),
            pl.col("brand_desc_right").alias("right_side"),
        )
    )


def load_pairs_labeled(path, file_name, sep=";"):
    return pl.read_csv(f"{path}{file_name}.csv", separator=sep)
