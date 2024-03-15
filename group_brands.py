import polars as pl
import sys
from loguru import logger
import os

from fuzzy_matching import group_similar_strings, add_master_brand


# Initialise paths
DATA_PROCESSED_PATH = "data/processed/"

# Initialise logs format
logger.remove(0)
logger.add(
    sys.stderr,
    format="{time} | {level} | {message} | {extra}",
)

if __name__ == "__main__":
    logger.debug("Load data")

    # Load pair brands with similar products if more than two datasets
    if os.path.exists(f"{DATA_PROCESSED_PATH}brands_with_same_products_paired.csv"):
        brands_with_same_products_paired = pl.read_csv(
            f"{DATA_PROCESSED_PATH}brands_with_same_products_paired.csv", separator=";"
        )
    else:
        schema = {"left_side": pl.Utf8, "right_side": pl.Utf8}
        brands_with_same_products_paired = pl.DataFrame(schema=schema)

    # Load predictions
    if os.path.exists(f"{DATA_PROCESSED_PATH}xgb_model_predictions.csv"):
        predictions = pl.read_csv(
            f"{DATA_PROCESSED_PATH}xgb_model_predictions.csv", separator=";"
        )
    else:
        raise ValueError(f"Predictions does not exist")

    # Load nb_products_by_brand
    if os.path.exists(f"{DATA_PROCESSED_PATH}nb_products_by_brand.csv"):
        nb_products_by_brand = pl.read_csv(
            f"{DATA_PROCESSED_PATH}nb_products_by_brand.csv", separator=";"
        )
    else:
        raise ValueError(f"nb_products_by_brand does not exist")

    # Load existing groups
    if os.path.exists(f"{DATA_PROCESSED_PATH}res_group_full.csv"):
        groups_init = pl.read_csv(
            f"{DATA_PROCESSED_PATH}res_group_full.csv", separator=";"
        )
    else:
        groups_init = None

    logger.debug("Create dataset")
    df_input_groups = (
        pl.concat(
            [
                predictions.select(
                    pl.col("left_side"),
                    pl.col("right_side"),
                    pl.col("proba_1").cast(float).alias("similarity"),
                ),
                brands_with_same_products_paired.select(
                    pl.col("left_side"),
                    pl.col("right_side"),
                    pl.lit(1.0).alias("similarity"),
                ),
            ]
        )
        .groupby(pl.col("left_side"), pl.col("right_side"))
        .agg(pl.col("similarity").max())
    )

    logger.debug("Groups similar brands")
    res_group = group_similar_strings(
        df_input_groups, groups_init=groups_init, min_similarity=0.8
    ).sort(by="group_name")

    logger.info(
        f"Nb brands : {res_group.shape[0]}, nb groups : {res_group.select('group_name').unique().shape[0]}"
    )

    logger.debug("Select a master brand")
    res_group_with_master = add_master_brand(nb_products_by_brand, res_group)

    # Write results
    res_group_with_master.write_csv(
        f"{DATA_PROCESSED_PATH}res_group_full.csv", separator=";"
    )
