import polars as pl
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--brand",
    type=str,
    required=True,
    help="Brand for which a recommendation is sought.",
)
parser.add_argument(
    "--nb_matches",
    type=int,
    required=False,
    default=3,
    help="Number of matches sought",
)

args = parser.parse_args()

brand = args.brand
nb_matches = args.nb_matches


def similarity_top_n(brand: str, n: int) -> list:
    """Returns a list with the n most likely matches of the brand

    Args:
        brand (str): Brand for which a matches are sought.
        n (int, optional): number of matches sought. Defaults to 3.

    Returns:
        list: list of the n most likely matches with the probalibity of similarity value.
    """
    ## Load similarity results
    df_similarity = (
        pl.read_csv(f"data/processed/xgb_model_predictions.csv", separator=";")
        .filter((pl.col("left_side") == brand) | (pl.col("right_side") == brand))
        .sort(["proba_1"], descending=True)
    )
    df_similarity_top_n = (
        df_similarity.limit(n)
        .with_columns(
            pl.when(pl.col("left_side") == brand)
            .then(pl.col("right_side"))
            .when(pl.col("right_side") == brand)
            .then(pl.col("left_side"))
            .otherwise(-1)
            .alias("match")
        )
        .select("match", pl.col("proba_1").round(3).alias("proba_sim"))
    )
    return df_similarity_top_n.rows(named=True)


print(similarity_top_n(brand, nb_matches))
