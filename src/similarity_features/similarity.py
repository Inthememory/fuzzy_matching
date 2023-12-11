import polars as pl
import numpy as np
from fuzzywuzzy import fuzz
from scipy.sparse import csr_matrix


def pairwise_similarity(
    cosine_sim_sparse: csr_matrix, items: list, top_n_matches: int = None
) -> pl.DataFrame:
    """_summary_

    Args:
        cosine_sim_sparse (csr_matrix): _description_
        items (list): _description_
        top_n_matches (_type_, optional): _description_. Defaults to None.

    Returns:
        pl.DataFrame: _description_
    """

    non_zeros = cosine_sim_sparse.nonzero()
    sparserows = non_zeros[0]
    sparsecols = non_zeros[1]

    if top_n_matches:
        nb_matches = top_n_matches
    else:
        nb_matches = sparsecols.size

    left_side = np.empty([nb_matches], dtype=object)
    right_side = np.empty([nb_matches], dtype=object)
    similarity = np.zeros(nb_matches)

    for index in range(0, nb_matches):
        left_side[index] = items[sparserows[index]]
        right_side[index] = items[sparsecols[index]]
        similarity[index] = cosine_sim_sparse.data[index]

    return pl.DataFrame(
        {
            "left_side": left_side.astype(str),
            "right_side": right_side.astype(str),
            "similarity": similarity,
        }
    )


def get_token_set_ratio(dataset: pl.DataFrame) -> pl.DataFrame:
    """_summary_

    Args:
        dataset (pl.DataFrame): _description_

    Returns:
        pl.DataFrame: _description_
    """
    return (
        dataset.with_columns(
            pl.struct(pl.col(["left_side", "right_side"])).alias("comb")
        )
        .with_columns(
            pl.col("comb")
            .apply(lambda df: fuzz.token_set_ratio(df["left_side"], df["right_side"]))
            .alias("token_set_ratio")
        )
        .drop("comb")
    )
