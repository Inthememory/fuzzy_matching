import polars as pl
import numpy as np
from fuzzywuzzy import fuzz
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity


def cosine_similarity_matrix(dataset: pl.DataFrame) -> csr_matrix:
    """_summary_

    Args:
        dataset (pl.DataFrame): _description_

    Returns:
        csr_matrix: _description_
    """
    # Convert to sparse matrix (CSR - Compressed Sparse Row)
    concat_matrix = csr_matrix(dataset)
    # Compute cosine similarity matrix
    cossim = cosine_similarity(concat_matrix)
    cossim_csr = csr_matrix(cossim)
    return cossim_csr


def pairwise_similarity(
    sparse_matrix: csr_matrix, items: list, top_n_matches: int = None
) -> pl.DataFrame:
    """_summary_

    Args:
        sparse_matrix (csr_matrix): _description_
        name_vector (list): _description_
        top (_type_, optional): _description_. Defaults to None.

    Returns:
        pl.DataFrame: _description_
    """
    non_zeros = sparse_matrix.nonzero()

    sparserows = non_zeros[0]
    sparsecols = non_zeros[1]

    if top_n_matches:
        nb_matches = top_n_matches
    else:
        nb_matches = sparsecols.size

    left_side, right_side, similarity = [], [], []
    for index in range(0, nb_matches):
        if sparse_matrix.data[index] > 0.2:
            left_side.append(items[sparserows[index]])
            right_side.append(items[sparsecols[index]])
            similarity.append(sparse_matrix.data[index])

    return pl.DataFrame(
        {
            "left_side": left_side,
            "right_side": right_side,
            "similarity": similarity,
        }
    )

    # left_side = np.empty([nb_matches], dtype=object)
    # right_side = np.empty([nb_matches], dtype=object)
    # similarity = np.zeros(nb_matches)

    # for index in range(0, nb_matches):
    #     left_side[index] = items[sparserows[index]]
    #     right_side[index] = items[sparsecols[index]]
    #     similarity[index] = sparse_matrix.data[index]

    # return pl.DataFrame(
    #     {
    #         "left_side": left_side.astype(str),
    #         "right_side": right_side.astype(str),
    #         "similarity": similarity,
    #     }
    # )


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
