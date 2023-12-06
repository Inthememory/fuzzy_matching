import polars as pl
import numpy as np
from fuzzywuzzy import fuzz
from scipy.sparse import csr_matrix
from src.preprocessing import remove_generic_words

GENERIC_WORDS = [
    "classique",
    "original",
    "origine",
    "spécial",
    "selection",
    "artisanal",
    "brasserie",
    "conserverie",
    "biscuiterie",
    "confiserie",
    "laboratoire",
    "fromagerie",
    "charcuterie",
    "domaine",
    "chateau",
    "maison",
    "casa",
    "creperie",
    "ferme",
    "vergers",
    "cafe",
    "salaisons",
    "moulin",
    "traiteur",
    "rucher",
    "jardin",
    "pasta",
    "saveurs",
    "delices",
    "petit",
    "saint",
    "gourmand",
    "gourmet",
    "bio",
    "pere",
    "france",
    "equitable",
]


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

    # left_side, right_side, similarity = [], [], []
    # for index in range(0, nb_matches):
    #     if sparse_matrix.data[index] > 0.2:
    #         left_side.append(items[sparserows[index]])
    #         right_side.append(items[sparsecols[index]])
    #         similarity.append(sparse_matrix.data[index])

    # return pl.DataFrame(
    #     {
    #         "left_side": left_side,
    #         "right_side": right_side,
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
            pl.col("left_side")
            .apply(lambda x: remove_generic_words(x))
            .alias("left_side_reduced")
        )
        .with_columns(
            pl.col(f"left_side_reduced").cast(pl.List(pl.Utf8)).list.join(" ")
        )
        .with_columns(
            pl.col("right_side")
            .apply(lambda x: remove_generic_words(x))
            .alias("right_side_reduced")
        )
        .with_columns(
            pl.col(f"right_side_reduced").cast(pl.List(pl.Utf8)).list.join(" ")
        )
        .with_columns(
            pl.struct(pl.col(["left_side_reduced", "right_side_reduced"])).alias("comb")
        )
        .with_columns(
            pl.col("comb")
            .apply(
                lambda df: fuzz.token_set_ratio(
                    df["left_side_reduced"], df["right_side_reduced"]
                )
            )
            .alias("token_set_ratio")
        )
        .drop("comb", "left_side_reduced", "right_side_reduced")
    )
