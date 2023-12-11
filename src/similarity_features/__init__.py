import polars as pl
import numpy as np
from sentence_transformers import SentenceTransformer, util
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity

from src.similarity_features.numerical_representation import (
    tfidf,
    convert_column_to_list,
)
from src.similarity_features.similarity import (
    pairwise_similarity,
    get_token_set_ratio,
)


def similarity_classification(dataset: pl.DataFrame, col_label: str) -> pl.DataFrame:
    # Load dataset processed
    print(
        "dataset_dense sparsity :",
        round(
            1.0 - np.count_nonzero(dataset) / (dataset.shape[0] * dataset.shape[1]),
            3,
        ),
    )
    # Convert dataset to Compressed Sparse Row (CSR) format for better performance
    dataset_sparse = csr_matrix(dataset.drop(col_label))

    # Calculate cosine similarity
    # cosine_sim_sparse[i, j] represents the cosine similarity between row i and row j
    cosine_sim_sparse = cosine_similarity(dataset_sparse, dense_output=False)

    # Compute pairwise_similarity
    df_cossim = pairwise_similarity(
        cosine_sim_sparse, convert_column_to_list(dataset, col_label)
    )
    return df_cossim.rename({"similarity": "similarity_classification"})


def similarity_classification_words(
    dataset: pl.DataFrame, col: str, col_label: str
) -> pl.DataFrame:
    # Create numerical representation
    dataset_dense = tfidf(
        dataset, col, analyzer="word", token_pattern=r"(?u)\b[A-Za-z]{2,}\b"
    )
    print("dataset_dense shape : ", dataset_dense.shape)
    print(
        "dataset_dense sparsity :",
        round(
            1.0
            - np.count_nonzero(dataset_dense)
            / (dataset_dense.shape[0] * dataset_dense.shape[1]),
            3,
        ),
    )

    # Convert dataset to Compressed Sparse Row (CSR) format for better performance
    dataset_sparse = csr_matrix(dataset_dense)

    # Calculate cosine similarity
    # cosine_sim_sparse[i, j] represents the cosine similarity between row i and row j
    cosine_sim_sparse = cosine_similarity(dataset_sparse, dense_output=False)

    # Compute pairwise_similarity
    df_cossim = pairwise_similarity(
        cosine_sim_sparse, convert_column_to_list(dataset, col_label)
    )
    return df_cossim.rename({"similarity": "similarity_classification_words"})


def similarity_semantic(
    dataset: pl.DataFrame, col: str, col_label: str
) -> pl.DataFrame:
    model = SentenceTransformer("all-MiniLM-L6-v2")

    sentences = convert_column_to_list(dataset, col)
    # Encode all sentences
    embeddings = model.encode(sentences)

    # Calculate cosine similarity
    # cossim[i, j] represents the cosine similarity between row i and row j
    cossim = np.zeros((len(sentences), len(sentences)))
    for i in range(len(sentences)):
        cossim[:, i] = util.cos_sim(embeddings[i], embeddings[:])

    df_cossim = pairwise_similarity(
        csr_matrix(cossim), convert_column_to_list(dataset, col_label)
    )
    df_cossim = df_cossim.sort(by=["similarity"], descending=True)
    return df_cossim.select(
        "left_side",
        "right_side",
        pl.col("similarity").alias("similarity_semantic"),
    )


def similarity_syntax_ngram(
    dataset: pl.DataFrame, col: str, col_label: str
) -> pl.DataFrame:
    # Create numerical representation
    dataset_dense = tfidf(dataset, col, analyzer="char", ngram_range=(2, 3))
    print("dataset_dense shape : ", dataset_dense.shape)
    print(
        "dataset_dense sparsity :",
        round(
            1.0
            - np.count_nonzero(dataset_dense)
            / (dataset_dense.shape[0] * dataset_dense.shape[1]),
            3,
        ),
    )

    # Convert dataset to Compressed Sparse Row (CSR) format for better performance
    dataset_sparse = csr_matrix(dataset_dense)

    # Calculate cosine similarity
    # cosine_sim_sparse[i, j] represents the cosine similarity between row i and row j
    cosine_sim_sparse = cosine_similarity(dataset_sparse, dense_output=False)

    # Compute pairwise_similarity
    df_cossim = pairwise_similarity(
        cosine_sim_sparse, convert_column_to_list(dataset, col_label)
    )
    return df_cossim.rename({"similarity": "similarity_syntax_ngram"})


def similarity_syntax_words(
    dataset: pl.DataFrame, col: str, col_label: str
) -> pl.DataFrame:
    # Create numerical representation
    dataset_dense = tfidf(
        dataset,
        col,
        analyzer="word",
        token_pattern=r"(?u)\b[A-Za-z]{2,}\b",
    )
    print("dataset_dense shape : ", dataset_dense.shape)
    print(
        "dataset_dense sparsity :",
        round(
            1.0
            - np.count_nonzero(dataset_dense)
            / (dataset_dense.shape[0] * dataset_dense.shape[1]),
            3,
        ),
    )

    # Convert dataset to Compressed Sparse Row (CSR) format for better performance
    dataset_sparse = csr_matrix(dataset_dense)

    # Calculate cosine similarity
    # cosine_sim_sparse[i, j] represents the cosine similarity between row i and row j
    cosine_sim_sparse = cosine_similarity(dataset_sparse, dense_output=False)

    # Compute pairwise_similarity
    df_cossim = pairwise_similarity(
        cosine_sim_sparse, convert_column_to_list(dataset, col_label)
    )
    return df_cossim.rename({"similarity": "similarity_syntax_words"})
