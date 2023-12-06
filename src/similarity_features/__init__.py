import polars as pl
import numpy as np
from loguru import logger
from sentence_transformers import SentenceTransformer, util
from scipy.sparse import csr_matrix
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import cosine_similarity


from src.preprocessing import (
    concat_brands_slug,
    convert_column_to_list,
    get_brand_classifications_dummies,
    get_brand_classification_words,
    get_brand_without_space,
)
from src.similarity_features.numerical_representation import pca, tfidf
from src.similarity_features.similarity import (
    pairwise_similarity,
    get_token_set_ratio,
)
from sklearn.decomposition import TruncatedSVD

STOPWORDS_LIST = stopwords.words("english") + stopwords.words("french")


def similarity_classification(datasets, classification_levels):
    # Load dataset processed
    dataset_dense = get_brand_classifications_dummies(datasets, classification_levels)
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
    dataset_sparse = csr_matrix(dataset_dense.drop("brand_desc_slug"))

    # Calculate cosine similarity
    # cosine_sim_sparse[i, j] represents the cosine similarity between row i and row j
    cosine_sim_sparse = cosine_similarity(dataset_sparse, dense_output=False)

    # Compute pairwise_similarity
    df_cossim = pairwise_similarity(
        cosine_sim_sparse, convert_column_to_list(dataset_dense, 0)
    )
    return df_cossim.rename({"similarity": "similarity_classification"})


def similarity_classification_words(datasets, classification_most_relevant_level):
    # Load dataset processed
    dataset = get_brand_classification_words(
        datasets, classification_most_relevant_level
    )
    print("dataset shape : ", dataset.shape)

    # Create numerical representation
    dataset_dense = tfidf(
        dataset, "level_slug", analyzer="word", token_pattern=r"(?u)\b[A-Za-z]{2,}\b"
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
        cosine_sim_sparse, convert_column_to_list(dataset, 0)
    )
    return df_cossim.rename({"similarity": "similarity_classification_words"})


def similarity_semantic(datasets):
    # Load dataset processed
    dataset = concat_brands_slug(datasets)

    model = SentenceTransformer("all-MiniLM-L6-v2")

    sentences = convert_column_to_list(dataset, 2)

    # Encode all sentences
    embeddings = model.encode(sentences)

    # Calculate cosine similarity
    # cossim[i, j] represents the cosine similarity between row i and row j
    cossim = np.zeros((len(sentences), len(sentences)))
    for i in range(len(sentences)):
        cossim[:, i] = util.cos_sim(embeddings[i], embeddings[:])

    brand_slug = convert_column_to_list(dataset, 0)
    df_cossim = pairwise_similarity(csr_matrix(cossim), brand_slug)
    df_cossim = df_cossim.sort(by=["similarity"], descending=True)
    return df_cossim.select(
        "left_side",
        "right_side",
        pl.col("similarity").alias("similarity_semantic"),
    )


def similarity_syntax_ngram(datasets):
    # Load dataset processed
    dataset = get_brand_without_space(datasets)
    print("dataset shape : ", dataset.shape)

    # Create numerical representation
    dataset_dense = tfidf(
        dataset, "brand_desc_without_space", analyzer="char", ngram_range=(2, 3)
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
        cosine_sim_sparse, convert_column_to_list(dataset, 0)
    )
    return df_cossim.rename({"similarity": "similarity_syntax_ngram"})


def similarity_syntax_words(datasets):
    # Load dataset processed
    dataset = concat_brands_slug(datasets)
    print("dataset shape : ", dataset.shape)

    # Create numerical representation
    dataset_dense = tfidf(
        dataset,
        "brand_desc_slug_reduced_lem",
        analyzer="word",
        stopwords_list=STOPWORDS_LIST,
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
        cosine_sim_sparse, convert_column_to_list(dataset, 0)
    )
    return df_cossim.rename({"similarity": "similarity_syntax_words"})
