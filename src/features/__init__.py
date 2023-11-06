import polars as pl
import numpy as np

from loguru import logger

from sentence_transformers import SentenceTransformer, util
from scipy.sparse import csr_matrix

from src.data import (
    datasets_merged_vertical,
    get_items_list,
    get_brand_classification,
    get_brand_classification_words,
    get_brand_without_space,
)
from src.features.build_numerical_representation import pca, tfidf
from src.features.build_features import (
    cosine_similarity_matrix,
    pairwise_similarity,
    get_token_set_ratio,
)


def classification(datasets, classification_levels):
    # Load dataset processed
    dataset = get_brand_classification(datasets, classification_levels)

    # Create numerical representation
    numerical_representation_pca = pca(
        dataset.drop("brand_desc_slug"), n_components=0.80
    )

    # Compute cosine_similarity_matrix
    cosine_similarity_csr = cosine_similarity_matrix(numerical_representation_pca)

    # Compute pairwise_similarity
    df_cossim = pairwise_similarity(cosine_similarity_csr, get_items_list(dataset, 0))
    return df_cossim


def classification_words(datasets, classification_most_relevant_level, stopwords_list):
    # Load dataset processed
    dataset = get_brand_classification_words(
        datasets, classification_most_relevant_level, stopwords_list
    )

    # Create numerical representation
    numerical_representation_tfidf = tfidf(
        dataset,
        "level_slug",
        analyzer="word",
        stopwords_list=stopwords_list,
        token_pattern=r"(?u)\b[A-Za-z]{2,}\b",
    )
    numerical_representation_pca = pca(
        numerical_representation_tfidf, n_components=0.80
    )

    # Compute cosine_similarity_matrix
    cosine_similarity_csr = cosine_similarity_matrix(numerical_representation_pca)

    # Compute pairwise_similarity
    df_cossim = pairwise_similarity(cosine_similarity_csr, get_items_list(dataset, 0))
    return df_cossim


def sentence_transformer(datasets):
    # Load dataset processed
    dataset = datasets_merged_vertical(datasets)

    model = SentenceTransformer("all-MiniLM-L6-v2")

    sentences = get_items_list(dataset, 0)

    # Encode all sentences
    embeddings = model.encode(sentences)

    cossim = np.zeros((len(sentences), len(sentences)))

    for i in range(len(sentences)):
        cossim[:, i] = util.cos_sim(embeddings[i], embeddings[:])

    df_cossim = pairwise_similarity(csr_matrix(cossim), sentences)
    df_cossim = df_cossim.sort(by=["similarity"], descending=True)
    return df_cossim.select("left_side", "right_side", "similarity")


def syntax_ngram(datasets):
    # Load dataset processed
    dataset = get_brand_without_space(datasets)

    # Create numerical representation
    numerical_representation_tfidf = tfidf(
        dataset, "brand_desc_without_space", analyzer="char", ngram_range=(2, 5)
    )
    numerical_representation_pca = pca(
        numerical_representation_tfidf, n_components=0.70
    )

    # Compute cosine_similarity_matrix
    cosine_similarity_csr = cosine_similarity_matrix(numerical_representation_pca)

    # Compute pairwise_similarity
    df_cossim = pairwise_similarity(cosine_similarity_csr, get_items_list(dataset, 0))
    return df_cossim


def syntax_words(datasets, stopwords_list):
    # Load dataset processed
    dataset = datasets_merged_vertical(datasets)

    # Create numerical representation
    numerical_representation_tfidf = tfidf(
        dataset,
        "brand_desc_slug",
        analyzer="word",
        stopwords_list=stopwords_list,
        token_pattern=r"(?u)\b[A-Za-z]{2,}\b",
    )
    numerical_representation_pca = pca(
        numerical_representation_tfidf, n_components=0.70
    )

    # Compute cosine_similarity_matrix
    cosine_similarity_csr = cosine_similarity_matrix(numerical_representation_pca)

    # Compute pairwise_similarity
    df_cossim = pairwise_similarity(cosine_similarity_csr, get_items_list(dataset, 0))
    return df_cossim
