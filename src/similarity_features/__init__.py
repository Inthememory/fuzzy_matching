import polars as pl
import numpy as np
from loguru import logger
from sentence_transformers import SentenceTransformer, util
from scipy.sparse import csr_matrix
from nltk.corpus import stopwords


from src.preprocessing import (
    concat_brands_slug,
    convert_column_to_list,
    get_brand_classifications_dummies,
    get_brand_classification_words,
    get_brand_without_space,
)
from src.similarity_features.numerical_representation import pca, tfidf
from src.similarity_features.similarity import (
    cosine_similarity_matrix,
    pairwise_similarity,
    get_token_set_ratio,
)

STOPWORDS_LIST = stopwords.words("english") + stopwords.words("french")


def similarity_classification(datasets, classification_levels):
    # Load dataset processed
    dataset = get_brand_classifications_dummies(datasets, classification_levels)
    print("dataset shape : ", dataset.shape)
    print(dataset.head(5))

    # Create numerical representation
    numerical_representation_pca = pca(
        dataset.drop("brand_desc_slug"), n_components=0.80
    )
    print("numerical_representation_pca shape : ", numerical_representation_pca.shape)
    print(numerical_representation_pca)

    # Compute cosine_similarity_matrix
    cosine_similarity_csr = cosine_similarity_matrix(numerical_representation_pca)
    print(cosine_similarity_csr)
    print("shape :", cosine_similarity_csr.shape)
    print("getnnz :", cosine_similarity_csr.getnnz())
    print("count_nonzero :", cosine_similarity_csr.count_nonzero())
    print(cosine_similarity_csr.count_nonzero() / cosine_similarity_csr.getnnz())

    # Compute pairwise_similarity
    df_cossim = pairwise_similarity(
        cosine_similarity_csr, convert_column_to_list(dataset, 0)
    )
    return df_cossim.rename({"similarity": "similarity_classification"})


def similarity_classification_words(datasets, classification_most_relevant_level):
    # Load dataset processed
    dataset = get_brand_classification_words(
        datasets, classification_most_relevant_level
    )
    print("dataset shape : ", dataset.shape)

    # Create numerical representation
    numerical_representation_tfidf = tfidf(
        dataset,
        "level_slug",
        analyzer="word",
        token_pattern=r"(?u)\b[A-Za-z]{2,}\b",
    )
    print(
        "numerical_representation_tfidf shape : ", numerical_representation_tfidf.shape
    )

    numerical_representation_pca = pca(
        numerical_representation_tfidf, n_components=0.80
    )
    print("numerical_representation_pca shape : ", numerical_representation_pca.shape)

    # Compute cosine_similarity_matrix
    cosine_similarity_csr = cosine_similarity_matrix(numerical_representation_pca)
    print("getnnz :", cosine_similarity_csr.getnnz())
    print("count_nonzero :", cosine_similarity_csr.count_nonzero())
    print(cosine_similarity_csr.count_nonzero(), cosine_similarity_csr.getnnz())

    # Compute pairwise_similarity
    df_cossim = pairwise_similarity(
        cosine_similarity_csr, convert_column_to_list(dataset, 0)
    )
    return df_cossim.rename({"similarity": "similarity_classification_words"})


def similarity_semantic(datasets):
    # Load dataset processed
    dataset = concat_brands_slug(datasets)

    model = SentenceTransformer("all-MiniLM-L6-v2")

    sentences = convert_column_to_list(dataset, 0)

    # Encode all sentences
    embeddings = model.encode(sentences)

    cossim = np.zeros((len(sentences), len(sentences)))

    for i in range(len(sentences)):
        cossim[:, i] = util.cos_sim(embeddings[i], embeddings[:])

    df_cossim = pairwise_similarity(csr_matrix(cossim), sentences)
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
    numerical_representation_tfidf = tfidf(
        dataset, "brand_desc_without_space", analyzer="char", ngram_range=(2, 5)
    )
    print(
        "numerical_representation_tfidf shape : ", numerical_representation_tfidf.shape
    )

    numerical_representation_pca = pca(
        numerical_representation_tfidf, n_components=0.70
    )
    print("numerical_representation_pca shape : ", numerical_representation_pca.shape)

    # Compute cosine_similarity_matrix
    cosine_similarity_csr = cosine_similarity_matrix(numerical_representation_pca)
    print("getnnz :", cosine_similarity_csr.getnnz())
    print("count_nonzero :", cosine_similarity_csr.count_nonzero())
    print(cosine_similarity_csr.count_nonzero(), cosine_similarity_csr.getnnz())

    # Compute pairwise_similarity
    df_cossim = pairwise_similarity(
        cosine_similarity_csr, convert_column_to_list(dataset, 0)
    )
    return df_cossim.rename({"similarity": "similarity_syntax_ngram"})


def similarity_syntax_words(datasets):
    # Load dataset processed
    dataset = concat_brands_slug(datasets)
    print("dataset shape : ", dataset.shape)

    # Create numerical representation
    numerical_representation_tfidf = tfidf(
        dataset,
        "brand_desc_slug",
        analyzer="word",
        stopwords_list=STOPWORDS_LIST,
        token_pattern=r"(?u)\b[A-Za-z]{2,}\b",
    )
    print(
        "numerical_representation_tfidf shape : ", numerical_representation_tfidf.shape
    )

    numerical_representation_pca = pca(
        numerical_representation_tfidf, n_components=0.70
    )
    print("numerical_representation_pca shape : ", numerical_representation_pca.shape)

    # Compute cosine_similarity_matrix
    cosine_similarity_csr = cosine_similarity_matrix(numerical_representation_pca)
    print("getnnz :", cosine_similarity_csr.getnnz())
    print("count_nonzero :", cosine_similarity_csr.count_nonzero())
    print(cosine_similarity_csr.count_nonzero(), cosine_similarity_csr.getnnz())

    # Compute pairwise_similarity
    df_cossim = pairwise_similarity(
        cosine_similarity_csr, convert_column_to_list(dataset, 0)
    )
    return df_cossim.rename({"similarity": "similarity_syntax_words"})
