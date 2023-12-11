import polars as pl
from sklearn.feature_extraction.text import TfidfVectorizer


def convert_column_to_list(df: pl.DataFrame, col: int) -> list:
    """Convert column of a polars dataframe to a python list

    Args:
        df (pl.Dataframe): polars dataframe containg the column to extract
        col_id (int): id of the column to convert

    Returns:
        list: list extract from the dataframe
    """
    return df.select(col).to_series().to_list()


def tfidf(
    dataset, col, analyzer, stopwords_list=None, token_pattern=None, ngram_range=(1, 1)
):
    """_summary_

    Args:
        dataset (_type_): _description_
        col (_type_): _description_
        analyzer (_type_): _description_
        stopwords_list (_type_, optional): _description_. Defaults to None.
        token_pattern (_type_, optional): _description_. Defaults to None.
        ngram_range (tuple, optional): _description_. Defaults to (1, 1).

    Returns:
        _type_: _description_
    """
    # Grab the column to tokenise
    dataset = dataset[col]

    # Generate the matrix of TF-IDF values for each item
    vectorizer = TfidfVectorizer(
        stop_words=stopwords_list,
        analyzer=analyzer,
        token_pattern=token_pattern,
        ngram_range=ngram_range,
    )
    tf_idf_matrix = vectorizer.fit_transform(dataset)

    # Get output feature names
    tfidf_tokens = vectorizer.get_feature_names_out()

    # Create dataframe
    df_tfidfvect = pl.DataFrame(
        data=tf_idf_matrix.toarray(), schema=tfidf_tokens.tolist()
    )

    return df_tfidfvect
