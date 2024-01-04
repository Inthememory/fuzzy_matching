import polars as pl
import numpy as np
from tqdm import tqdm
from typing import Union

from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from fuzzywuzzy import fuzz
from fuzzy_matching.distances import DiscountedLevenshtein, FuzzyWuzzyTokenSort


class Similarity:    
    """This class handle similarity computation

    Args:
        dataset (pl.dataframe) : 
        name (str) : name of the similarity
        label_col (str) : name of the column that contains brand desc
        col (Union[str, None]) : name of the column to use to compute similarity
        tfidf_required (bool) : True if tfidf is required, else False
    """

    def __init__(
        self,
        dataset: pl.DataFrame,
        name: str,
        label_col: str,
        col: Union[str, None] = None,
        tfidf_required: bool = False,
    ) -> None:
        self.dataset = dataset
        self.name = name
        self._label_col = label_col
        self._col = col
        self.tfidf_required = tfidf_required
        self._analyzer = "word"
        self._stopwords_list = None
        self._token_pattern = None
        self._ngram_range = (1, 1)

        self.sparse_dataset = None
        self.tfidf_dataset = None
        self.cossim_dataset = None
        self.labels = self.dataset.select(self.label_col).to_series().to_list()
        self.pairwise_dataset = None

    @property
    def col(self):
        return self._col

    @col.setter
    def col(self, value):
        if type(value) == str and value in self.dataset.columns:
            self._col = value
        else:
            raise TypeError(f"str expected, got '{type(value).__name__}'")

    @property
    def label_col(self):
        return self._label_col

    @label_col.setter
    def label_col(self, value):
        if type(value) == str:
            if value in self.dataset.columns:
                self._label_col = value
                self.labels = self.dataset.select(value).to_series().to_list()
            else:
                raise ValueError("label_col must be a column of the dataset.")
        else:
            raise TypeError(f"str expected, got '{type(value).__name__}'")

    @property
    def analyzer(self):
        return self._analyzer

    @analyzer.setter
    def analyzer(self, value):
        if type(value) == str:
            if value in value in ["word", "char", "char_wb"]:
                self._analyzer = value
            else:
                raise ValueError('"word", "char" or "char_wb" expected')
        else:
            raise TypeError(f"str expected, got '{type(value).__name__}'")

    @property
    def stopwords_list(self):
        return self._stopwords_list

    @stopwords_list.setter
    def stopwords_list(self, value):
        if type(value) == list:
            self._stopwords_list = value
        else:
            raise TypeError(f"list expected, got '{type(value).__name__}'")

    @property
    def ngram_range(self):
        return self._ngram_range

    @ngram_range.setter
    def ngram_range(self, value):
        if type(value) == tuple:
            self._ngram_range = value
        else:
            raise TypeError(f"tuple expected, got '{type(value).__name__}'")

    @property
    def token_pattern(self):
        return self._token_pattern

    @token_pattern.setter
    def token_pattern(self, value):
        self._token_pattern = value

    def cos_sim(
        self,
        dense_output: bool = False,
        pairwise: bool = True,
        min_similarity: float = 0.0,
    )->Union[np.array, csr_matrix]:
        """Calculate cosine similarity.
           cosine_sim_sparse[i, j] represents the cosine similarity between row i and row j.

        Args:
            dense_output (bool, optional): Whether to return dense output even when the input is sparse. If False, the output is sparse if both input arrays are sparse. Defaults to False.
            pairwise (bool, optional): Compute pairwise dataset if True. Defaults to False.

        Returns:
            Union[np.array, csr_matrix]: Returns the cosine similarity between all samples in sparse_dataset.
        """
        # Create a sparse matrix
        self._init_sparse_matrix()

        # Compute  pairwise cosine similarity between samples all samples in the sparse dataset
        self.cossim_dataset = cosine_similarity(
            self.sparse_dataset, dense_output=dense_output
        )
        # If pairwise is True, turn ndarray cossim_dataset into dataframe and add labels
        if pairwise:
            self._init_pairwise_dataset(min_similarity)
        return self.cossim_dataset

    def _init_sparse_matrix(self)->Union[np.array, csr_matrix]:
        """Compressed as sparse matrix

        Returns:
            Union[np.array, csr_matrix]: a sparse matrix
        """
        # Create sparse_dataset if self.sparse_dataset hasn't already been instantiated
        if not self.sparse_dataset:
            # If required, create dataframe of TF-IDF features
            if self.tfidf_required:
                if self.tfidf_dataset is None:
                    self._init_tfidf_dataset()
                # Convert into a sparse matrix
                self.sparse_dataset = csr_matrix(self.tfidf_dataset)
            else:
                self.sparse_dataset = csr_matrix(self.dataset.drop(self._label_col))
        return self.sparse_dataset

    def _init_tfidf_dataset(self)->pl.dataframe:
        """Convert a collection of string to a dataframe of TF-IDF features.
        Returns:
            pl.dataframe: dataframe of TF-IDF features.
        """
        # Grab the column to tokenise
        dataset = self.dataset[self.col]

        # Generate the matrix of TF-IDF values for each item
        vectorizer = TfidfVectorizer(
            stop_words=self._stopwords_list,
            analyzer=self._analyzer,
            token_pattern=self._token_pattern,
            ngram_range=self._ngram_range,
        )
        tf_idf_matrix = vectorizer.fit_transform(dataset)

        # Get output feature names
        tfidf_tokens = vectorizer.get_feature_names_out()

        # Create dataframe
        df_tfidfvect = pl.DataFrame(
            data=tf_idf_matrix.toarray(), schema=tfidf_tokens.tolist()
        )

        self.tfidf_dataset = df_tfidfvect.select(sorted(df_tfidfvect.columns))
        return self.tfidf_dataset

    def _init_pairwise_dataset(self, min_similarity:float) -> pl.DataFrame:
        """Convert a sparse matrix containing probability of similarity into dataframe with labels and probability of similarity above a threshold

        Args:
            min_similarity (float): threshold

        Raises:
            ValueError: raise error if cossim_dataset and labels aren't instantiated

        Returns:
            pl.DataFrame: dataframe with pairs of string with probability of similarity above a threshold
        """
        if self.cossim_dataset is not None and self.labels is not None:
            sparserows = self.cossim_dataset.nonzero()[0]
            sparsecols = self.cossim_dataset.nonzero()[1]
            nb_matches = sparsecols.size

            left_side, right_side, similarity = [], [], []

            for index in range(0, nb_matches):
                if (
                    sparserows[index] > sparsecols[index]
                    and self.cossim_dataset.data[index] > min_similarity
                ):
                    if self.labels[sparserows[index]] > self.labels[sparsecols[index]]:
                        left_side.append(self.labels[sparserows[index]])
                        right_side.append(self.labels[sparsecols[index]])
                        similarity.append(self.cossim_dataset.data[index])
                    else:
                        right_side.append(self.labels[sparserows[index]])
                        left_side.append(self.labels[sparsecols[index]])
                        similarity.append(self.cossim_dataset.data[index])

            self.pairwise_dataset = pl.DataFrame(
                {
                    "left_side": left_side,
                    "right_side": right_side,
                    f"similarity_{self.name}": similarity,
                }
            )
            return self.pairwise_dataset
        else:
            raise ValueError(f"self.cossim_dataset and self.labels must be instantiated")

    def sparsity(self)->float:
        """Returns the sparsity of a dataset (the proportion of zero-value elements)

        Raises:
            ValueError: raise error if tfidf_dataset isn't instantiated

        Returns:
            float: sparsity of the dataset
        """
        if self.tfidf_required:
            if self.tfidf_dataset is not None:
                df = self.tfidf_dataset
            else:
                raise ValueError("tfidf_dataset must be instantiated")
        else:
            df = self.dataset
        return round(1.0 - np.count_nonzero(df) / (df.shape[0] * df.shape[1]), 3)

    @staticmethod
    def distance_metrics(
        dataset: pl.DataFrame, col_left: str, col_right: str
    ) -> pl.DataFrame:
        """This function computes string distance measures between two columns.

        Args:
            dataset (pl.DataFrame): dataset that contains string to compare
            col_left (str) : name of the left column for comparison
            col_right (str) : name of the right column for comparison

        Returns:
            pl.DataFrame: dataset with discounted_levenshtein and partial_ratio string distance measures
        """
        return (
            dataset
            .with_columns(pl.struct(pl.col([col_left, col_right])).alias("comb"))
            .with_columns(
                pl.col("comb")
                .apply(lambda df: fuzz.partial_ratio(df[col_left], df[col_right]) / 100)
                .alias("fuzzy_ratio")
            )
            .with_columns(
                pl.col("comb")
                .apply(lambda df: DiscountedLevenshtein().sim(df[col_left], df[col_right]))
                .alias("discounted_levenshtein")
            )
            .drop("comb")
        )
