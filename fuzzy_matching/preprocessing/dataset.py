import polars as pl
from slugify import slugify
import string
import nltk
from itertools import combinations


class Dataset:
    """This class handle retailer dataset

    Args:
        df (pl.DataFrame) : dataframe that contains retailer's data
        retailer (str) : name of the retailer
        nb_levels (int) : number of levels filled
        levels_col (str) : column that contains tht list of levels
        level0_included (list) : list of level0 to include
        level1_excluded (list) : list of level1 to exclude
        replacements_brand (list) : list of replacement rules e.g. [['|', 'or'], ['%', 'percent']]
    """

    def __init__(
        self,
        df: pl.DataFrame,
        retailer: str,
        nb_levels: int,
        levels_col: str,
        level0_included: list = [],
        level1_excluded: list = [],
        replacements_brand: list = [],
    ) -> None:
        self.retailer = retailer
        self.df = df
        self.levels_col = levels_col
        self.nb_levels = nb_levels
        self.level0_included = level0_included
        self.level1_excluded = level1_excluded
        self.replacements_brand = replacements_brand
        self.dataset_preprocessed = None

    def preprocess(self) -> pl.DataFrame:
        """Proceed to a panel of transformations to clean the dataset

        Returns:
            pl.Dataframe: a polars Dataframe preprocessed
        """
        # Select columns
        dataset = self.df.select(
            pl.col("ean").alias("product_id"),
            pl.col("brand").alias("brand_desc"),
            self.levels_col,
        )

        # Explode levels into multiple columns
        dataset = self.expand_levels(dataset, self.levels_col, self.nb_levels)

        # Filter dataset
        dataset = (
            dataset.filter(pl.col("product_id").str.contains("^[0-9]*$"))
            .filter(pl.col("product_id").str.contains("[1-9]+"))
            .filter(pl.col("brand_desc").is_not_null())
        )

        # Transform columns
        dataset = (
            dataset.with_columns(pl.lit(self.retailer).alias("retailer"))
            .with_columns(pl.col("product_id").str.zfill(13))
            .with_columns(
                pl.col("brand_desc").apply(lambda x: Dataset.remove_non_ascii(x))
            )
            .with_columns(pl.col("brand_desc").str.to_uppercase().str.strip())
            .filter(~pl.col("brand_desc").str.contains(self.retailer.upper()))
            .filter(~pl.col("brand_desc").str.contains("AUTRE MARQUE"))
            .with_columns(
                pl.col("brand_desc")
                .apply(
                    lambda x: Dataset.upper_slug(
                        x, replacements=self.replacements_brand
                    )
                )
                .alias("brand_desc_slug")
            )
        )

        # Upper level columns
        dataset = dataset.select(
            ["retailer", "product_id", "brand_desc", "brand_desc_slug"]
            + [
                pl.col(f"level{i}").apply(lambda x: Dataset.upper_slug(x))
                for i in range(self.nb_levels)
            ]
        )

        self.dataset_preprocessed = dataset
        return dataset

    @staticmethod
    def expand_levels(
        df: pl.DataFrame, levels_col: str, nb_levels: int
    ) -> pl.DataFrame:
        """Explode horizontally the column that contains the list of level into several columns

        Args:
            df (pl.DataFrame): _description_
            levels_col (str): column that contains the list of levels
            nb_levels (int): number of levels filled

        Returns:
            pl.DataFrame: a dataframe with the levels split into several columns
        """
        return df.with_columns(
            [
                pl.col(levels_col).list.get(level_id).alias(f"level{level_id}")
                for level_id in range(nb_levels)
            ]
        ).drop(levels_col)

    def filter_dataset(self, unknown_brands: list = []) -> pl.DataFrame:
        """Remove outliers base on levels and brand

        Args:
            unknown_brands (list, optional): list of unknown brands to remove. Defaults to [].

        Returns:
            pl.Dataframe: a dataframe filtered
        """
        dataset_filtered = (
            self.preprocess()
            .filter(~pl.col("level1").is_in(self.level1_excluded))
            .filter(~pl.col("brand_desc_slug").is_in([item for item in unknown_brands]))
            .filter(pl.col("brand_desc_slug").str.contains("[a-zA-Z]+"))
            .filter(pl.col("brand_desc_slug").str.contains("\w{3,}"))
        )
        if len(self.level0_included) > 0:
            dataset_filtered = dataset_filtered.filter(
                pl.col("level0").is_in(self.level0_included)
            )
        return dataset_filtered

    @staticmethod
    def upper_slug(sentence: str, replacements: list = [], separator: str = " ") -> str:
        """proceed to a pannel of transformations : slugify, upper, strip

        Args:
            sentence (str): sentence to update
            replacements (list, optional): list of replacement rules e.g. [['|', 'or'], ['%', 'percent']]. Defaults to [].
            separator (str, optional): separator between words. Defaults to " ".

        Returns:
            str: sentence updated
        """
        return (
            slugify(sentence, replacements=replacements, separator=separator)
            .upper()
            .strip()
        )

    @staticmethod
    def remove_non_ascii(string):
        return string.encode("ascii", "ignore").decode("utf8").casefold()


class DatasetsMerged:
    """This class handle a list of retailers datasets

    Args:
        sdf (list) : a list that contains retailers dataframes
    """

    def __init__(self, sdf: list) -> None:
        self.sdf = sdf
        self.brand_classification_dummy = None
        self.brand_classification_words = None
        self.brands_updated = None

    def get_brand_classification(self, levels: list) -> pl.DataFrame:
        """Create a dataframe containing brands classification

        Args:
            self
            levels (list): classification's levels to extract

        Returns:
            pl.Dataframe: a dataframe gathering all classifications
        """
        return (
            pl.concat(
                [
                    dataset.select(
                        [
                            pl.col("product_id"),
                            pl.col("brand_desc_slug").alias(f"brand_desc_slug_{i}"),
                        ]
                        + [pl.col(f"level{l}").alias(f"level{l}_{i}") for l in levels]
                    )
                    for i, dataset in enumerate(self.sdf)
                ],
                how="align",
            )
            .with_columns(
                pl.concat_list(
                    [f"brand_desc_slug_{i}" for i, _ in enumerate(self.sdf)]
                ).alias("brand_desc_slug")
            )
            .drop([f"brand_desc_slug_{i}" for i, _ in enumerate(self.sdf)])
            .explode("brand_desc_slug")
            .filter(pl.col("brand_desc_slug").is_not_null())
            .drop("product_id")
            .unique()
        )

    def get_brand_classification_dummy(self, levels: list) -> pl.DataFrame:
        """Convert classification variable into dummy variables.

        Args:
            self
            levels (list): classification levels to convert into dummy variables

        Returns:
            pl.DataFrame: polars dataframe
        """
        # Select levels
        brands_classification = self.get_brand_classification(levels)

        # Convert levels into dummy variables.
        brands_classification_dummy = (
            brands_classification.select(
                ["brand_desc_slug"]
                + [
                    pl.col(c)
                    for c in brands_classification.columns
                    if c.startswith("level")
                ]
            )
            .unique()
            .to_dummies(
                [c for c in brands_classification.columns if c.startswith("level")]
            )
        )

        # Agregate at brand level using maximum
        brands_classification_dummy_agregated = (
            brands_classification_dummy.drop(
                [
                    col
                    for col in brands_classification_dummy.columns
                    if col.endswith("null")
                ]
            )
            .groupby("brand_desc_slug")
            .max()
        )
        self.brand_classification_dummy = brands_classification_dummy_agregated
        return brands_classification_dummy_agregated

    def get_brand_classification_words(
        self, levels: list, lemmatizer, list_stopwords: list = []
    ) -> pl.DataFrame:
        """Preprocess level to make comparisons easier

        Args:
            levels (list): levels to process
            lemmatizer (type): lemmatizer
            list_stopwords (list, optional): list of stopwords. Defaults to [].

        Returns:
            pl.Dataframe: _description_
        """
        ## For level in levels list levels describing each brand
        L = []
        for level in levels:
            # Select level to process
            brands_classification_level = self.get_brand_classification([level])
            brands_classification_words_level = (
                brands_classification_level
                # List levels describing each brand
                .with_columns(
                    pl.concat_list(
                        [
                            c
                            for c in brands_classification_level.columns
                            if c.startswith("level")
                        ]
                    ).alias(f"level")
                )
                .select("brand_desc_slug", "level")
                .explode(f"level")
                .filter(pl.col(f"level").is_not_null())
            )
            # Add dataframe to the list of dataframes to merge
            L.append(brands_classification_words_level)

        brands_classification_words = (
            pl.concat(L)
            .groupby("brand_desc_slug")
            .agg(pl.col(f"level"))
            .with_columns(pl.col(f"level").cast(pl.List(pl.Utf8)).list.join(" "))
        )

        # Keep significant words
        brands_classification_words_updated = DatasetsMerged.update_level_col(
            brands_classification_words, lemmatizer, list_stopwords
        )
        self.brand_classification_words = brands_classification_words_updated
        return brands_classification_words_updated

    def extract_brands(
        self,
        lemmatizer,
        list_stopwords: list = [],
        generic_words: list = [],
        replacements: list = [],
    ) -> pl.DataFrame:
        """Concat vertically brand_desc_slug columns.

        Args:
            lemmatizer (_type_): _description_
            list_stopwords (list, optional): _description_. Defaults to [].
            generic_words (list, optional): _description_. Defaults to [].
            replacements (list, optional): _description_. Defaults to [].

        Returns:
            pl.DataFrame: _description_
        """
        # Concat vertically brand_desc_slug
        brands_df = pl.concat(
            [dataset.select(pl.col("brand_desc_slug")) for dataset in self.sdf],
            how="vertical",
        ).unique()

        # Preprocess brand name to make comparisons easier
        brands_df_updated = DatasetsMerged.update_brand_col(
            brands_df, lemmatizer, list_stopwords, generic_words, replacements
        )
        self.brands_updated = brands_df_updated
        return brands_df_updated

    def pair_brands_with_same_products(self) -> pl.DataFrame:
        """Pairs two brands related two the same product.

        Args:
            datasets (list): list of dataframe containg products dans brands

        Returns:
            pl.DataFrame: a polars dataframe reporting pairs of brands
        """
        # All possible pairs in List using combinations()
        pairs = list(combinations(self.sdf, 2))

        # Pairs two brands related two the same product
        datasets_paired = []
        for pair in pairs:
            datasets_paired.append(
                pl.concat(
                    [
                        pair[0].select(
                            pl.col("product_id"),
                            pl.col("brand_desc_slug").alias("brand_desc_left"),
                        ),
                        pair[1].select(
                            pl.col("product_id"),
                            pl.col("brand_desc_slug").alias("brand_desc_right"),
                        ),
                    ],
                    how="align",
                )
                .filter(pl.col("brand_desc_left").is_not_null())
                .filter(pl.col("brand_desc_right").is_not_null())
                .filter(pl.col("brand_desc_left") != pl.col("brand_desc_right"))
                .groupby("brand_desc_left", "brand_desc_right")
                .count()
                .filter(pl.col("count") > 1)
                .select("brand_desc_left", "brand_desc_right")
            )

        datasets_paired_concat = pl.concat(datasets_paired, how="vertical")

        # Invert columns
        datasets_paired_concat_invert = (
            datasets_paired_concat.with_columns(pl.col("brand_desc_left").alias("tmp"))
            .with_columns(pl.col("brand_desc_right").alias("brand_desc_left"))
            .with_columns(pl.col("tmp").alias("brand_desc_right"))
            .select(pl.col("brand_desc_left"), pl.col("brand_desc_right"))
        )

        return (
            pl.concat(
                [datasets_paired_concat, datasets_paired_concat_invert], how="vertical"
            )
            .unique()
            .select(
                pl.col("brand_desc_left").alias("left_side"),
                pl.col("brand_desc_right").alias("right_side"),
            )
        )

    @staticmethod
    def cross_join(df: pl.DataFrame, cols: list) -> pl.DataFrame:
        """Create a dataframe containg all brand pairs
        Args:
            df (pl.DataFrame): dataframe input
            cols (list): columns to keep

        Returns:
            pl.DataFrame: Cartesian Product of the dataframe
        """
        return df.join(
            df,
            how="cross",
        ).select(
            [pl.col(col).alias(f"{col}_left") for col in cols]
            + [pl.col(f"{col}_right") for col in cols]
        )

    def get_nb_products_by_brand(self) -> pl.DataFrame:
        """Create a dataframe containing the number of products by brand sulgified.

        Returns:
            pl.Dataframe: dataframe listing brand slugified and products
        """
        return (
            pl.concat(
                [
                    dataset.select(
                        pl.col("product_id"),
                        pl.col("brand_desc_slug"),
                        pl.lit(f"{i}").alias("retailer_id"),
                    )
                    for i, dataset in enumerate(self.sdf)
                ],
                how="vertical",
            )
            .unique()
            .groupby("brand_desc_slug")
            .agg(
                [
                    pl.count("product_id").alias("count"),
                    pl.min("retailer_id").alias("retailer_id"),
                ]
            )
        )

    @staticmethod
    def deduplicate_sentence(sentence: str) -> str:
        sentence_tokenized = nltk.tokenize.word_tokenize(sentence)
        sentence_deduplicated = list(
            dict.fromkeys([word for word in sentence_tokenized])
        )
        return " ".join(word.upper() for word in sentence_deduplicated)

    @staticmethod
    def sort_sentence(sentence: str) -> str:
        sentence_tokenized = nltk.tokenize.word_tokenize(sentence)
        sentence_tokenized_sorted = sorted(
            sentence_tokenized, key=lambda x: (len(x), x)
        )
        return " ".join(word.upper() for word in sentence_tokenized_sorted)

    @staticmethod
    def lemmatize_sentence(sentence: str, lemmatizer) -> str:
        sentence_tokenized = nltk.tokenize.word_tokenize(sentence)
        sentence_lemmatized = [
            lemmatizer.lemmatize(word) for word in sentence_tokenized
        ]
        return " ".join(word.upper() for word in sentence_lemmatized)

    @staticmethod
    def remove_digit(sentence: str) -> str:
        sentence_w_num = "".join(i for i in sentence if not i.isdigit())
        return sentence_w_num

    @staticmethod
    def remove_generic_words(sentence: str, generic_words: list) -> str:
        sentence_tokenized = nltk.tokenize.word_tokenize(sentence)
        sentence_w_generic_words = [
            word for word in sentence_tokenized if word.lower() not in generic_words
        ]
        return " ".join(word.upper() for word in sentence_w_generic_words)

    @staticmethod
    def remove_punctuation(sentence: str) -> str:
        sentence_w_punct = "".join(
            [i.lower() for i in sentence if i not in string.punctuation]
        )
        return sentence_w_punct

    @staticmethod
    def remove_stopwords(sentence, list_stopwords: list) -> str:
        sentence_tokenized = nltk.tokenize.word_tokenize(sentence)
        sentence_w_stopwords = [
            word for word in sentence_tokenized if word.lower() not in list_stopwords
        ]
        return " ".join(word for word in sentence_w_stopwords)

    @staticmethod
    def update_brand_col(
        df: pl.DataFrame,
        lemmatizer,
        list_stopwords: list,
        generic_words: list,
        replacements: list,
        col: str = "brand_desc_slug",
    ) -> pl.DataFrame:
        """proceed to a pannel of transformations on a column : remove_stopwords, remove_generic_words, lemmatize_sentence, deduplicate_sentence, slugify

        Args:
            df (pl.DataFrame): _description_
            lemmatizer (_type_): lemmatizer
            list_stopwords (list): list of stopwords to remove
            generic_words (list): list of generic words to remove
            replacements (list): list of replacements to performe
            col (str, optional): col. Defaults to "brand_desc_slug".

        Returns:
            pl.DataFrame: a dataframe with transformations on the sepcified column performed
        """
        return (
            df.with_columns(
                pl.col(col)
                .apply(lambda x: DatasetsMerged.remove_stopwords(x, list_stopwords))
                .alias(f"{col}_updated")
            )
            .with_columns(
                pl.col(f"{col}_updated").apply(
                    lambda x: DatasetsMerged.lemmatize_sentence(x, lemmatizer)
                )
            )
            .with_columns(
                pl.col(f"{col}_updated").apply(
                    lambda x: DatasetsMerged.remove_generic_words(x, generic_words)
                )
            )
            .filter(pl.col(f"{col}_updated") != "")
            .with_columns(
                pl.col(f"{col}_updated").apply(
                    lambda x: slugify(x, separator=" ", replacements=replacements)
                )
            )
            .with_columns(
                pl.col(f"{col}_updated").apply(
                    lambda x: DatasetsMerged.deduplicate_sentence(x)
                )
            )
            .with_columns(
                pl.col(f"{col}_updated").apply(
                    lambda x: DatasetsMerged.sort_sentence(x)
                )
            )
            .with_columns(
                pl.col(f"{col}_updated")
                .apply(lambda x: slugify(x, separator=""))
                .alias(f"{col}_updated_w_space")
            )
        )

    @staticmethod
    def update_level_col(
        df: pl.DataFrame, lemmatizer, list_stopwords: list, col: str = "level"
    ) -> pl.DataFrame:
        """proceed to a pannel of transformations on a column : remove_stopwords, remove_digit, lemmatize_sentence, deduplicate_sentence

        Args:
            df (pl.DataFrame): _description_
            lemmatizer (_type_): lemmatizer
            list_stopwords (list): list_stopwords to remove
            col (str, optional): col. Defaults to "level".

        Returns:
            pl.DataFrame: a dataframe with transformations on the sepcified column performed
        """
        return (
            df.with_columns(
                pl.col(col)
                .apply(lambda x: DatasetsMerged.remove_stopwords(x, list_stopwords))
                .alias(f"{col}_updated")
            )
            .with_columns(
                pl.col(f"{col}_updated").apply(lambda x: DatasetsMerged.remove_digit(x))
            )
            .with_columns(
                pl.col(f"{col}_updated").apply(
                    lambda x: DatasetsMerged.lemmatize_sentence(x, lemmatizer)
                )
            )
            .with_columns(
                pl.col(f"{col}_updated").apply(
                    lambda x: DatasetsMerged.deduplicate_sentence(x)
                )
            )
            .with_columns(
                pl.col(f"{col}_updated").apply(
                    lambda x: DatasetsMerged.sort_sentence(x)
                )
            )
        )
