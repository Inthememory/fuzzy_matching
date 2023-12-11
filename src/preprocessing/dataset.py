import polars as pl
from slugify import slugify
import string
import nltk
from itertools import combinations


class Dataset:
    def __init__(
        self,
        df: pl.DataFrame,
        retailer: str,
        nb_levels: int,
        levels_col: str = "crumb",
        level0_included: list = [],
        level1_excluded: list = [],
    ) -> None:
        self.retailer = retailer
        self.df = df
        self.levels_col = levels_col
        self.nb_levels = nb_levels
        self.level0_included = level0_included
        self.level1_excluded = level1_excluded

    def clean_dataset(self):
        return (
            self.expand_levels(
                self.df.select(
                    pl.col("ean").alias("product_id"),
                    pl.col("brand_name").alias("brand_desc"),
                    "crumb",
                ),
                self.levels_col,
                self.nb_levels,
            )
            .filter(pl.col("product_id").str.contains("^[0-9]*$"))
            .filter(pl.col("product_id").str.contains("[1-9]+"))
            .filter(pl.col("brand_desc").is_not_null())
            .select(
                [
                    pl.lit(self.retailer).alias("retailer"),
                    pl.col("product_id").str.zfill(13).alias("product_id"),
                    pl.col("brand_desc").str.to_uppercase().str.strip(),
                    pl.col("brand_desc")
                    .apply(lambda x: Dataset.upper_slug(x, separator=" "))
                    .alias("brand_desc_slug"),
                ]
                + [
                    pl.col(f"level{i}").apply(
                        lambda x: Dataset.upper_slug(x, separator=" ")
                    )
                    for i in range(self.nb_levels)
                ]
            )
        )

    @staticmethod
    def expand_levels(df, levels_col, nb_levels):
        return df.with_columns(
            [
                pl.col(levels_col).list.get(level_id).alias(f"level{level_id}")
                for level_id in range(nb_levels)
            ]
        ).drop(levels_col)

    def filter_dataset(self, unknown_brands):
        dataset_filtered = (
            self.clean_dataset()
            .filter(~pl.col("level1").is_in(self.level1_excluded))
            .filter(
                ~pl.col("brand_desc_slug").str.contains(
                    "|".join(item for item in unknown_brands)
                )
            )
            .filter(pl.col("brand_desc_slug").str.contains("[a-zA-Z]+"))
        )
        if len(self.level0_included) > 0:
            dataset_filtered = dataset_filtered.filter(
                pl.col("level0").is_in(self.level0_included)
            )
        return dataset_filtered

    @staticmethod
    def upper_slug(sentence: str, separator: str = " ") -> str:
        return (
            slugify(sentence, replacements=[["&", "et"]], separator=separator)
            .upper()
            .strip()
        )


class DatasetsMerged:
    def __init__(self, sdf: list) -> None:
        self.sdf = sdf

    def get_brand_classification(self, levels: list) -> pl.DataFrame:
        """Create a dataframe containing brands classification

        Args:
            self
            levels (list): classification's levels to extract

        Returns:
            pl.Dataframe:
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
        brands_classification = self.get_brand_classification(levels)

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
        return brands_classification_dummy_agregated

    def get_brand_classification_words(
        self, level, lemmatizer, list_stopwords: list = []
    ):
        brands_classification = self.get_brand_classification([level])
        brands_classification_words = (
            brands_classification.with_columns(
                pl.concat_list(
                    [c for c in brands_classification.columns if c.startswith("level")]
                ).alias(f"level")
            )
            .select("brand_desc_slug", "level")
            .explode(f"level")
            .filter(pl.col(f"level").is_not_null())
            .groupby("brand_desc_slug")
            .agg(pl.col(f"level"))
            .with_columns(pl.col(f"level").cast(pl.List(pl.Utf8)).list.join(" "))
        )

        brands_classification_words_updated = DatasetsMerged.update_level_col(
            brands_classification_words, lemmatizer, list_stopwords
        )
        return brands_classification_words_updated

    def extract_brands(
        self, lemmatizer, list_stopwords: list = [], generic_words: list = []
    ) -> pl.DataFrame:
        """Concat verticaly brand_desc_slug columns.

        Args:
            lemmatizer (_type_): _description_
            list_stopwords (list, optional): _description_. Defaults to [].
            generic_words (list, optional): _description_. Defaults to [].

        Returns:
            pl.DataFrame: _description_
        """
        brands_df = pl.concat(
            [dataset.select(pl.col("brand_desc_slug")) for dataset in self.sdf],
            how="vertical",
        ).unique()
        brands_df_updated = DatasetsMerged.update_brand_col(
            brands_df, lemmatizer, list_stopwords, generic_words
        )
        return brands_df_updated

    @staticmethod
    def deduplicate_sentence(sentence: str) -> str:
        sentence_tokenized = nltk.tokenize.word_tokenize(sentence)
        sentence_deduplicated = list(
            dict.fromkeys([word for word in sentence_tokenized])
        )
        return " ".join(word.upper() for word in sentence_deduplicated)

    @staticmethod
    def update_brand_col(
        df: pl.DataFrame,
        lemmatizer,
        list_stopwords: list,
        generic_words: list,
        col: str = "brand_desc_slug",
    ) -> pl.DataFrame:
        return (
            df.with_columns(
                pl.col(col)
                .apply(lambda x: DatasetsMerged.remove_stopwords(x, list_stopwords))
                .alias(f"{col}_updated")
            )
            .with_columns(
                pl.col(f"{col}_updated").apply(
                    lambda x: DatasetsMerged.remove_generic_words(x, generic_words)
                )
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
                pl.col(f"{col}_updated")
                .apply(lambda x: slugify(x, separator=""))
                .alias(f"{col}_updated_w_space")
            )
        )

    @staticmethod
    def update_level_col(
        df: pl.DataFrame, lemmatizer, list_stopwords: list, col: str = "level"
    ) -> pl.DataFrame:
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
        )

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
    def remove_stopwords(sentence, list_stopwords: str) -> str:
        sentence_tokenized = nltk.tokenize.word_tokenize(sentence)
        sentence_w_stopwords = [
            word for word in sentence_tokenized if word.lower() not in list_stopwords
        ]
        return " ".join(word for word in sentence_w_stopwords)

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
