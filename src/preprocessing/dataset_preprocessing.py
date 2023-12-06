import polars as pl
import string
from slugify import slugify
from nltk.corpus import stopwords
from itertools import combinations
from french_lefff_lemmatizer.french_lefff_lemmatizer import FrenchLefffLemmatizer
from src.preprocessing import french_preprocess_sentence

STOPWORDS_LIST = stopwords.words("english") + stopwords.words("french")
GENERIC_WORDS = [
    "classique",
    "original",
    "origine",
    "special",
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


def remove_generic_words(string):
    tokens = list(string.split(" "))
    return list(
        dict.fromkeys([token for token in tokens if token.lower() not in GENERIC_WORDS])
    )


def get_classifications_by_brand(datasets: list, levels: list) -> pl.DataFrame:
    """Create a dataframe contaning all retailer's classifications for each brand slugified

    Args:
        datasets (list): list of datasets containing products, brands and classifications
        levels (list): nb of levels to extract from classifications

    Returns:
        pl.Dataframe: dataframe listing retailer's classifications for each brand slugified
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
                for i, dataset in enumerate(datasets)
            ],
            how="align",
        )
        .with_columns(
            pl.concat_list(
                [f"brand_desc_slug_{i}" for i, _ in enumerate(datasets)]
            ).alias("brand_desc_slug")
        )
        .drop([f"brand_desc_slug_{i}" for i, _ in enumerate(datasets)])
        .explode("brand_desc_slug")
        .filter(pl.col("brand_desc_slug").is_not_null())
        .drop("product_id")
        .unique()
    )


def concat_brands_slug(datasets: list) -> pl.DataFrame:
    """Concat verticaly brand_desc_slug columns

    Args:
        datasets (list): list of datasets containing products and brands

    Returns:
        pl.Dataframe: dataframe gathering brands slugified
    """
    return (
        pl.concat(
            [dataset.select(pl.col("brand_desc_slug")) for dataset in datasets],
            how="vertical",
        )
        .unique()
        .with_columns(
            pl.col("brand_desc_slug")
            .apply(lambda x: remove_generic_words(x))
            .alias("brand_desc_slug_reduced")
        )
        .with_columns(
            pl.col(f"brand_desc_slug_reduced").cast(pl.List(pl.Utf8)).list.join(" ")
        )
        .with_columns(
            pl.col("brand_desc_slug_reduced")
            .apply(lambda x: french_preprocess_sentence(x, with_digit=True))
            .alias("brand_desc_slug_reduced_lem")
        )
        .select(
            "brand_desc_slug", "brand_desc_slug_reduced", "brand_desc_slug_reduced_lem"
        )
    )


def convert_column_to_list(df: pl.DataFrame, col_id: int) -> list:
    """Convert column of a polars dataframe to a python list

    Args:
        df (pl.Dataframe): polars dataframe containg the column to extract
        col_id (int): id of the column to convert

    Returns:
        list: list extract from the dataframe
    """
    return df.get_columns()[col_id].to_list()


def get_brand_classifications_dummies(
    datasets: list, classification_levels: list
) -> pl.DataFrame:
    """Convert classification variable into dummy variables.

    Args:
        datasets (list): list of dataframe contaning brands dans classifications
        classification_levels (list): classification levels to convert into dummy variables

    Returns:
        pl.DataFrame: polars dataframe
    """
    product_classification = get_classifications_by_brand(
        datasets, classification_levels
    )

    product_classification_dummies = (
        product_classification.select(
            ["brand_desc_slug"]
            + [
                pl.col(c)
                for c in product_classification.columns
                if c.startswith("level")
            ]
        )
        .unique()
        .to_dummies(
            [c for c in product_classification.columns if c.startswith("level")]
        )
    )
    brand_classification = (
        product_classification_dummies.drop(
            [
                col
                for col in product_classification_dummies.columns
                if col.endswith("null")
            ]
        )
        .groupby("brand_desc_slug")
        .max()
    )
    return brand_classification


def get_brand_classification_words(
    datasets: list, classification_most_relevant_level: int
):
    product_classification = get_classifications_by_brand(
        datasets, [classification_most_relevant_level]
    )

    brand_classification = (
        product_classification.select(
            ["brand_desc_slug"]
            + [
                pl.col(c)
                for c in product_classification.columns
                if c.startswith("level")
            ]
        )
        .unique()
        .with_columns(
            pl.concat_list(
                [c for c in product_classification.columns if c.startswith("level")]
            ).alias(f"level")
        )
        .select("brand_desc_slug", "level")
        .explode(f"level")
        .filter(pl.col(f"level").is_not_null())
        .unique()
    )

    brand_classificationWords = (
        brand_classification.groupby("brand_desc_slug")
        .agg(pl.col(f"level"))
        .with_columns(
            pl.col(f"level").cast(pl.List(pl.Utf8)).list.join(" ").alias("level_slug")
        )
    )
    return brand_classificationWords


def get_brand_without_space(datasets: list) -> pl.DataFrame:
    """Concat brand_desc_slug columns and add a column containing brands sligified without spaces
    Args:
        datasets (pl.DataFrame): list of polars dataframes containing a brand column

    Returns:
        pl.DataFrame: polars dataframe with two columns
    """
    return (
        concat_brands_slug(datasets)
        .with_columns(
            pl.col("brand_desc_slug_reduced_lem")
            .apply(lambda x: slugify(x, separator=""))
            .alias("brand_desc_without_space")
        )
        .select("brand_desc_slug", pl.col("brand_desc_without_space").str.strip())
    )


def pair_brands_with_same_products(datasets: list) -> pl.DataFrame:
    """Pairs two brands related two the same product.

    Args:
        datasets (list): list of dataframe containg products dans brands

    Returns:
        pl.DataFrame: a polars dataframe reporting pairs of brands
    """
    # All possible pairs in List using combinations()
    pairs = list(combinations(datasets, 2))

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
