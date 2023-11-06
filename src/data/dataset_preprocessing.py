import polars as pl
import string
from french_lefff_lemmatizer.french_lefff_lemmatizer import FrenchLefffLemmatizer
from slugify import slugify


def datasets_merged_align(datasets, levels):
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
        .collect()
    )


def datasets_merged_vertical(datasets):
    return (
        pl.concat(
            [dataset.select(pl.col("brand_desc_slug")) for dataset in datasets],
            how="vertical",
        )
        .unique()
        .collect()
    )


def get_items_list(df, col_id):
    return df.get_columns()[col_id].to_list()


def get_brand_classification(datasets, classification_levels):
    product_classification = datasets_merged_align(datasets, classification_levels)

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
    datasets, classification_most_relevant_level, list_stopwords
):
    def convert_to_list_of_words(list_of_sentences):
        l = []
        for s in list_of_sentences:
            l += [
                "".join(
                    char.lower()
                    for char in item
                    if char not in string.punctuation and len(char) > 0
                )
                for item in s.split()
            ]
        return l

    def lemmatize_words(list_of_words):
        lemmatizer = FrenchLefffLemmatizer()
        words_w_stopwords = [i for i in list_of_words if i not in list_stopwords]
        return [lemmatizer.lemmatize(w) for w in words_w_stopwords]

    def remove_duplicates(l):
        return list(set(l))

    product_classification = datasets_merged_align(
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
        .with_columns(pl.col(f"level").apply(convert_to_list_of_words))
        .with_columns(
            pl.col(f"level").apply(remove_duplicates).alias(f"level_lemmatize")
        )
        # .with_columns(pl.col(f"level").apply(lemmatize_words).alias(f"level_lemmatize"))
        .with_columns(pl.col(f"level_lemmatize").cast(pl.List(pl.Utf8)).list.join(" "))
        .with_columns(
            pl.col(f"level_lemmatize")
            .apply(lambda x: slugify(x, separator=" ").upper().strip())
            .alias(f"level_slug")
        )
    )
    return brand_classificationWords


def get_brand_without_space(datasets):
    return (
        datasets_merged_vertical(datasets)
        .with_columns(
            pl.col("brand_desc_slug")
            .apply(lambda x: slugify(x, separator=""))
            .alias("brand_desc_without_space")
        )
        .select("brand_desc_slug", "brand_desc_without_space")
    )
