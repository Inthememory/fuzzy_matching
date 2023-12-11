import polars as pl
import sys
import yaml
import argparse
from loguru import logger
import pickle

from french_lefff_lemmatizer.french_lefff_lemmatizer import FrenchLefffLemmatizer
from nltk.corpus import stopwords

from src import (
    Dataset,
    DatasetsMerged,
    similarity_classification,
    similarity_classification_words,
    similarity_semantic,
    similarity_syntax_ngram,
    similarity_syntax_words,
    create_input_for_prediction,
    launch_training,
    get_predictions,
    evaluate_model,
    group_similar_strings,
    add_master_brand,
)


DATA_RAW_PATH = "data/raw/"
DATA_PROCESSED_PATH = "data/processed/"
MODELS_PATH = "models/"
MODEL_NAME = "xgb_model"

logger.remove(0)
logger.add(
    sys.stderr,
    format="{time} | {level} | {message} | {extra}",
)

if __name__ == "__main__":
    # Add argparse for the command line:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets", type=str, required=True, nargs="+", help="Datasets listing."
    )
    parser.add_argument(
        "--training",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Execute training.",
    )
    args = parser.parse_args()

    # Load the configuration file:
    logger.info("Loading YAML configuration file")
    with open("config.yml", "r") as file:
        config = yaml.safe_load(file)

    unknown_brands = config["generic_words"]
    generic_words = config["generic_words"]
    list_stopwords = stopwords.words("french")
    lemmatizer = FrenchLefffLemmatizer()

    # Create datasets:
    datasetsLogger = logger.bind(datasets=[dataset for dataset in args.datasets])
    datasetsLogger.info("Create datasets")
    datasets = [
        Dataset(
            pl.read_parquet(f"data/raw/{dataset}.parquet"),
            dataset,
            nb_levels=config["retailer"][dataset]["nb_levels"],
        ).filter_dataset(unknown_brands)
        for dataset in args.datasets
    ]
    datasets_merged = DatasetsMerged(datasets)

    brand_classification_dummy = datasets_merged.get_brand_classification_dummy(
        levels=config["classification_levels"]
    )
    print(f"brand_classification_dummy : {brand_classification_dummy.shape}")

    brand_classification_words = datasets_merged.get_brand_classification_words(
        level=config["classification_most_relevant_level"],
        lemmatizer=lemmatizer,
        list_stopwords=list_stopwords,
    )
    print(f"brand_classification_words : {brand_classification_words.shape}")

    brands_updated = datasets_merged.extract_brands(
        lemmatizer=lemmatizer,
        list_stopwords=list_stopwords,
        generic_words=generic_words,
    )
    print(f"brands_updated : {brands_updated.shape}")

    # brands_with_same_products_paired = datasets_merged.pair_brands_with_same_products()
    # print(
    #     f"brands_with_same_products_paired : {brands_with_same_products_paired.shape}"
    # )
    # brands_with_same_products_paired.write_csv(
    #     f"{DATA_PROCESSED_PATH}brands_with_same_products_paired.csv", separator=";"
    # )

    # Build similarity features
    logger.info("Build features")

    logger.info("similarity_classification")
    df_similarity_classification = similarity_classification(
        brand_classification_dummy, col_label="brand_desc_slug"
    )
    df_similarity_classification.write_csv(
        f"{DATA_PROCESSED_PATH}similarity_classification.csv", separator=";"
    )

    logger.info("similarity_classification_words")
    df_similarity_classification_words = similarity_classification_words(
        brand_classification_words, col="level_updated", col_label="brand_desc_slug"
    )
    df_similarity_classification_words.write_csv(
        f"{DATA_PROCESSED_PATH}similarity_classification_words.csv", separator=";"
    )

    logger.info("similarity_semantic")
    df_similarity_semantic = similarity_semantic(
        brands_updated, col="brand_desc_slug_updated", col_label="brand_desc_slug"
    )
    df_similarity_semantic.write_csv(
        f"{DATA_PROCESSED_PATH}similarity_semantic.csv", separator=";"
    )

    logger.info("similarity_syntax_ngram")
    df_similarity_syntax_ngram = similarity_syntax_ngram(
        brands_updated,
        col="brand_desc_slug_updated_w_space",
        col_label="brand_desc_slug",
    )
    df_similarity_syntax_ngram.write_csv(
        f"{DATA_PROCESSED_PATH}similarity_syntax_ngram.csv", separator=";"
    )

    logger.info("similarity_syntax_words")
    df_similarity_syntax_words = similarity_syntax_words(
        brands_updated, col="brand_desc_slug_updated", col_label="brand_desc_slug"
    )
    df_similarity_syntax_words.write_csv(
        f"{DATA_PROCESSED_PATH}similarity_syntax_words.csv", separator=";"
    )

    df_for_prediction = create_input_for_prediction(
        [
            df_similarity_classification,
            df_similarity_classification_words,
            df_similarity_semantic,
            df_similarity_syntax_ngram,
            df_similarity_syntax_words,
        ]
    )
    df_for_prediction.write_csv(
        f"{DATA_PROCESSED_PATH}df_for_prediction.csv", separator=";"
    )

    indicators_var = config["indicators_var"]
    label_var = config["label_var"]
    target_var = config["target_var"]

    # Train the model
    if args.training:
        logger.info("Training of XGBoost Classifier")
        labeled_pairs = pl.read_csv(
            f"{DATA_PROCESSED_PATH}training_dataset.csv", separator=";"
        )

        xgb_model, test_predictions = launch_training(
            df_for_prediction,
            labeled_pairs,
            indicators_var,
            label_var,
            target_var,
        )

        test_predictions.write_csv(
            f"{DATA_PROCESSED_PATH}xgb_model_predictions_test.csv", separator=";"
        )

        # Save model
        pickle.dump(xgb_model, open(f"{MODELS_PATH}{MODEL_NAME}.pickle", "wb"))

    ## Validation
    val_dataset = df_for_prediction.join(
        pl.read_csv(f"{DATA_PROCESSED_PATH}validation_dataset.csv", separator=";"),
        on=["left_side", "right_side"],
        how="inner",
    )
    val_predictions = get_predictions(xgb_model, val_dataset, config)

    val_predictions.write_csv(
        f"{DATA_PROCESSED_PATH}xgb_model_predictions_val.csv", separator=";"
    )

    log_loss_val, roc_auc_score_val, confusion_matrix_val = evaluate_model(
        xgb_model,
        val_dataset.select(indicators_var),
        val_dataset.select(target_var),
        "val",
    )

    # Get predictions
    logger.info("Predict")
    xgb_model = pickle.load(open(f"{MODELS_PATH}{MODEL_NAME}.pickle", "rb"))

    predictions = get_predictions(xgb_model, df_for_prediction, config)
    print(predictions.filter(pl.col("prediction") == 1).shape[0] / predictions.shape[0])
    predictions.write_csv(
        f"{DATA_PROCESSED_PATH}xgb_model_predictions.csv", separator=";"
    )

    # Create groups
    logger.info("Create groups")
    df_input_groups = (
        pl.concat(
            [
                predictions.select(
                    pl.col("left_side"),
                    pl.col("right_side"),
                    pl.col("proba_1").cast(float).alias("similarity"),
                ),
                # brands_with_same_products_paired.select(
                #     pl.col("left_side"),
                #     pl.col("right_side"),
                #     pl.lit(1.0).alias("similarity"),
                # ),
            ]
        )
        .groupby(pl.col("left_side"), pl.col("right_side"))
        .agg(pl.col("similarity").max())
    )
    res_group = group_similar_strings(df_input_groups, min_similarity=0.8).sort(
        by="group_name"
    )
    print(
        f"Nb brands : {res_group.shape[0]}, nb groups : {res_group.select('group_name').unique().shape[0]}"
    )

    res_group_with_master = add_master_brand(datasets, res_group)

    res_group_with_master.write_csv(
        f"{DATA_PROCESSED_PATH}res_group_full.csv", separator=";"
    )
