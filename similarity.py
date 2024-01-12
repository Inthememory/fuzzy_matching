import polars as pl
import numpy as np
import sys
import yaml
import argparse
from loguru import logger
import pickle
import os

from french_lefff_lemmatizer.french_lefff_lemmatizer import FrenchLefffLemmatizer
from nltk.corpus import stopwords

from fuzzy_matching import Dataset, DatasetsMerged, Similarity
from fuzzy_matching import (
    create_input_for_prediction,
    launch_training,
    get_predictions,
    evaluate_model,
)


# Initialise paths
DATA_RAW_PATH = "data/raw/"
DATA_PROCESSED_PATH = "data/processed/"
MODELS_PATH = "models/"
MODEL_NAME = "xgb_model"

# Initialise logs format
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

    # Set loguru LEVEL
    logger.configure(handlers=[{"sink": sys.stderr, "level": "DEBUG"}])

    # Load the configuration file and set parameters
    logger.debug("Loading YAML configuration file")
    with open("config.yml", "r") as file:
        config = yaml.safe_load(file)

    unknown_brands = config["unknown_brands"]
    generic_words = (
        pl.read_csv(
            f"{DATA_RAW_PATH}generic_words.csv", separator=";", has_header=False
        )
        .to_series()
        .to_list()
    )
    list_stopwords = [word for word in stopwords.words("french")] + ["a", "o"]
    lemmatizer = FrenchLefffLemmatizer()
    sl_replacements = config["sl_replacements"]

    ## 1. Preprocessing: calculate input datasets
    # Create a list of clean datasets to proceed
    logger.info(f"Create datasets : {args.datasets}")
    datasets = [
        Dataset(
            df=pl.read_parquet(f"data/raw/{dataset}.parquet"),
            retailer=dataset,
            nb_levels=config["retailer"][dataset]["nb_levels"],
            levels_col="crumb",
            level0_included=config["retailer"][dataset]["level0"],
            level1_excluded=config["retailer"][dataset]["level1_to_delete"],
            replacements_brand=[["&", "et"]],
        ).filter_dataset(unknown_brands=unknown_brands)
        for dataset in args.datasets
    ]
    datasets_merged = DatasetsMerged(datasets)

    # Merge datasets and convert classification variable into dummy variables.
    brand_classification_dummy = datasets_merged.get_brand_classification_dummy(
        levels=config["classification_levels"]
    )
    logger.debug(f"brand_classification_dummy : {brand_classification_dummy.shape}")

    # Merge datasets and clean a specified level column.
    brand_classification_words = datasets_merged.get_brand_classification_words(
        levels=config["classification_most_relevant_levels"],
        lemmatizer=lemmatizer,
        list_stopwords=list_stopwords,
    )
    logger.debug(f"brand_classification_words : {brand_classification_words.shape}")
    # Write results
    brand_classification_words.write_csv(
        f"{DATA_PROCESSED_PATH}brand_classification_words.csv", separator=";"
    )

    # Concat vertically brands from each dataset
    brands_updated = datasets_merged.extract_brands(
        lemmatizer=lemmatizer,
        list_stopwords=list_stopwords,
        generic_words=generic_words,
        replacements=sl_replacements,
    )
    logger.debug(f"brands_updated : {brands_updated.shape}")
    # Write results
    brands_updated.write_csv(f"{DATA_PROCESSED_PATH}brands_updated.csv", separator=";")

    # Create all pairs of brands with a cartesian product
    brands_cross_join = DatasetsMerged.cross_join(
        brands_updated, ["brand_desc_slug", "brand_desc_slug_updated"]
    )
    logger.debug(f"brands_cross_join : {brands_cross_join.shape}")

    # Calculate the number of product by brand and the best retailer
    nb_products_by_brand = datasets_merged.get_nb_products_by_brand()
    nb_products_by_brand.write_csv(
        f"{DATA_PROCESSED_PATH}nb_products_by_brand.csv", separator=";"
    )
    logger.debug(f"nb_products_by_brand : {nb_products_by_brand.shape}")

    ## 2. Build similarity features
    logger.debug("Build features")
    # Initialise an empty list to stores similarity datasets
    similarities_features = []

    logger.debug("similarity_syntax_ngram")
    # Create Similarity object
    similarity_syntax_ngram = Similarity(
        brands_updated,
        name="syntax_ngram",
        label_col="brand_desc_slug",
        col="brand_desc_slug_updated_w_space",
        tfidf_required=True,
    )
    # Fix parameters
    similarity_syntax_ngram.analyzer = "char"
    similarity_syntax_ngram.ngram_range = (2, 3)
    # Compute cosin similarity
    similarity_syntax_ngram.cos_sim(min_similarity=0.2)
    similarities_features.append(similarity_syntax_ngram.pairwise_dataset)
    logger.debug(
        f"sparsity : {similarity_syntax_ngram.sparsity()}, \
                 shape {similarity_syntax_ngram.pairwise_dataset.shape}"
    )

    logger.debug("similarity_classification")
    # Create Similarity object
    similarity_classification = Similarity(
        brand_classification_dummy, name="classification", label_col="brand_desc_slug"
    )
    # Compute cosinm similarity
    similarity_classification.cos_sim()
    similarities_features.append(similarity_classification.pairwise_dataset)
    logger.debug(
        f"sparsity : {similarity_classification.sparsity()}, \
                 shape {similarity_classification.pairwise_dataset.shape}"
    )

    logger.debug("similarity_classification_words")
    # Create Similarity object
    similarity_classification_words = Similarity(
        brand_classification_words,
        name="classification_words",
        label_col="brand_desc_slug",
        col="level_updated",
        tfidf_required=True,
    )
    # Fix parameters
    similarity_classification_words.token_pattern = r"(?u)\b[A-Za-z]{2,}\b"
    # Compute cosin similarity
    similarity_classification_words.cos_sim()
    similarities_features.append(similarity_classification_words.pairwise_dataset)
    logger.debug(
        f"sparsity : {similarity_classification_words.sparsity()}, \
                 shape {similarity_classification_words.pairwise_dataset.shape}"
    )

    ## 3. Create input for prediction
    similarities_features.append(
        brands_cross_join.rename({"brand_desc_slug_left": "left_side"}).rename(
            {"brand_desc_slug_right": "right_side"}
        )
    )

    input_prediction_init = create_input_for_prediction(similarities_features)
    logger.debug(f"input_prediction_init shape: {input_prediction_init.shape}")

    # Add distance_metrics
    logger.debug("similarity_fuzzy")
    input_prediction_completed = Similarity.distance_metrics(
        input_prediction_init,
        col_left="brand_desc_slug_updated_left",
        col_right="brand_desc_slug_updated_right",
    )
    logger.debug(
        f"input_prediction_completed shape: {input_prediction_completed.shape}"
    )
    input_prediction_completed.write_csv(
        f"{DATA_PROCESSED_PATH}input_prediction_completed.csv", separator=";"
    )
    input_prediction_completed = input_prediction_completed.drop(
        "brand_desc_slug_updated_left", "brand_desc_slug_updated_right"
    )

    ## 4. Fit the model (XGBoost Classifier) if training = True
    # Set variables
    indicators_var, label_var, target_var = (
        config["indicators_var"],
        config["label_var"],
        config["target_var"],
    )

    if args.training:
        logger.debug("Training of XGBoost Classifier")
        labeled_pairs = pl.read_csv(
            f"{DATA_RAW_PATH}training_dataset.csv", separator=";"
        )

        # Make predictions and evaluate model
        xgb_model, test_predictions = launch_training(
            input_prediction_completed,
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

    ## 5. Evaluate the model on a validation dataset
    val_dataset = input_prediction_completed.join(
        pl.read_csv(f"{DATA_RAW_PATH}validation_dataset.csv", separator=";"),
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

    ## 6. Make predictions on the whole dataset
    logger.debug("Predict")
    xgb_model = pickle.load(open(f"{MODELS_PATH}{MODEL_NAME}.pickle", "rb"))

    predictions = get_predictions(xgb_model, input_prediction_completed, config)
    predictions.write_csv(
        f"{DATA_PROCESSED_PATH}xgb_model_predictions.csv", separator=";"
    )
