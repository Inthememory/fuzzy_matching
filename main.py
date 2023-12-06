import polars as pl
import sys
import yaml
import argparse
from loguru import logger
import pickle

from src import (
    gold,
    similarity_classification,
    similarity_classification_words,
    similarity_semantic,
    similarity_syntax_ngram,
    similarity_syntax_words,
    create_input_for_prediction,
    launch_training,
    export_prediction,
    evaluate_model,
    pair_brands_with_same_products,
    group_similar_strings,
    add_master_brand,
    french_preprocess_sentence,
    concat_brands_slug,
    get_brand_without_space,
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

    # Create datasets:
    datasetsLogger = logger.bind(datasets=[dataset for dataset in args.datasets])
    datasetsLogger.info("Create datasets")
    datasets = [gold(DATA_RAW_PATH, dataset, config) for dataset in args.datasets]

    # Build similarity features
    logger.info("Build features")

    logger.info("similarity_classification")
    df_similarity_classification = similarity_classification(
        datasets, config["classification_levels"]
    )
    df_similarity_classification.write_csv(
        f"{DATA_PROCESSED_PATH}similarity_classification.csv", separator=";"
    )

    logger.info("similarity_classification_words")
    df_similarity_classification_words = similarity_classification_words(
        datasets, config["classification_most_relevant_level"]
    )
    df_similarity_classification_words.write_csv(
        f"{DATA_PROCESSED_PATH}similarity_classification_words.csv", separator=";"
    )

    logger.info("similarity_semantic")
    df_similarity_semantic = similarity_semantic(datasets)
    df_similarity_semantic.write_csv(
        f"{DATA_PROCESSED_PATH}similarity_semantic.csv", separator=";"
    )

    logger.info("similarity_syntax_ngram")
    df_similarity_syntax_ngram = similarity_syntax_ngram(datasets)
    df_similarity_syntax_ngram.write_csv(
        f"{DATA_PROCESSED_PATH}similarity_syntax_ngram.csv", separator=";"
    )

    logger.info("similarity_syntax_words")
    df_similarity_syntax_words = similarity_syntax_words(datasets)
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

    indicators_var = [
        "similarity_syntax_ngram",
        "similarity_syntax_words",
        "similarity_semantic",
        "similarity_classification",
        "similarity_classification_words",
        "token_set_ratio",
    ]
    label_var = ["left_side", "right_side"]
    target_var = "target"

    # Train the model
    if args.training:
        logger.info("Training of XGBoost Classifier")
        labeled_pairs = pl.read_csv(
            f"{DATA_PROCESSED_PATH}training_dataset.csv", separator=";"
        )

        xgb_model = launch_training(
            df_for_prediction,
            labeled_pairs,
            indicators_var,
            label_var,
            target_var,
        )

        # Save model
        pickle.dump(xgb_model, open(f"{MODELS_PATH}{MODEL_NAME}.pickle", "wb"))

    ## Validation
    val_dataset = df_for_prediction.join(
        pl.read_csv(f"{DATA_PROCESSED_PATH}validation_dataset.csv", separator=";"),
        on=["left_side", "right_side"],
        how="inner",
    )
    predictions_val = export_prediction(
        [
            val_dataset.select(label_var),
            val_dataset.select(indicators_var),
            val_dataset.select(target_var),
        ],
        xgb_model.predict(val_dataset.select(indicators_var)),
        xgb_model.predict_proba(val_dataset.select(indicators_var)),
        "val",
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

    predictions = export_prediction(
        [
            df_for_prediction.select(label_var),
            df_for_prediction.select(indicators_var),
        ],
        xgb_model.predict(df_for_prediction.select(indicators_var)),
        xgb_model.predict_proba(df_for_prediction.select(indicators_var)),
        "",
    )
    print(predictions.filter(pl.col("prediction") == 1).shape[0] / predictions.shape[0])

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
                # pair_brands_with_same_products(datasets).select(
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
