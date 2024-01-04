import polars as pl
import numpy as np
import sys
import yaml
import argparse
from loguru import logger
import pickle
from fuzzy_matching import (
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
    # Load the configuration file:
    logger.info("Loading YAML configuration file")
    with open("config.yml", "r") as file:
        config = yaml.safe_load(file)

    indicators_var, label_var, target_var = (
        config["indicators_var"],
        config["label_var"],
        config["target_var"],
    )

    input_prediction_completed = pl.read_csv(
        f"{DATA_PROCESSED_PATH}input_prediction_completed.csv", separator=";"
    )

    logger.info("Training of XGBoost Classifier")
    labeled_pairs = pl.read_csv(
        f"{DATA_PROCESSED_PATH}training_dataset.csv", separator=";"
    )

    xgb_model, test_predictions = launch_training(
        input_prediction_completed,
        labeled_pairs,
        indicators_var,
        label_var,
        target_var,
    )

    # test_predictions.write_csv(
    #     f"{DATA_PROCESSED_PATH}xgb_model_predictions_test.csv", separator=";"
    # )

    # # Save model
    # pickle.dump(xgb_model, open(f"{MODELS_PATH}{MODEL_NAME}.pickle", "wb"))

    ## Validation
    val_dataset = input_prediction_completed.join(
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
