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
)


# Initialise paths
DATA_RAW_PATH = "data/raw_v2/"
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
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--training",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Execute training.",
    )
    args = parser.parse_args()

    # Set loguru LEVEL
    logger.configure(handlers=[{"sink": sys.stderr, "level": "DEBUG"}])

    # --------------------------------------------------
    # 0. Load the configuration file and set parameters
    # --------------------------------------------------
    logger.debug("Loading YAML configuration file")
    with open("config.yml", "r") as file:
        config = yaml.safe_load(file)

    input_prediction = pl.read_csv(
        f"{DATA_PROCESSED_PATH}input_prediction_completed.csv", separator=";"
    )
    input_prediction = input_prediction.drop(
        "brand_desc_slug_updated_left", "brand_desc_slug_updated_right"
    )

    # ---------------------------------------------------------
    # 4. Fit the model (XGBoost Classifier) if training = True
    # ---------------------------------------------------------
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
            input_prediction,
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

    # # ---------------------------------------------
    # # 5. Evaluate the model on a validation dataset
    # # ---------------------------------------------
    # val_dataset = input_prediction.join(
    #     pl.read_csv(f"{DATA_RAW_PATH}validation_dataset.csv", separator=";"),
    #     on=["left_side", "right_side"],
    #     how="inner",
    # )
    # val_predictions = get_predictions(xgb_model, val_dataset, config)

    # val_predictions.write_csv(
    #     f"{DATA_PROCESSED_PATH}xgb_model_predictions_val.csv", separator=";"
    # )

    # log_loss_val, roc_auc_score_val, confusion_matrix_val = evaluate_model(
    #     xgb_model,
    #     val_dataset.select(indicators_var),
    #     val_dataset.select(target_var),
    #     "val",
    # )

    # ----------------------------------------
    # 6. Make predictions on the whole dataset
    # ----------------------------------------
    logger.debug("Predict")
    xgb_model = pickle.load(open(f"{MODELS_PATH}{MODEL_NAME}.pickle", "rb"))

    predictions = get_predictions(xgb_model, input_prediction, config)
    predictions.write_csv(
        f"{DATA_PROCESSED_PATH}xgb_model_predictions.csv", separator=";"
    )
