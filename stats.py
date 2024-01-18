import polars as pl
import os
import yaml
import argparse

# Initialise paths
DATA_RAW_PATH = "data/raw/"
DATA_PROCESSED_PATH = "data/processed/"
MODELS_PATH = "models/"
MODEL_NAME = "xgb_model"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--set", type=str, required=True)
    args = parser.parse_args()

    # Load features
    if os.path.exists(f"{DATA_PROCESSED_PATH}xgb_model_predictions_test.csv"):
        predictions = pl.read_csv(
            f"{DATA_PROCESSED_PATH}xgb_model_predictions_{args.set}.csv",
            separator=";",
        )
    else:
        raise ValueError(f"File xgb_model_predictions_{args.set}.csv does not exist")

    # Load the configuration file and set parameters
    with open("config.yml", "r") as file:
        config = yaml.safe_load(file)
        features = config["indicators_var"]

    pred_0 = predictions.filter(pl.col("target") == 0).select(
        [pl.col(col).alias(f"target_0_{col}") for col in features]
    )
    pred_1 = predictions.filter(pl.col("target") == 1).select(
        [pl.col(col).alias(f"target_1_{col}") for col in features]
    )
    pred_0_stats = pred_0.describe()
    pred_1_stats = pred_1.describe()
    stats = pred_0_stats.join(pred_1_stats, on="describe")

    print(stats)
    # Write results
    stats.write_csv(f"{DATA_PROCESSED_PATH}{args.set}_stats.csv", separator=";")
