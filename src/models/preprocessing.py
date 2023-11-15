import polars as pl
from sklearn.model_selection import train_test_split

from src.similarity_features import get_token_set_ratio


def create_input_for_prediction(sdf):
    df_input = (
        pl.concat(sdf, how="align")
        .filter(pl.col("left_side") != pl.col("right_side"))
        .fill_null(0)
    )
    return get_token_set_ratio(df_input)


def label_dataset(dataset, labeled_pairs):
    return dataset.join(labeled_pairs, on=["left_side", "right_side"], how="inner")


def get_train_test(
    dataset_labeled,
    indicators_var,
    label_var=["left_side", "right_side"],
    target_var="target",
):
    label = dataset_labeled.select(label_var)
    X = dataset_labeled.select(indicators_var)
    Y = dataset_labeled.select(target_var)

    # Splitting the data into train and test
    # we are stratifying on the label to ensure we receive a similar proportion of matched items in each set
    label_train, label_test, X_train, X_test, Y_train, Y_test = train_test_split(
        label, X, Y, test_size=0.3, random_state=42, stratify=Y
    )
    return label_train, label_test, X_train, X_test, Y_train, Y_test
