import pandas as pd
import polars as pl
import numpy as np
from typing import Union


def group_similar_strings(
    df: pl.DataFrame, min_similarity: float = 0.7
) -> pl.DataFrame:
    """_summary_

    Args:
        df (pl.DataFrame): _description_
        min_similarity (float, optional): _description_. Defaults to 0.7.

    Returns:
        pl.DataFrame: _description_
    """

    def create_new_group_id(groups: dict) -> int:
        """_summary_

        Args:
            groups (dict): _description_

        Returns:
            int: _description_
        """
        if len(groups) > 0:
            return max(list(groups.values())) + 1
        else:
            return 0

    def find_group(groups: dict, left_side: str, right_side: str) -> Union[int, None]:
        """_summary_

        Args:
            groups (dict): _description_
            left_side (str): _description_
            right_side (str): _description_

        Returns:
            Union[int, None]: _description_
        """
        # If either the row or the col string have already been given a group, return that group id.
        # Otherwise return a new group id
        if left_side in groups:
            return groups[left_side]
        elif right_side in groups:
            return groups[right_side]
        else:
            return create_new_group_id(groups)

    def add_pair_to_dict(groups: dict, left_side: str, right_side: str):
        """Assign an group id to both strings (left_side, right_side)

        Args:
            groups (dict): _description_
            left_side (str): _description_
            right_side (str): _description_
        """
        # first, see if one has already been added
        group_id = find_group(groups, left_side, right_side)
        # Once we know the group id, set it as the value for both strings in the dictionnary
        groups[left_side] = group_id
        groups[right_side] = group_id

    def add_orphan_to_dict(groups: dict, string: str):
        """Assign an group id to string

        Args:
            groups (dict): _description_
            string (str): _description_
        """
        # first, see if string has already been added to groups
        if string in groups:
            group_id = groups[string]
        # else add to a new group
        else:
            group_id = create_new_group_id(groups)
        # Once we know the group id, set it as the value in the dictionnary
        groups[string] = group_id

    # Instantiate empty dict
    groups = {}

    # for each left_side and right_side in dataframe
    # if they're similar add them to the same group
    for left_side, right_side, similarity in zip(
        df.get_column("left_side").to_list(),
        df.get_column("right_side").to_list(),
        df.get_column("similarity").to_list(),
    ):
        if similarity >= min_similarity:
            add_pair_to_dict(groups, left_side, right_side)
        else:
            add_orphan_to_dict(groups, left_side)
            add_orphan_to_dict(groups, right_side)

    return pl.DataFrame(
        list(zip(groups.keys(), groups.values())), schema=["name", "group_name"]
    )


def get_nb_products_by_brand(datasets: list) -> pl.DataFrame:
    """Create a dataframe containing the number of products by brand sulgified.

    Args:
        datasets (list): list of datasets containing products, brands and classifications

    Returns:
        pl.Dataframe: dataframe listing brand slugified and products
    """
    return (
        pl.concat(
            [
                dataset.select(
                    pl.col("product_id"),
                    pl.col("brand_desc_slug"),
                    pl.lit(f"{i}").alias("retailer_id"),
                )
                for i, dataset in enumerate(datasets)
            ],
            how="vertical",
        )
        .unique()
        .groupby("brand_desc_slug")
        .agg(
            [
                pl.count("product_id").alias("count"),
                pl.min("retailer_id").alias("retailer_id"),
            ]
        )
    )


def add_master_brand(datasets: list, res_group: pl.DataFrame) -> pl.DataFrame:
    """Add master brand_desc_slug for each group of brands

    Args:
        datasets (list): list of datasets containing products, brands
        res_group (pl.DataFrame): dataframe containing the group_name for each brand

    Returns:
        pl.DataFrame: dataframe containing the group_name and the master brand for each brand
    """
    master_brand_nb_products = (
        res_group.select("group_name", pl.col("name").alias("brand_desc_slug"))
        .join(
            get_nb_products_by_brand(datasets).select("brand_desc_slug", "count"),
            "brand_desc_slug",
            "left",
        )
        .sort(["count"], descending=[True])
        .unique(subset=["group_name"], keep="first", maintain_order=True)
        .select(
            "group_name", pl.col("brand_desc_slug").alias("master_brand_nb_products")
        )
    )

    master_brand_origin = (
        res_group.select("group_name", pl.col("name").alias("brand_desc_slug"))
        .join(get_nb_products_by_brand(datasets), "brand_desc_slug", "left")
        .sort(["retailer_id", "count"], descending=[False, True])
        .unique(subset=["group_name"], keep="first", maintain_order=True)
        .select("group_name", pl.col("brand_desc_slug").alias("master_brand_origin"))
    )

    return res_group.join(master_brand_nb_products, ["group_name"], "left").join(
        master_brand_origin, ["group_name"], "left"
    )
