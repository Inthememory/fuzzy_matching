import pandas as pd
import polars as pl
import numpy as np
from typing import Union, Dict

def group_similar_strings( 
    df: pl.dataframe,
    groups_init: Union[pl.DataFrame, None],
    min_similarity: float = 0.7,
) -> pl.DataFrame:
    """Group stirngs whose probability of similarity is above a certain threshold

    Args:
        df (pl.DataFrame): dataset that contains pairs of strings and probability of similarity between the two
        groups (Union): previously formed groups
        min_similarity (float, optional): probability of similarity threshold. Defaults to 0.7.

    Returns:
        pl.DataFrame: a dataframe containing group_ids assigned to each string (each string has a unique group_id)
    """

    # Instantiate empty dict if groups is None
    if groups_init is None:
        groups = {}
    else:
        # Get a Python dict from columns  "name", "group_name" of groups_init dataframe (e.g. {name1: groupe_name1})
        groups = dict(groups_init.select( "name", "group_name").iter_rows())

    # Instantiate empty dict from group_id imputation (groups that will be merged)
    groupId_imputation = {}

    # sort dataset to add orphan at the end
    dataset = df.sort(["similarity"], descending=True)

    # for each left_side and right_side in dataframe if they're similar add them to the same group
    for left_side, right_side, similarity in zip(
        dataset.get_column("left_side").to_list(),
        dataset.get_column("right_side").to_list(),
        dataset.get_column("similarity").to_list(),
    ):
        if similarity >= min_similarity:
            add_pair_to_dict(groups, groupId_imputation, left_side, right_side)
        else:
            add_orphan_to_dict(groups, left_side)
            add_orphan_to_dict(groups, right_side)

    # Iterate through the dictionary items and update the values based on groupId_imputation
    for key, value in groups.items():
        if value is groupId_imputation:
            # Replace the value with the new value
            groups[key] = groupId_imputation[
                value
            ]  # Replace 'new_value' with the value to set

    return pl.DataFrame(
        list(zip(groups.keys(), groups.values())), schema=["name", "group_name"]
    )


def create_new_group_id(groups: dict) -> int:
    """Return a new group_id in an incremental way

    Args:
        groups (dict): previously formed groups

    Returns:
        int: new group_id
    """
    if len(groups) == 0:
        return 0
    else:
        return max(list(groups.values())) + 1

def find_group(groups: dict, left_side: str, right_side: str) -> Union[int, tuple]:
    """Return a group id:
        - if both strings are new create a new group
        - if one of the two strings has already received a group_id, return that group id.
        - if both strings have already received a group_id, return a tuple with tese two ids.

    Args:
        groups (dict): previously formed groups
        left_side (str): first string for comparison
        right_side (str): second string for comparison

    Returns:
        Union[int, None]: a group_id or a tuple of group_ids
    """
    # Both strings have already received a group_id, return a tuple with tese two ids
    if left_side in groups and right_side in groups:
        return (groups[left_side], groups[right_side])
    # One of the two strings has already received a group_id, return that group id.
    elif left_side in groups:
        return groups[left_side]
    elif right_side in groups:
        return groups[right_side]
    # Create a new group
    else:
        return create_new_group_id(groups)

def add_pair_to_dict(groups: dict, groupId_imputation: dict, left_side: str, right_side: str):
    """Assign a group id to two strings (left_side, right_side)

    Args:
        groups (dict): previously formed groups
        groupId_imputation (dict): groups to merge
        left_side (str): first string to which a group must be assigned
        right_side (str): second string to which a group must be assigned
    """
    # Get group_id(s) to assign
    group_id = find_group(groups, left_side, right_side)
    # if there is a unique group_id, assign it to both strings
    if isinstance(group_id, int):
        groups[left_side] = group_id
        groups[right_side] = group_id
    # if there are two distinct group_ids, add these two id to a dict as key and value. These groups will merge.
    elif group_id[0] != group_id[1]:
        groupId_imputation[group_id[0]] = group_id[1]

def add_orphan_to_dict(groups: dict, string: str):
    """Assign a group id to a string

    Args:
        groups (dict): previously formed groups
        string (str): string to which a group must be assigned
    """
    # If string hasn't already been added to groups create a new group and set it as the value in the dictionnary
    if string not in groups:
        group_id = create_new_group_id(groups)
        groups[string] = create_new_group_id(groups)
        

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
