import polars as pl
from slugify import slugify


def slug(x):
    if x:
        return slugify(x, replacements=[["&", "et"]], separator=" ").upper().strip()
    else:
        return None


def bronze(path: str, retailer: str, config: dict) -> pl.DataFrame:
    """_summary_

    Args:
        path (str): _description_
        retailer (str): _description_
        config (dict): _description_

    Returns:
        pl.DataFrame: _description_
    """
    return (
        pl.scan_parquet(f"{path}{retailer}.parquet")
        .select(
            pl.col("ean").alias("product_id"),
            pl.col("brand_name").alias("brand_desc"),
            "crumb",
        )
        .with_columns(
            [
                pl.col("crumb").list.get(i).alias(f"level{i}")
                for i in range(config["retailer"][retailer]["nb_levels"])
            ]
        )
        .drop("crumb")
    )


def silver(path: str, retailer: str, config: dict) -> pl.DataFrame:
    """_summary_

    Args:
        path (str): _description_
        retailer (str): _description_
        config (dict): _description_

    Returns:
        pl.DataFrame: _description_
    """
    return (
        bronze(path, retailer, config)
        .filter(pl.col("product_id").str.contains("^\d*$"))
        .filter(~pl.col("product_id").str.contains("\+"))
        .filter(~pl.col("product_id").str.contains("-"))
        .filter(pl.col("product_id").is_not_null())
        .filter(pl.col("brand_desc").is_not_null())
        .with_columns(pl.col("brand_desc").str.to_uppercase().str.strip())
        .with_columns(
            pl.col("brand_desc")
            .apply(
                lambda x: slugify(
                    x, replacements=config["slug_replacements"], separator=" "
                )
                .upper()
                .strip()
            )
            .alias("brand_desc_slug")
        )
        .select(
            [
                pl.lit(retailer).alias("retailer"),
                pl.col("product_id").str.zfill(13).alias("product_id"),
                pl.col("brand_desc"),
                pl.col("brand_desc_slug"),
            ]
            + [
                pl.col(f"level{i}").apply(
                    lambda x: slugify(x, separator=" ").upper().strip()
                )
                for i in range(config["retailer"][retailer]["nb_levels"])
            ]
        )
        .filter(pl.col("product_id") != "0000000000000")
    )


def gold(path: str, retailer: str, config: dict) -> pl.DataFrame:
    """_summary_

    Args:
        path (str): _description_
        retailer (str): _description_
        config (dict): _description_

    Returns:
        pl.DataFrame: _description_
    """
    return (
        silver(path, retailer, config)
        .filter(pl.col("level0").is_in(config["retailer"][retailer]["level0"]))
        .filter(
            ~pl.col("level1").is_in(config["retailer"][retailer]["level1_to_delete"])
        )
        .filter(~pl.col("brand_desc_slug").is_in(config["unknown_brands"]))
    )
