import polars as pl


if __name__ == "__main__":
    ## Create dataframe with new brand
    schema = {"ean": pl.Utf8, "brand_name": pl.Utf8, "crumb": pl.Object(pl.Utf8)}
    data = {
        "ean": ["0000030068544"],
        "brand_name": ["PRESIDEND"],
        "crumb": [["P.L.S.", "FROMAGES, CREMERIE, BEURRE, OEUF", "PATES MOLLES"]],
    }

    new_brand = pl.DataFrame(data=data)
    print(new_brand)

    new_brand.write_parquet("data/raw/new_brand.parquet")
