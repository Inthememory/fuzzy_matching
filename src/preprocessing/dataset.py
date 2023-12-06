import polars as pl

class Dataset:

    def __init__(self, retailer, path) -> None:
        self.retailer = retailer
        self.path = path

    def scan_dataframe(self):
        self.dataframe = pl.scan_parquet(f"{self.path}{self.retailer}.parquet")
        return self.dataframe

    def expand_crumb(self, size):
        