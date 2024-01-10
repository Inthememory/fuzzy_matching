import polars as pl
import nltk
import itertools
from collections import defaultdict
import argparse
from loguru import logger
import sys
import os

if __name__ == "__main__":
    # Parse arg from the command line:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--word",
        type=str,
        required=False,
        help="Word for which the number of occurrences is sought.",
    )
    parser.add_argument(
        "--threshold",
        type=int,
        required=False,
        default=10,
        help="Number minimum of occurences",
    )

    args = parser.parse_args()
    FILE_PATH = "data/processed/brands_updated.csv"

    # Set loguru LEVEL
    logger.configure(handlers=[{"sink": sys.stderr, "level": "DEBUG"}])

    def count_words_occurences():
        # initializing brand desc list
        brands_df = (
            pl.read_csv(FILE_PATH, separator=";")
            .select("brand_desc_slug")
            .with_columns(
                pl.col("brand_desc_slug").apply(
                    lambda x: nltk.tokenize.word_tokenize(x)
                )
            )
        )
        brands_list = brands_df.to_series().to_list()
        brands_list_flat_list = list(itertools.chain(*brands_list))

        # initializing empty dict to store counts
        dict_count = defaultdict(int)

        # count the number of occurrences
        for sub in brands_list_flat_list:
            for word in sub.split():
                dict_count[word] += 1

        return dict_count

    if os.path.exists("data/processed/brands_updated.csv"):
        dict_count = count_words_occurences()

        if args.word is not None:
            if args.word in dict_count:
                logger.info(f"{args.word} appears {dict_count[args.word]} times.")
            else:
                raise ValueError(f"{args.word} does not exist")

        if args.threshold is not None:
            # filter words with nb of occurences above the threshold
            res = dict(
                (k, v)
                for k, v in sorted(
                    dict_count.items(), key=lambda item: item[1], reverse=True
                )
                if v > 10
            )
            logger.info(
                f"Words that appear more than {args.threshold} times : \n {res}"
            )

    else:
        raise ValueError('"data/processed/brands_updated.csv" does not exist')
