import yaml
from loguru import logger

from nltk.corpus import stopwords

from utils.load_data import *

STOPWORDS_LIST = stopwords.words("english") + stopwords.words("french")

# Loading of the configuration file:
logger.info("Loading YAML configuration file")
with open("config.yml", "r") as file:
    config = yaml.safe_load(file)

# if __name__ == '__main__':
