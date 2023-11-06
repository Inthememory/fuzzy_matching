from src.data import gold, load_pairs_labeled

from src.features import (
    classification,
    classification_words,
    sentence_transformer,
    syntax_ngram,
    syntax_words,
)
from src.models import (
    create_input_for_prediction,
    label_dataset,
    launch_training,
    get_prediction,
)
