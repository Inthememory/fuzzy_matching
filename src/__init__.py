from src.preprocessing import (
    gold,
    french_preprocess_sentence,
    pair_brands_with_same_products,
)

from src.similarity_features import (
    similarity_classification,
    similarity_classification_words,
    similarity_semantic,
    similarity_syntax_ngram,
    similarity_syntax_words,
)
from src.models import (
    create_input_for_prediction,
    launch_training,
    get_prediction,
)
from src.group import group_similar_strings
