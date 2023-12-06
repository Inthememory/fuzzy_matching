from src.preprocessing import (
    gold,
    french_preprocess_sentence,
    pair_brands_with_same_products,
    concat_brands_slug,
    get_brand_without_space,
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
    export_prediction,
    evaluate_model,
)
from src.group import group_similar_strings, add_master_brand
