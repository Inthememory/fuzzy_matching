from fuzzy_matching.preprocessing.dataset import Dataset, DatasetsMerged
from fuzzy_matching.similarity_features import Similarity

from fuzzy_matching.models import (
    create_input_for_prediction,
    launch_training,
    get_predictions,
    evaluate_model,
)
from fuzzy_matching.group import group_similar_strings, add_master_brand
