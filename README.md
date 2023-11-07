# fuzzy_matching

## Presentation
Databases often have multiple entries that relate to the same entity. In our case a same brand might appear several times with slightly different spellings. This project aims to group similar brand names together, and pick one single as the identifier for each group. 

## How to run it ?

```bash
python main.py --datasets <dataset_name_1> <dataset_name_2> <dataset_name_3> --training
or 
python main.py --datasets <dataset_name_1> <dataset_name_2> <dataset_name_3> --no-training
```
