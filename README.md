# fuzzy_matching

## Presentation
Databases often have multiple entries that relate to the same entity. In our case a same brand might appear several times with slightly different spellings. This project aims to group similar brand names together, and pick one single as the identifier for each group. 

## How to run on Linux ?

```
python -m venv.venv
source/bin/activate
pip install -r requirements.txt
python main.py --datasets <retailer_1 retailer_2 retailer_3> [--training | --no-training]
```
