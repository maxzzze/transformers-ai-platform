# Bert Utils

Utility files for training custom BERT models. 

## Installation
```
pip install -e .
```

## Data Formatters

Just call `format-csvs-bert`

```
(venv) ➜  bert-utils git:(master) ✗ format-csvs-bert --help
usage: format-csvs-bert [-h] --data_dir DATA_DIR --text_col TEXT_COL --y_col
                        Y_COL [--split SPLIT]

optional arguments:
  -h, --help           show this help message and exit
  --data_dir DATA_DIR  Where all the input csvs will be read from and all tsvs
                       will be written
  --text_col TEXT_COL  Column in the CSVs that the text is located
  --y_col Y_COL        Column in the CSVs that the prediction label is located
  --split SPLIT        Test/train split amount
```