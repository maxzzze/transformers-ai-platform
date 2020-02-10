import pandas as pd
import uuid
import logging
import glob
import argparse
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

def dataframe_to_bert(df, text_col, y_col):
    return pd.DataFrame(
        {
            "id": range(len(df)),
            "label": df[y_col],
            "alpha": ["a"] * df.shape[0],
            "text": df[text_col]
            .astype(str)
            .replace(r"\n", " ", regex=True)
            .replace(r"\t", " ", regex=True)
            .replace(r"\r", " ", regex=True),
        }
    )


def format_for_bert(split, files, text_col, y_col):
    """
    Formats a list of input CSVs for BERT classification

    returns (train_df, test_df)
    """
    if split > 1:
        split = int(split)
    df = pd.DataFrame()
    for f in files:
        df = df.append(pd.read_csv(f, index_col=0))
    
    X_train, X_test, y_train, y_test = train_test_split(
        df[text_col],
        df[y_col],
        test_size=split,
        random_state=42,
        stratify=df[y_col]
    )
    train = (
        pd.DataFrame(X_train)
        .reset_index()
        .drop("index", axis=1)
    )
    train[y_col] = y_train.reset_index()[y_col]
    test = (
        pd.DataFrame(X_test)
        .reset_index()
        .drop("index", axis=1)
    )
    test[y_col] = y_test.reset_index()[y_col]
    train_df_bert = dataframe_to_bert(
        train, text_col, y_col
    )
    dev_df_bert = dataframe_to_bert(test, text_col, y_col)
    return train_df_bert, dev_df_bert


def write_tsv(pdf, path):
    """
    Write a pandas dataframe to a .tsv file
    Expects a full path to a tsv file and a dataframe
    with the columns output of dataframe_to_bert 
    """
    pdf[["id", "label", "alpha", "text"]].to_csv(
        path, sep="\t", index=False, header=False
    )

def format_for_bert_main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        default=None,
        type=str,
        required=True,
        help="Where all the input csvs will be read from and all tsvs will be written",
    )
    parser.add_argument(
        "--text_col",
        default=None,
        type=str,
        required=True,
        help="Column in the CSVs that the text is located",
    )
    parser.add_argument(
        "--y_col",
        default=None,
        type=str,
        required=True,
        help="Column in the CSVs that the prediction label is located",
    )
    parser.add_argument(
        "--split",
        type=float,
        default=50.0,
        help="Test/train split amount",
    )
    args = parser.parse_args()
    print(args)
    logger.info(
        "Doing a {} split on files located in {}" \
        "With the {} text col and {} y col".format(
            args.split,
            args.data_dir,
            args.text_col,
            args.y_col
        )
    )
    train, dev = format_for_bert(
        files=[g for g in glob.glob(args.data_dir + "*")],
        text_col=args.text_col,
        y_col=args.y_col,
        split=args.split
    )
    logger.info("Finished splitting and converting")
    logger.info(
        "Writing training to {}".format(
            args.data_dir + "train.tsv"
        )
    )
    write_tsv(pdf=train, path=args.data_dir + "train.tsv")
    logger.info(
        "Writing testing to {}".format(
            args.data_dir + "dev.tsv"
        )
    )
    write_tsv(pdf=dev, path=args.data_dir + "dev.tsv")
    logger.info("FINISHED!")


if __name__ == "__main__":
    pass
