import numpy as np
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
import pandas as pd

if __name__ == "__main__":
    df = pd.read_csv("../csv/train.csv")

    print(df.head(5))
    # create column for kfold with values -1 for temp
    df.loc[:, "kfold"] = -1

    df = df.sample(frac=1).reset_index(drop = True)

    X = df.image_id.values
    y = df[["grapheme_root", "vowel_diacritic", "consonant_diacritic"]].values

    msk = MultilabelStratifiedKFold(n_splits=5)

    for fold, (train_, val_) in enumerate(msk.split(X,y)):
        print("train =", train_, "val =", val_)
        df.loc[val_,"kfold"] = fold

    print(df.kfold.value_counts())
    df.to_csv("../csv/train_folds.csv")
