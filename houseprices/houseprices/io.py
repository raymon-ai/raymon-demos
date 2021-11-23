import math
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
import json


ROOT = Path("..")
to_drop = [
    "Alley",
    "PoolQC",
    "Fence",
    "MiscFeature",
    "BsmtQual",
    "BsmtCond",
    "BsmtExposure",
    "BsmtFinType1",
    "BsmtFinType2",
    "GarageType",
    "GarageFinish",
    "GarageQual",
    "FireplaceQu",
    "GarageCond",
    "Utilities",
    "RoofMatl",
    "Heating",
    "GarageYrBlt",
    "LotFrontage",
]


def load_data(fpath):
    target_col = "SalePrice"
    df = pd.read_csv(fpath, index_col=False)
    df = df.set_index("Id", drop=False)
    X = df.drop([target_col], axis="columns").drop(to_drop, axis="columns", errors="ignore")
    y = df[target_col]
    return X, y


def save_data(X, y, fpath):
    df = pd.concat([X, y.to_frame()], axis="columns")
    df.reset_index(drop=True).to_csv(fpath, index=False)
    return df


def dropid(df):
    return df.drop("Id", axis="columns")


def load_client_data(client_type="cheap", root=ROOT):
    X, y = load_data(root / f"data/subset-{client_type}-test.csv")

    return X, y


def load_pkl(fpath):
    with open(fpath, "rb") as f:
        return pickle.load(f)
