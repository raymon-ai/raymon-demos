# %%
# Import packages
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from raymon_agent.io import load_data, save_data

pd.set_option("display.max_rows", 300)
ROOT = Path(".")

# %%
def split_midmarket(midmarket_feats, midmarket_target):
    mm_cheap_mask = np.random.random(len(midmarket_target)) > 0.5
    mm_cheap_feats, mm_cheap_target = (
        midmarket_feats[mm_cheap_mask],
        midmarket_target[mm_cheap_mask],
    )

    mm_exp_mask = ~mm_cheap_mask
    mm_exp_feats, mm_exp_target = (
        midmarket_feats[mm_exp_mask],
        midmarket_target[mm_exp_mask],
    )

    return mm_cheap_feats, mm_cheap_target, mm_exp_feats, mm_exp_target


def split_clients(df, target):
    # TODO this function is too complex, refactor
    quantiles = target.quantile([0.5, 0.75])

    poorq = quantiles.iloc[0]
    richq = quantiles.iloc[1]
    # Pure cheap houses
    cheap_mask = target <= poorq
    cheap_feats, cheap_target = df[cheap_mask.values], target[cheap_mask.values]
    # Pure expensive houses
    expensive_mask = target >= richq
    expensive_feats, expensive_target = (
        df[expensive_mask.values],
        target[expensive_mask.values],
    )
    # Split midmarket
    midmarket_mask = (poorq < target) & (target < richq)
    midmarket_feats, midmarket_target = (
        df[midmarket_mask.values],
        target[midmarket_mask.values],
    )
    mm_cheap_feats, mm_cheap_target, mm_exp_feats, mm_exp_target = split_midmarket(midmarket_feats, midmarket_target)

    # Concat cheaps
    cheap_feats_comb = pd.concat([cheap_feats, mm_cheap_feats])
    cheap_target_comb = pd.concat([cheap_target, mm_cheap_target])

    # Concat expansives
    exp_feats_comb = pd.concat([expensive_feats, mm_exp_feats])
    exp_target_comb = pd.concat([expensive_target, mm_exp_target])

    return cheap_feats_comb, cheap_target_comb, exp_feats_comb, exp_target_comb


# %%


def plot_clients(cheap_target, exp_target):
    cheap_df = cheap_target.to_frame()
    exp_df = exp_target.to_frame()

    cheap_df["client"] = "cheap"
    exp_df["client"] = "expensive"
    df = pd.concat([cheap_df, exp_df])
    fig = px.histogram(df, x="SalePrice", color="client")
    fig.show()


def save_train_test(X, y, ref):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=True, random_state=1000)
    save_data(X=X_train, y=y_train, fpath=ROOT / f"data/subset-{ref}-train.csv")
    save_data(X=X_test, y=y_test, fpath=ROOT / f"data/subset-{ref}-test.csv")


# %%
if __name__ == "__main__":
    # Load the data
    X, y = load_data(fpath=ROOT / "data/train.csv")

    # cheap_feats_comb, cheap_target_comb, exp_feats_comb, exp_target_comb = split_clients(X, y)
    X_cheap, y_cheap, X_exp, y_exp = split_clients(X, y)

    # Save train and tests sets for every client type
    save_train_test(X=X_cheap, y=y_cheap, ref="cheap")
    save_train_test(X=X_exp, y=y_exp, ref="exp")

    # Save full client datasets
    save_data(X=X_cheap, y=y_cheap, fpath=ROOT / "data/subset-cheap.csv")
    save_data(X=X_exp, y=y_exp, fpath=ROOT / "data/subset-exp.csv")

    # %%
    plot_clients(cheap_target=y_cheap, exp_target=y_exp)

    # %%
    X, y = load_data(fpath=ROOT / "data/subset-cheap.csv")
    X

# %%
