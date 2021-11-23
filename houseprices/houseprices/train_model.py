#%%
# %load_ext autoreload
# %autoreload 2
# Import packages
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from copy import deepcopy
import plotly.express as px
import copy
import json
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor, IsolationForest

from raymon.profiling.extractors.structured import KMeansOutlierScorer
from raymon.profiling.extractors import SequenceSimpleExtractor
from raymon.profiling import (
    ModelProfile,
    InputComponent,
    OutputComponent,
    ActualComponent,
    EvalComponent,
    MeanScore,
)
from raymon.profiling.extractors.structured import (
    generate_components,
    ElementExtractor,
    IsolationForestOutlierScorer,
)
from raymon.profiling.extractors.structured.scoring import (
    AbsoluteRegressionError,
    SquaredRegressionError,
)


from houseprices.io import load_data

ROOT = Path("..")

# %%
def get_feature_selectors(X_train):
    no_id = list(X_train.drop("Id", axis="columns").columns)
    # Select categorical features
    cat_columns = list(X_train[no_id].select_dtypes(include=["object"]))
    # Select numeric features
    num_columns = list(X_train[no_id].select_dtypes(exclude=["object"]))
    feature_selector = num_columns + cat_columns
    return feature_selector, cat_columns, num_columns


def build_prep(num_columns, cat_columns):
    # For categorical features: impute missing values as 'UNK', onehot encode
    cat_pipe = make_pipeline(
        SimpleImputer(strategy="constant", fill_value="UNK", verbose=1),
        OneHotEncoder(sparse=False, handle_unknown="ignore"),  # May lead to silent failure!
    )
    # For numerical features: impute missing values as -1
    num_pipe = make_pipeline(
        SimpleImputer(strategy="constant", fill_value=-1, verbose=1),
    )

    coltf = ColumnTransformer(transformers=[("num", num_pipe, num_columns), ("cat", cat_pipe, cat_columns)])
    return coltf


def prep_df(X_train, cat_columns, num_columns, feature_selector):
    coltf = build_prep(num_columns=num_columns, cat_columns=cat_columns)
    # train preprocessor
    Xtf_train = coltf.fit_transform(X_train[feature_selector])
    cat_columns_ohe = list(coltf.transformers_[1][1]["onehotencoder"].get_feature_names(cat_columns))
    # Save the order and name of features
    feature_selector_ohe = num_columns + cat_columns_ohe
    # Transform X_train and make sure features are in right order
    Xtf_train = pd.DataFrame(Xtf_train, columns=feature_selector_ohe)
    return Xtf_train, coltf, feature_selector_ohe


def train(X_train, y_train):
    # Setup preprocessing
    feature_selector, cat_columns, num_columns = get_feature_selectors(X_train)
    Xtf_train, coltf, feature_selector_ohe = prep_df(
        X_train=X_train,
        cat_columns=cat_columns,
        num_columns=num_columns,
        feature_selector=feature_selector,
    )

    # Train
    rf = RandomForestRegressor(n_estimators=25)
    rf.fit(Xtf_train, y_train)
    return rf, coltf, feature_selector_ohe, feature_selector


def get_importances(rf, feature_selector_ohe):
    return (
        pd.DataFrame(rf.feature_importances_, index=feature_selector_ohe, columns=["importance"])
        .rename_axis("feature")
        .sort_values("importance", ascending=False)
        .reset_index(drop=False)
    )


def plot_importances(feat_imp):
    fig = px.bar(feat_imp, x="feature", y="importance")
    fig.layout.width = 800
    fig.layout.height = 400
    fig.show()


def corrupt_df(X):
    to_drop = ["OverallQual", "GrLivArea"]
    X_corrupt = X.copy()
    X_corrupt[to_drop] = 0
    return X_corrupt


def do_pred(X, feature_selector, feature_selector_ohe):
    Xtf = pd.DataFrame(coltf.transform(X[feature_selector]), columns=feature_selector_ohe)

    y_pred = rf.predict(Xtf[feature_selector_ohe])
    return y_pred


#%%
if __name__ == "__main__":

    # Load data
    X_train, y_train = load_data(ROOT / "data/subset-cheap-train.csv")
    X_test, y_test = load_data(ROOT / "data/subset-cheap-test.csv")

    X_exp_test, y_exp_test = load_data(ROOT / "data/subset-exp-test.csv")

    rf, coltf, feature_selector_ohe, feature_selector = train(X_train=X_train, y_train=y_train)
    # %%

    with open(ROOT / "models/HousePrices-RF-v3.0.0.pkl", "wb") as f:
        pickle.dump(rf, f)
    with open(ROOT / "models/HousePrices-RF-v3.0.0-coltf.pkl", "wb") as f:
        pickle.dump(coltf, f)
    with open(ROOT / "models/HousePrices-RF-v3.0.0-ohe-sel.pkl", "wb") as f:
        pickle.dump(feature_selector_ohe, f)
    with open(ROOT / "models/HousePrices-RF-v3.0.0-sel.pkl", "wb") as f:
        pickle.dump(feature_selector, f)

    # %%
    Xtf_test = pd.DataFrame(coltf.transform(X_test[feature_selector]), columns=feature_selector_ohe)

    Xtf_exp_test = pd.DataFrame(coltf.transform(X_exp_test[feature_selector]), columns=feature_selector_ohe)

    y_pred = rf.predict(Xtf_test[feature_selector_ohe])
    y_exp_pred = rf.predict(Xtf_exp_test[feature_selector_ohe])

    mae = mean_absolute_error(y_test, y_pred)
    print(f"MAE on test set: {mae}")

    #%%
    # Construct profiles's
    X = X_test[feature_selector]

    components = generate_components(X.dtypes, complass=InputComponent) + [
        InputComponent(
            name="outlier_score",
            extractor=SequenceSimpleExtractor(prep=coltf, extractor=KMeansOutlierScorer()),
        ),
        OutputComponent(name="prediction", extractor=ElementExtractor(element=0)),
        ActualComponent(name="actual", extractor=ElementExtractor(element=0)),
        EvalComponent(name="abs_error", extractor=AbsoluteRegressionError()),
    ]
    scores = [
        MeanScore(
            name="MAE",
            inputs=["abs_error"],
            preference="low",
        ),
        MeanScore(
            name="mean_outlier_score",
            inputs=["outlier_score"],
            preference="low",
        ),
    ]

    profile = ModelProfile(
        name="HousePricesCheap",
        version="3.0.0",
        components=components,
        scores=scores,
    )
    profile.build(input=X, output=y_pred[:, None], actual=y_test[:, None])
    profile.save(ROOT / "models")
    profile.view(mode="external")
    profile_exp = copy.deepcopy(profile)
    profile_exp.name = "HousePricesExpensive"
    profile_exp.build(input=X_exp_test, output=y_exp_pred[:, None], actual=y_exp_test[:, None])
    profile_exp.save(ROOT / "models")

    # %%
    # Check the feature importances
    feat_imp = get_importances(rf=rf, feature_selector_ohe=feature_selector_ohe)
    plot_importances(feat_imp)
    feat_imp.loc[0:2, "feature"].to_list()

    ##
    jcr_contrast = profile.contrast(profile_exp)
    with open(ROOT / f"models/{profile.group_idfr}vs{profile_exp.group_idfr}.json", "w") as f:
        json.dump(jcr_contrast, f, indent=4)

    profile.view_contrast(profile_exp, mode="external")

    # %%
    # Corrupt some data
    # X_corrupt = corrupt_df(X=X_test.copy())
    # y_corr_pred = do_pred(
    #     X=X_corrupt,
    #     feature_selector=feature_selector,
    #     feature_selector_ohe=feature_selector_ohe,
    # )
    # mae = mean_absolute_error(y_test, y_corr_pred)
    # print(f"MAE on test set without top2 features: {mae}")

    # features = generate_components(X_corrupt.dtypes)
    # profile_corrupt = ModelProfile(input_components=features)
    # profile_corrupt.build(input=X_corrupt)
    # profile_corrupt.save(ROOT / "models/schema-houseprices-v3.0.0-corrupt.json")
    # # %%
    # profile.view_contrast(profile_corrupt, mode="external")

    # #%%
    # # Check the effect of more expensive houses
    # y_exp_pred = do_pred(
    #     X=X_exp_test,
    #     feature_selector=feature_selector,
    #     feature_selector_ohe=feature_selector_ohe,
    # )
    # mae = mean_absolute_error(y_exp_test, y_exp_pred)
    # print(f"MAE on expensive test set: {mae}")
    # # %%
    # # Make a schema of normal output
    # schema_preds = ModelProfile(
    #     name="model_output",
    #     version="3.0.0",
    #     input_components=[
    #         FloatComponent(
    #             name="predicted_price", extractor=ElementExtractor(element=0)
    #         )
    #     ],
    # )
    # schema_preds.build(y_pred[None, :])
    # schema_preds.save(ROOT / "models/schema-houseprices-v3.0.0-output.json")
    # # %%
    # onepred = do_pred(
    #     X=X.iloc[0:1, :],  # .to_frame().T,
    #     feature_selector=feature_selector,
    #     feature_selector_ohe=feature_selector_ohe,
    # )
    # schema_preds.check(onepred)
    # # %%
    # # make a schema for abnormal output
    # schema_preds_exp = deepcopy(schema_preds)
    # schema_preds_exp.build(y_exp_pred[None, :])

    # schema_preds.contrast(schema_preds_exp)

    # # %%
    # for tag, feature in profile.components.items():
    #     print(f"{tag} {type(feature)} is built? {feature.is_built()}")
    # # %%
    # profile.is_built()
    # # %%

#%%
