#%%
# %load_ext autoreload
# %autoreload 2
import pandas as pd
import numpy as np
import random
from PIL import Image
from PIL import ImageFile
import torchvision
from torchvision import transforms

from pathlib import Path
from pydoc import locate
from raymon.profiling import (
    ModelProfile,
    InputComponent,
    OutputComponent,
    ActualComponent,
    EvalComponent,
    DataType,
    extractors,
)
from raymon.profiling.extractors.vision import DN2AnomalyScorer, AvgIntensity, Sharpness
from raymon.profiling.extractors.structured.element import ElementExtractor

from raymon.profiling.extractors.structured.scoring import (
    AbsoluteRegressionError,
    ClassificationErrorType,
)
from raymon.profiling.scores import MeanScore, PrecisionScore, RecallScore
from models import ModelOracle, RetinopathyMockModel
from PIL import ImageFilter
from PIL import ImageEnhance


ImageFile.LOAD_TRUNCATED_IMAGES = True
ROOT = Path("..").resolve()
LABELPATH = ROOT / "data/trainLabels.csv"
DATA_PATH = Path("../data/1/")
LIM = 500

#%%
def load_data(dpath, lim):
    files = dpath.glob("*.jpeg")
    images = []
    metadata = []
    for n, fpath in enumerate(files):
        if n == lim:
            break
        img = Image.open(fpath)
        img.thumbnail(size=(500, 500))
        images.append(img)
        metadata.append([{"name": "srcfile", "value": fpath.stem, "type": "label"}])

    return images, metadata


def get_preds_actuals(metadata, loaded_data):
    oracle = ModelOracle(labelpath=LABELPATH)
    model = RetinopathyMockModel(oracle=oracle)
    model.train(loaded_data)

    preds = []
    for meta, img in zip(metadata, loaded_data):
        pred = model.predict(data=img, metadata=meta)
        preds.append([pred])
    targets = []
    for meta in metadata:
        target = model.oracle.get_target(metadata=meta)
        targets.append([target])
    return preds, targets


# %%


loaded_data, metadata = load_data(dpath=DATA_PATH, lim=LIM)
preds, targets = get_preds_actuals(metadata, loaded_data)

profile = ModelProfile(
    name="retinopathy",
    version="3.0.0",
    components=[
        InputComponent(
            name="sharpness",
            extractor=Sharpness(),
            dtype=DataType.FLOAT,
        ),
        InputComponent(
            name="intensity",
            extractor=AvgIntensity(),
            dtype=DataType.FLOAT,
        ),
        InputComponent(
            name="outlierscore",
            extractor=DN2AnomalyScorer(k=20),
            dtype=DataType.FLOAT,
        ),
        OutputComponent(
            "model_prediction", extractor=ElementExtractor(0), dtype=DataType.INT
        ),
        ActualComponent(
            "model_actual", extractor=ElementExtractor(0), dtype=DataType.INT
        ),
        EvalComponent(
            "regression_error",
            extractor=AbsoluteRegressionError(),
            dtype=DataType.FLOAT,
        ),
        EvalComponent(
            "classification_error",
            extractor=ClassificationErrorType(positive=0),
            dtype=DataType.CAT,
        ),
    ],
    scores=[
        MeanScore(
            name="mean_absolute_error",
            inputs=["regression_error"],
            preference="low",
        ),
        PrecisionScore(
            name="precision",
            inputs=["classification_error"],
        ),
        RecallScore(
            name="recall",
            inputs=["classification_error"],
        ),
    ],
)


profile.build(input=loaded_data, output=preds, actual=targets, silent=False)
profile.view()
# profile.save(".")
#%%
