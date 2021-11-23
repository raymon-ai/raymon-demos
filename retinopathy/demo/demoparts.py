#%%
import glob
import io
import json
import math
import os
import random
import time
import uuid
from pathlib import Path

# Import packages
import numpy as np
import pandas as pd
import pendulum
import requests
import yaml
from PIL import Image, ImageOps
from random_word import RandomWords
from raymon import Ray, RaymonAPI
from raymon import types as rt
from rdv import Schema
from PIL import ImageFilter

#%%


#%%
# Deployment
tag_choices = {
    "age": range(30, 90),
    "hospital": ["St Jozef", "Maria Middelares", "UZA", "UZ Gent"],
    "eye": ["left", "right"],
    "machine_id": {
        "St Jozef": [
            "a466a0ff-da21-4522-816e-08f89bd213b4",
            "99d332c8-7064-44a8-a649-6c767bdb0ce9",
        ],
        "Maria Middelares": [
            "de576b99-6aea-4802-990f-c34b1cecb248",
            "fcb0d900-1a80-454f-84fc-7f4bdd1a3fbf",
        ],
        "UZA": [
            "2f930ab2-3e8a-4869-a157-1bc5cd327244",
            "c098be7e-b09a-4584-ad7d-13b505e4b0f3",
        ],
        "UZ Gent": [
            "1dbdc031-d805-4ffe-9800-e53aaddd02a7",
            "9e2e7870-1de4-40b6-914a-8b4ebc7edc07",
        ],
    },
}


class RetinopathyDeployment:
    def __init__(self, version, schema):
        self.version = version
        self.schema = schema
        self.raymon = RaymonAPI(url=raymon_api, project_id=PROJECT_NAME)

    def add_metadata(self, ray, metadata):
        tags = metadata + [
            {
                "type": "label",
                "name": "deployment_version",
                "value": self.version,
            }
        ]
        ray.tag(tags)

    def get_eye(self, metadata):
        for tag in metadata:
            if tag["name"] == "eye":
                return tag["value"]

    def model_predict(self, data, metadata):
        return model_predict(data, metadata)

    def process(self, ray_id, data, metadata):
        ray = Ray(logger=self.raymon, ray_id=ray_id)
        try:
            # Add metadata
            self.add_metadata(ray, metadata)
            # Save input data
            ray.info(f"Received prediction request.")
            ray.log(peephole="request_data", data=rt.ImageRGB(data))
            # Validate input data
            tags = self.schema.check(data)
            ray.tag(tags)
            # Do some processing
            if self.get_eye(metadata) == "left":
                ray.info("Image is a left eye, will mirror.")
                data = ImageOps.mirror(data)
            else:
                ray.info("Image is a right eye, will not mirror.")
            ray.log(peephole="flipped_data", data=rt.ImageRGB(data))
            # Resize image for the model
            resized_img = data.resize((512, 512))
            ray.log(peephole="resized_data", data=rt.ImageRGB(resized_img))
            # Predict
            pred = self.model_predict(data, metadata)
            # Log prediction
            ray.info(f"Pred: {pred}, {type(pred)}")
            ray.log(peephole="model_prediction", data=rt.Number(pred))
            return pred

        except Exception as exc:
            raise
            print(f"Exception for req_id {ray}: {exc}")
            # raise
            ray.tag([{"type": "err", "name": "Processing Exception", "value": type(exc)}])
            ray.info(traceback.format_exc())


#%%
# Data schema
data = load_data(dpath=DATA_PATH, lim=LIM)

schema = Schema(
    name="Retinopathy",
    version="1.0.0",
    components=[
        NumericComponent(name="sharpness", extractor=Sharpness()),
        NumericComponent(name="intensity", extractor=AvgIntensity()),
        NumericComponent(name="outlierscore", extractor=DN2OutlierScorer(k=16)),
    ],
)
schema.build(data=data)
schema.save(".")

#%%
# Oracle
class Oracle:
    def __init__(self, labelpath):
        self.raymon = RaymonAPI(url=raymon_api, project_id=PROJECT_NAME)
        self.labels = pd.read_csv(labels)

    def process(self, ray_id, metadata):
        ray = Ray(logger=self.raymon, ray_id=str(ray_id))
        ray.info(f"Logging ground truth for {ray}")
        dataid = get_src(metadata)
        target = self.get_target(dataid)
        ray.log(peephole="actual", data=rt.Number(target))

    def get_target(self, dataid):
        return int(self.labels.loc[self.labels["image"] == dataid, "level"].values[0])
