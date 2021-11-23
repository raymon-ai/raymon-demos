#%%

import os
import random
import uuid
from pathlib import Path
import traceback

import pandas as pd
import pendulum
from PIL import Image, ImageOps
from raymon import Trace, RaymonAPILogger
from raymon import types as rt
from raymon import ModelProfile
from PIL import ImageFilter

import sys

sys.path.insert(0, "/Users/kv/Raymon/Code/raymon-api/demonstrators/retinopathy")

from retinopathy.models import ModelOracle, RetinopathyMockModel


ROOT = Path("..").resolve()
N_RAYS = int(os.environ.get("RAYMON_N_RAYS", 100))
PROJECT_ID = os.environ.get("PROJECT_ID", "c14005c0-c57d-492c-8339-53cc694cb743")
RAYMON_URL = os.environ.get("RAYMON_ENDPOINT", "http://localhost:15000/v0")
ENV = os.environ.get("ENV", "dev")
SECRET = Path(
    os.environ.get("RAYMON_CLIENT_SECRET_FILE", ROOT / "m2mcreds-retinopathy.json")
)
LABELPATH = ROOT / "data/trainLabels.csv"
VERSION = "retinopathy@3.0.0"

# Only for local developer runs
if "dev" in ENV:
    RAYMON_ENV = {
        "auth_url": "https://raymon-dev.eu.auth0.com",
        "audience": "raymon-backend-api",
        "client_id": "3520tUZUQQ4Xy0cBsIr5ee6pZEfJH37Y",
    }
elif "staging" in ENV:
    RAYMON_ENV = {
        "auth_url": "https://raymon-staging.eu.auth0.com",
        "audience": "raymon-backend-api",
        "client_id": "RVqFoPtgLgYcvcFXsaQejHosnwtbaDvo",
    }
else:
    RAYMON_ENV = None


print(f"Running for project id: {PROJECT_ID}")
print(f"Using endpoint: {RAYMON_URL}")
print(f"Secret path: {SECRET}, exists? {SECRET.exists()}")

#%%
def pick_tags(tag_choices):
    tags = []
    for key in ["age", "hospital", "eye"]:
        value = random.choice(tag_choices[key])
        tags.append({"name": key, "value": value, "type": "label"})
        if key == "hospital":
            machine_id = random.choice(tag_choices["machine_id"][value])
            tags.append({"name": "machine_id", "value": machine_id, "type": "label"})
    return tags


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
bad_machine = tag_choices["machine_id"]["UZA"][0]

#%%
class Oracle:
    def __init__(self, model_oracle):
        self.raymon = RaymonAPILogger(
            url=RAYMON_URL,
            project_id=PROJECT_ID,
            auth_path=SECRET,
            env=RAYMON_ENV,
            batch_size=20,
        )
        self.model_oracle = model_oracle
        self.profile = profile

    def process(self, trace_id, metadata):
        trace = Trace(logger=self.raymon, trace_id=str(trace_id))
        trace.info(f"Logging ground truth for {trace}")
        target = self.model_oracle.get_target(metadata)
        trace.log(ref="actual", data=rt.Native(target))
        trace.tag(profile.validate_actual(actual=[target]))
        trace.logger.flush()
        return target


class RetinopathyDeployment:
    def __init__(self, version, model, profile):
        self.version = version
        self.model = model
        self.profile = profile
        self.raymon = RaymonAPILogger(
            url=RAYMON_URL,
            project_id=PROJECT_ID,
            auth_path=SECRET,
            env=RAYMON_ENV,
            batch_size=20,
        )

    def add_metadata(self, trace, metadata):
        tags = metadata + [
            {
                "type": "label",
                "name": "deployment_version",
                "value": self.version,
            }
        ]
        trace.tag(tags)

    def process(self, trace_id, data, metadata):
        trace = Trace(logger=self.raymon, trace_id=trace_id)
        try:
            self.add_metadata(trace, metadata)
            trace.info(f"Received prediction request.")
            trace.log(ref="request_data", data=rt.Image(data))
            tags = self.profile.validate_input(data)
            trace.tag(tags)
            resized_img = data.resize((512, 512))
            trace.log(ref="resized_data", data=rt.Image(resized_img))
            pred = self.model.predict(data, metadata)
            trace.info(f"Pred: {pred}, {type(pred)}")
            trace.log(ref="model_prediction", data=rt.Native(pred))
            trace.tag(self.profile.validate_output([pred]))
            trace.logger.flush()
            return pred
        except Exception as exc:
            # raise
            print(f"Exception for req_id {trace}: {exc}")
            # raise
            trace.tag(
                [{"type": "err", "name": "Processing Exception", "value": str(exc)}]
            )
            trace.info(traceback.format_exc())
            trace.logger.flush()


files = list((ROOT / "data/1").glob("*.jpeg"))
profile = ModelProfile().load(f"../models/{VERSION}.json")
model_oracle = ModelOracle(labelpath=LABELPATH)
model = RetinopathyMockModel(oracle=model_oracle, bad_machines=[bad_machine])
oracle = Oracle(model_oracle=model_oracle)
deployment = RetinopathyDeployment(version=VERSION, model=model, profile=profile)


def run():
    # Create a client, fetch data and send it to the deployment
    trace_ids = []
    for i in range(N_RAYS):
        trace_id = str(uuid.uuid4())
        metadata = pick_tags(tag_choices)
        idx = i % len(files)
        imgpath = files[idx]
        metadata.append({"name": "srcfile", "value": imgpath.stem, "type": "label"})
        img = Image.open(imgpath)
        img.thumbnail(size=(500, 500))
        if model.get_machine(metadata) == bad_machine:
            print(f"blurring {trace_id}...")
            img = img.filter(ImageFilter.GaussianBlur(radius=10))
        pred = deployment.process(trace_id=trace_id, data=img, metadata=metadata)
        actual = oracle.process(trace_id=trace_id, metadata=metadata)
        trace_ids.append(trace_id)
    return trace_ids


#%%
start_ts = pendulum.now()
trace_ids = run()
end_ts = pendulum.now()
print(f"Start: {str(start_ts.in_tz('utc'))}, End: {str(end_ts.in_tz('utc'))}")
