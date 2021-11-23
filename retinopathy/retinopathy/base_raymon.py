#%%
import os
import random
import uuid
import traceback

import pendulum
from PIL import Image
from raymon import Trace, RaymonAPILogger
from raymon import types as rt
from raymon import ModelProfile
from PIL import ImageFilter

from .models import ModelOracle, RetinopathyMockModel
from .const import TAG_CHOICES, BAD_MACHINE, ROOT, SECRET, LABELPATH

N_RAYS = int(os.environ.get("RAYMON_N_RAYS", 100))
PROJECT_ID = os.environ.get("PROJECT_ID", "c14005c0-c57d-492c-8339-53cc694cb743")
RAYMON_URL = os.environ.get("RAYMON_ENDPOINT", "https://api.raymon.ai/v0/")
VERSION = "retinopathy@3.0.0"

#%%
class RetinopathyDeployment:
    def __init__(self, version, model, profile):
        self.version = version
        self.model = model
        self.profile = profile
        self.raymon = RaymonAPILogger(
            url=RAYMON_URL,
            project_id=PROJECT_ID,
            auth_path=SECRET,
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

    def process(self, trace_id, data, metadata, p_corr):
        trace = Trace(logger=self.raymon, trace_id=trace_id)
        try:
            self.add_metadata(trace, metadata)
            trace.log(ref="request_data", data=rt.Image(data))
            tags = self.profile.validate_input(data)
            trace.tag(tags)
            resized_img = data.resize((512, 512))
            trace.log(ref="resized_data", data=rt.Image(resized_img))
            pred = self.model.predict(resized_img, metadata, p_corr=p_corr)
            trace.log(ref="model_prediction", data=rt.Native(pred))
            trace.tag(self.profile.validate_output([pred]))
            trace.logger.flush()
            return pred
        except Exception as exc:
            # raise
            trace.tag(
                [{"type": "err", "name": "Processing Exception", "value": str(exc)}]
            )
            trace.info(traceback.format_exc())
            trace.logger.flush()


def pick_tags(TAG_CHOICES):
    tags = []
    for key in ["age", "hospital", "eye"]:
        value = random.choice(TAG_CHOICES[key])
        tags.append({"name": key, "value": value, "type": "label"})
        if key == "hospital":
            machine_id = random.choice(TAG_CHOICES["machine_id"][value])
            tags.append({"name": "machine_id", "value": machine_id, "type": "label"})
    return tags


def get_machine(metadata):
    for tag in metadata:
        if tag["name"] == "machine_id":
            return tag["value"]


def run():
    # Create a client, fetch data and send it to the deployment
    for i in range(N_RAYS):
        trace_id = str(uuid.uuid4())
        metadata = pick_tags(TAG_CHOICES)
        idx = i % len(files)
        imgpath = files[idx]
        metadata.append({"name": "srcfile", "value": imgpath.stem, "type": "label"})
        img = Image.open(imgpath)
        img.thumbnail(size=(500, 500))
        if get_machine(metadata) == BAD_MACHINE:
            img = img.filter(ImageFilter.GaussianBlur(radius=10))
            p_corr = 0.2
        else:
            p_corr = 0.95
        pred = deployment.process(
            trace_id=trace_id, data=img, metadata=metadata, p_corr=p_corr
        )


#%%
print(f"Running for project id: {PROJECT_ID}")
print(f"Using endpoint: {RAYMON_URL}")
print(f"Secret path: {SECRET}, exists? {SECRET.exists()}")

files = list((ROOT / "data/1").glob("*.jpeg"))
profile = ModelProfile().load(f"../models/{VERSION}.json")
model_oracle = ModelOracle(labelpath=LABELPATH)
model = RetinopathyMockModel(oracle=model_oracle, BAD_MACHINEs=[BAD_MACHINE])
deployment = RetinopathyDeployment(version=VERSION, model=model, profile=profile)


start_ts = pendulum.now()
trace_ids = run()
end_ts = pendulum.now()
print(f"Start: {str(start_ts.in_tz('utc'))}, End: {str(end_ts.in_tz('utc'))}")
