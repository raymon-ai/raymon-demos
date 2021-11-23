#%%
import os
import random
import uuid
import traceback
import pendulum
from PIL import Image

from models import ModelOracle, RetinopathyMockModel
from const import TAG_CHOICES, BAD_MACHINE, ROOT, SECRET, LABELPATH


class RetinopathyDeployment:
    def __init__(self, version, model):
        self.version = version
        self.model = model

    def process(self, trace_id, data, metadata, p_corr):
        try:
            resized_img = data.resize((512, 512))
            pred = self.model.predict(resized_img, metadata, p_corr=p_corr)
            print(f"Processed trace {trace_id}. Prediction: {pred}")
            return pred
        except Exception as exc:
            print(traceback.format_exc())


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
    for i in range(N_RAYS):
        trace_id = str(uuid.uuid4())
        metadata = pick_tags(TAG_CHOICES)
        idx = i % len(files)
        imgpath = files[idx]
        metadata.append({"name": "srcfile", "value": imgpath.stem, "type": "label"})
        img = Image.open(imgpath)
        img.thumbnail(size=(500, 500))
        p_corr = 0.95
        pred = deployment.process(
            trace_id=trace_id, data=img, metadata=metadata, p_corr=p_corr
        )


#%%
N_RAYS = int(os.environ.get("RAYMON_N_RAYS", 100))
VERSION = "retinopathy@3.0.0"

files = list((ROOT / "data/1").glob("*.jpeg"))
model_oracle = ModelOracle(labelpath=LABELPATH)
model = RetinopathyMockModel(oracle=model_oracle)
deployment = RetinopathyDeployment(version=VERSION, model=model)


start_ts = pendulum.now()
trace_ids = run()
end_ts = pendulum.now()
print(f"Start: {str(start_ts.in_tz('utc'))}, End: {str(end_ts.in_tz('utc'))}")
#%%
