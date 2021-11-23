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


from models import ModelOracle, RetinopathyMockModel
from const import TAG_CHOICES, BAD_MACHINE, ROOT, SECRET, LABELPATH


class RetinopathyDeployment:
    def __init__(self, version, model, profile):
        self.version = version
        self.model = model
        self.raymon = RaymonAPILogger(
            url=RAYMON_URL,
            project_id=PROJECT_ID,
            batch_size=20,
        )
        self.profile = profile

    def process(self, trace_id, data, metadata, p_corr):
        trace = Trace(logger=self.raymon, trace_id=trace_id)
        trace.tag(metadata)
        try:
            trace.log(ref="request_data", data=rt.Image(data))
            tags = self.profile.validate_input(data)
            trace.tag(tags)
            resized_img = data.resize((512, 512))
            trace.log(ref="resized_data", data=rt.Image(resized_img))
            pred = self.model.predict(resized_img, metadata, p_corr=p_corr)
            print(f"Processed trace {trace_id}. Prediction: {pred}")
            trace.tag(self.profile.validate_output([pred]))
            return pred
        except Exception as exc:
            print(traceback.format_exc())
            trace.tag(
                [{"type": "err", "name": "Processing Exception", "value": str(exc)}]
            )
        finally:
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
N_RAYS = int(os.environ.get("RAYMON_N_TRACES", 50))
VERSION = "retinopathy@3.0.0"
PROJECT_ID = "4854ecdf-725e-4627-8600-4dadf1588072"
RAYMON_URL = "https://api.raymon.ai/v0"

files = list((ROOT / "data/1").glob("*.jpeg"))
model_oracle = ModelOracle(labelpath=LABELPATH)
model = RetinopathyMockModel(oracle=model_oracle)
profile = ModelProfile().load(f"../models/{VERSION}.json")

deployment = RetinopathyDeployment(version=VERSION, model=model, profile=profile)


start_ts = pendulum.now()
trace_ids = run()
end_ts = pendulum.now()
print(f"Start: {str(start_ts.in_tz('utc'))}, End: {str(end_ts.in_tz('utc'))}")
#%%
