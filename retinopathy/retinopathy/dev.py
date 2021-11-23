#%%
import uuid
from PIL import Image

from const import TAG_CHOICES, BAD_MACHINE, ROOT, SECRET, LABELPATH
from models import RetinopathyMockModel, ModelOracle


idx = 0
files = list((ROOT / "data/1").glob("*.jpeg"))
trace_id = str(uuid.uuid4())
imgpath = files[idx]
img = Image.open(imgpath)
metadata = [{"name": "srcfile", "value": imgpath.stem, "type": "label"}]

oracle = ModelOracle(labelpath=LABELPATH)
model = RetinopathyMockModel(oracle=oracle)
model.predict(data=img, metadata=metadata, p_corr=0.95)
oracle.get_target(metadata=metadata)
# %%
from raymon import RaymonAPI

with open("../manifest.yaml", "r") as f:
    cfg = f.read()
api = RaymonAPI(url=f"https://api.raymon.ai/v0")
resp = api.orchestration_apply(
    project_id="4854ecdf-725e-4627-8600-4dadf1588072", cfg=cfg
)
resp

# %%
from raymon import ModelProfile
from raymon import RaymonAPI

api = RaymonAPI(url=f"https://api.raymon.ai/v0")

schema = ModelProfile.load(ROOT / f"models/retinopathy@3.0.0.json")
resp = api.profile_create(
    project_id="4854ecdf-725e-4627-8600-4dadf1588072", profile=schema
)
resp.json()

# %%
