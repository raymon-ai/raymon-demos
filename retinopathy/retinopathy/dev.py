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
