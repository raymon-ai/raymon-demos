#%%
import json
import os
import requests
import pendulum
from raymon import RaymonAPI
from raymon import types as rt


PROJECT_ID = "3a976e26-9a74-41e1-9607-af5aafbd6b46"

api = RaymonAPI(url="https://api.staging.raymon.ai/v0")

resp = api.object_search(
    project_id="3a976e26-9a74-41e1-9607-af5aafbd6b46",
    ray_id="04ab7326-b350-4dcd-a7d7-ddb5bcee122f",
    peephole="request_data",
)

img = rt.load_jcr(resp.json()["obj_data"])
img.data
# %%
begin = str(pendulum.parse("2021-03-23 00:00:00"))
end = str(pendulum.parse("2021-03-24 00:00:00"))
resp = api.trace_ls(project_id=PROJECT_ID, begin=begin, end=end, slicestr="outlierscore>=50", limit=250)
traces = resp.json()
len(traces)
# %%

resp = api.trace_get(project_id=PROJECT_ID, ray_id=rays[0]["ray_id"])
ray = resp.json()
ray["tags"]
# %%
peephole = [p for p in ray["trace"] if p["peephole"] == "request_data"][0]
rt.load_jcr(peephole["object"]).data
