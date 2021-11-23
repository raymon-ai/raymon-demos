#%%
import json
from raymon.auth import m2m
from raymon.auth import DEFAULT_ENV

from pathlib import Path
from raymon import RaymonAPI

DEMO = "retinopathy"
PROJECT_NAME = f"{DEMO}_demo"
ENV = ""  # ".dev"  # .staging"
SETUP_M2M = False
# api = RaymonAPI(url="http://localhost:8000/v0")
if "dev" in ENV:
    login_env = {
        "auth_url": "https://raymon-dev.eu.auth0.com",
        "audience": "raymon-backend-api",
        "client_id": "3520tUZUQQ4Xy0cBsIr5ee6pZEfJH37Y",
    }
    api = RaymonAPI(url="http://localhost:15000/v0", env=login_env)

elif "staging" in ENV:
    login_env = {
        "auth_url": "https://raymon-staging.eu.auth0.com",
        "audience": "raymon-backend-api",
        "client_id": "RVqFoPtgLgYcvcFXsaQejHosnwtbaDvo",
    }
    api = RaymonAPI(url=f"https://api{ENV}.raymon.ai/v0", env=login_env)

else:
    login_env = DEFAULT_ENV
    api = RaymonAPI(url=f"https://api{ENV}.raymon.ai/v0", env=login_env)


ROOT = Path(f"./{DEMO}")


project_id = None
resp = api.project_search(project_name=PROJECT_NAME)
if resp.ok:
    print(f"Project found")
    project_id = resp.json()["project_id"]
else:
    print(f"Creating new project")
    resp = api.project_create(project_name=PROJECT_NAME)
    project = resp.json()
    project_id = project["project_id"]

print(f"Project ID: {project_id}")
#%%
import json

if SETUP_M2M:
    resp = api.project_m2mclient_get(project_id=project_id)
    if resp.ok:
        print(f"M2M creds found")
        m2mcreds = resp.json()
    else:
        print(f"Creating m2m creds")
        resp = api.project_m2mclient_add(project_id=project_id)
        m2mcreds = resp.json()

    outpath = (ROOT / f"m2mcreds-{DEMO}{ENV}.json").resolve()

    print(f"Saving creds to {outpath.resolve()}")

    m2m.save_m2m_config(
        existing={},
        project_id=project_id,
        auth_endpoint=login_env["auth_url"],
        audience=login_env["audience"],
        client_id=m2mcreds[project_id]["client_id"],
        client_secret=m2mcreds[project_id]["client_secret"],
        grant_type="client_credentials",
        out=outpath,
    )


# %%
import yaml


def load_orch(fpath):
    with open(fpath, "r") as f:
        cfg = f.read()
    return cfg


cfg = load_orch(ROOT / "manifest.yml")
resp = api.orchestration_apply(project_id=project_id, cfg=cfg)
resp
# %%
from raymon import ModelProfile

if DEMO == "retinopathy":
    schema = ModelProfile.load(ROOT / f"models/{DEMO}@3.0.0.json")
    resp = api.profile_create(project_id=project_id, profile=schema)
    resp.json()
else:
    schema = ModelProfile.load(ROOT / f"models/housepricescheap@3.0.0.json")
    resp = api.profile_create(project_id=project_id, profile=schema)
    resp.json()


# %%

# %%
