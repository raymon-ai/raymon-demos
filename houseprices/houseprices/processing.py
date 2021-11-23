# %%
# %load_ext autoreload
# %autoreload 2
import json
import os
import pickle
import random
import time
from pathlib import Path
import random
import pendulum
import uuid
import time

# Import packages
import pandas as pd
from raymon import Trace, RaymonAPILogger, Tag
from raymon import types as rt
from raymon.profiling import ModelProfile

from houseprices.io import load_data

ROOT = Path("..")
ENV = os.environ.get("ENV", ".dev")
N_RAYS = int(os.environ.get("RAYMON_N_TRACES", 1000))
PROJECT_ID = os.environ.get("PROJECT_ID", "cc025a4a-8189-4bd7-92ae-5a0b6f9d6840")
RAYMON_URL = os.environ.get(
    "RAYMON_ENDPOINT", "http://localhost:15000/v0"
)  # "https://api.raymon.ai/v0"  # "http://localhost:5000/v0")
SECRET = Path(
    os.environ.get(
        "RAYMON_CLIENT_SECRET_FILE", ROOT / f"m2mcreds-houseprices{ENV}.json"
    )
).resolve()

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

print(f"Using endpoint: {RAYMON_URL}")
print(f"Secret path: {SECRET}, exists? {SECRET.exists()}")
# %%
# Load data
def load_client_data(client_type="cheap"):
    X, y = load_data(ROOT / f"data/subset-{client_type}-test.csv")

    return X, y


def load_pkl(fpath):
    with open(fpath, "rb") as f:
        return pickle.load(f)


def load_profile(fpath):
    with open(fpath, "r") as f:
        profile = ModelProfile.from_jcr(json.load(f))
        return profile


rf = load_pkl(ROOT / "models/HousePrices-RF-v3.0.0.pkl")
coltf = load_pkl(ROOT / "models/HousePrices-RF-v3.0.0-coltf.pkl")
feature_selector_ohe = load_pkl(ROOT / "models/HousePrices-RF-v3.0.0-ohe-sel.pkl")
feature_selector = load_pkl(ROOT / "models/HousePrices-RF-v3.0.0-sel.pkl")
profile = load_profile(ROOT / "models/housepricescheap@3.0.0.json")

# %%
class ClientKWR:
    def __init__(self, df):
        self.df = df
        self.idx = 0
        self.metadata = {"client": "KWR", "app": "v2.0.0"}

    def send_data(self):
        print("Sending KWR data")
        data = self.df.sample().iloc[0, :]
        rid = f"{uuid.uuid4()}@{data['Id']}"
        data = data.drop("Id")
        self.idx += 1
        return rid, data, self.metadata


class ClientZillow:
    def __init__(self, df):
        self.df = df
        self.idx = 0
        self.metadata = {"client": "Zillow", "app": "v1.0.0"}

    def send_data(self):
        print("Sending Zillow data")
        data = self.df.sample().iloc[0, :]
        rid = f"{uuid.uuid4()}@{data['Id']}"
        data = data.drop("Id")
        self.idx += 1
        return rid, data, self.metadata


class ClientRemax:
    def __init__(self, df, switch_begin, switch_end):
        self.df = df
        self.idx = 0
        self.switch_begin = switch_begin
        self.switch_end = switch_end

    def send_data_oldapp(self):
        data = self.df.sample().iloc[0, :]
        rid = f"{uuid.uuid4()}@{data['Id']}"
        data = data.drop("Id")
        self.idx += 1
        return rid, data, {"client": "Remax", "app": "v1.0.0"}

    def send_data_newapp(self):
        data = self.df.sample().iloc[0, :]
        rid = f"{uuid.uuid4()}@{data['Id']}"
        data = data.drop("Id")
        # Corrupt the data
        to_drop = ["OverallQual", "GrLivArea"]
        data[to_drop] = 0
        self.idx += 1
        return rid, data, {"client": "Remax", "app": "v2.0.0"}

    def send_data(self):
        print(f"Sending Remax data for idx {self.idx}")
        # See which app to send from
        if self.idx < self.switch_begin:
            print(f"Before switch.")
            return self.send_data_oldapp()
        elif self.idx > self.switch_end:
            print(f"After switch.")
            return self.send_data_newapp()
        else:
            print(f"During switch")
            # User are switching linearly between switch begin and switch end.
            p_switched = (
                1
                / (self.switch_end - self.switch_begin)
                * (self.idx - self.switch_begin)
            )
            if random.random() >= p_switched:
                print(f"Old app.")
                return self.send_data_oldapp()
            else:
                print(f"New app")
                return self.send_data_newapp()


class HousePricingDeployment:
    def __init__(self, version, prep, model, profile):
        self.version = version
        self.prep = prep
        self.model = model
        self.profile = profile
        self.raymon = RaymonAPILogger(
            url=RAYMON_URL,
            project_id=PROJECT_ID,
            auth_path=SECRET,
            env=RAYMON_ENV,
            batch_size=100,
        )

    def add_metadata(self, trace, metadata):
        tags = [
            {
                "type": "label",
                "name": key,
                "value": value,
            }
            for key, value in metadata.items()
        ]

        tags = tags + [
            {
                "type": "label",
                "name": "deployment_version",
                "value": self.version,
            }
        ]

        trace.tag(tags)

    def add_timing_tags(self, trace, pred_start, pred_end):
        trace.tag(
            [
                {
                    "type": "metric",
                    "name": "pred_time",
                    "value": pred_end - pred_start,
                }
            ]
        )

    def process(self, req_id, data, metadata):
        trace = Trace(logger=self.raymon, trace_id=str(req_id))
        print(f"Logging...")
        try:
            # Let's log some segmentator tags: model id and deployment version
            self.add_metadata(trace, metadata)
            trace.info(f"Received prediction request.")
            trace.log(ref="request_data", data=rt.Series(data))
            # validate data
            input_tags = self.profile.validate_input(input=data)
            trace.tag(input_tags)
            # Convert to pandas series
            datapd = pd.Series(data).to_frame().transpose()
            # Transform the data
            pred_start = time.time()
            prep = self.prep.transform(datapd)
            prep_df = pd.DataFrame(prep, columns=feature_selector_ohe)
            trace.log(ref="preprocessed_input", data=rt.DataFrame(prep_df))
            # Model prediction
            pred_arr = self.model.predict(prep_df)
            pred = float(pred_arr[0])
            pred_end = time.time()
            trace.info(f"Pred: {pred}, {type(pred)}")
            trace.log(ref="pricing_prediction", data=rt.Native(pred))
            output_tags = self.profile.validate_output(output=pred_arr)
            trace.tag(output_tags)

            self.add_timing_tags(trace=trace, pred_start=pred_start, pred_end=pred_end)
            self.raymon.flush()

            return pred
        except Exception as exc:
            print(f"Exception for req_id {req_id}: {exc}")
            # raise
            trace.tag([Tag(type="error", name="processing_error", value=str(exc))])


class Oracle:
    def __init__(self, y, y_zillow):
        self.raymon = RaymonAPILogger(
            url=RAYMON_URL,
            project_id=PROJECT_ID,
            auth_path=SECRET,
            env=RAYMON_ENV,
            batch_size=100,
        )
        self.y = y
        self.y_zillow = y_zillow

    def process(self, req_id, metadata):
        trace = Trace(logger=self.raymon, trace_id=str(req_id))
        trace.info(f"Logging ground truth for {trace}")
        idx = int(req_id.split("@")[1])
        if metadata["client"].lower() == "zillow":
            target = float(self.y_zillow.loc[idx])
        else:
            target = float(self.y.loc[idx])
        trace.log(ref="actual", data=rt.Native(target))
        actual_tags = profile.validate_actual(actual=[target])
        trace.tag(actual_tags)
        self.raymon.flush()

        return target


def run_process():
    X, y = load_client_data(client_type="cheap")
    ratio_remax = 0.2
    split = int((1 - ratio_remax) * len(X))
    switch_begin = int((ratio_remax) * N_RAYS * 0.3)
    switch_end = int((ratio_remax) * N_RAYS * 0.7)
    print(
        f"Client remax will start switching at: {switch_begin}, end at: {switch_end} out"
    )

    X_kwr = X.iloc[:split, :]
    X_remax = X.iloc[split:, :]

    X_zillow, y_zillow = load_client_data(client_type="exp")
    # create our deployement
    deployment = HousePricingDeployment(
        version="housepricing-1.0.0",
        prep=coltf,
        model=rf,
        profile=profile,
    )
    oracle = Oracle(y=y, y_zillow=y_zillow)
    client_kwr = ClientKWR(
        df=X_kwr,
    )
    client_zillow = ClientZillow(
        df=X_zillow,
    )
    client_remax = ClientRemax(
        df=X_remax, switch_begin=switch_begin, switch_end=switch_end
    )

    # Create a client, fetch data and send it to the deployment
    trace_ids = []
    for i in range(N_RAYS):
        if i % 5 <= 1:
            req_id, data, metadata = client_remax.send_data()
        elif i % 5 == 2:
            req_id, data, metadata = client_zillow.send_data()
        else:
            req_id, data, metadata = client_kwr.send_data()

        pred = deployment.process(req_id=req_id, data=data, metadata=metadata)
        actual = oracle.process(req_id, metadata=metadata)
        scores = profile.validate_eval(output=pred, actual=actual)
        # We could post this here, but it would ruin the example. We want to demonstrate the mappers!
        # api.post()
        trace_ids.append(req_id)
        # time.sleep(0.2)
    return trace_ids


#%%
if __name__ == "__main__":
    start_ts = pendulum.now()
    trace_ids = run_process()
    end_ts = pendulum.now()
    print(f"Start: {str(start_ts)}, End: {str(end_ts)}")


# %%
# X, y = load_client_data(client_type="cheap")
# ratio_remax = 0.2
# split = int((1 - ratio_remax) * len(X))
# switch_begin = int(ratio_remax * N_RAYS * 0.3)
# switch_end = int(ratio_remax * N_RAYS * 0.7)
# %%
