#%%
import requests
import pendulum

from raymon import types as rt
from raymon import Trace
from raymon.profiling import ModelProfile
from raymon.profiling.extractors.vision import DN2AnomalyScorer, AvgIntensity, Sharpness
from raymon.profiling.extractors.structured.scoring import (
    ClassificationErrorType,
)
from raymon.profiling.extractors.structured import ElementExtractor
from raymon.profiling.extractors.vision import DN2AnomalyScorer, AvgIntensity, Sharpness
from raymon.profiling.extractors.structured.element import ElementExtractor

from raymon.profiling.extractors.structured.scoring import (
    AbsoluteRegressionError,
    ClassificationErrorType,
)
from raymon.profiling.reducers import (
    MeanReducer,
    PrecisionRecallReducer,
)
from raymon.profiling import (
    ModelProfile,
    InputComponent,
    OutputComponent,
    ActualComponent,
    EvalComponent,
    DataType,
)

#%%
tags = {
    "age": 80,
    "hospital": "Maria Middelares",
    "eye": "right",
    "machine_id": "a466a0ff-da21-4522-816e-08f89bd213b4",
}

#%%
data = ...
predictions = ...
actuals = ...
profile = ModelProfile(
    name="retinopathy",
    version="2.0.0",
    components=[
        InputComponent(
            name="sharpness",
            extractor=Sharpness(),
            dtype=DataType.FLOAT,
        ),
        InputComponent(
            name="intensity",
            extractor=AvgIntensity(),
            dtype=DataType.FLOAT,
        ),
        InputComponent(
            name="outlierscore",
            extractor=DN2AnomalyScorer(k=20, size=(256, 256)),
            dtype=DataType.FLOAT,
        ),
        OutputComponent("model_prediction", extractor=ElementExtractor(0), dtype=DataType.INT),
        ActualComponent("model_actual", extractor=ElementExtractor(0), dtype=DataType.INT),
        EvalComponent(
            "regression_error",
            extractor=AbsoluteRegressionError(),
            dtype=DataType.FLOAT,
        ),
        EvalComponent(
            "classification_error",
            extractor=ClassificationErrorType(positive=0),
            dtype=DataType.CAT,
        ),
    ],
    reducers=[
        MeanReducer(
            name="mean_absolute_error",
            inputs=["regression_error"],
            preferences={"mean": "low"},
        ),
        PrecisionRecallReducer(
            name="precision_recall",
            inputs=["classification_error"],
        ),
    ],
)
profile.build(input=data, output=predictions, actual=actuals)
profile.save()


class RetinopathyDeployment:
    def process(self, data, metadata):
        trace = Trace(logger=self.raymon)
        # Log text message
        trace.info("Received prediction request.")
        # Save input data
        trace.log(ref="request_data", data=rt.Image(data))
        # Add metadata tags
        trace.tag(metadata)
        # Check incoming data and add data metrics to trace as tags
        data_metrics = self.profile.validate_input(data)
        trace.tag(data_metrics)
        # Predict
        pred = self.model_predict(data, metadata)
        # Log prediction
        trace.log(ref="model_prediction", data=pred)
        # Check predictions and add data metrics to trace as tags
        pred_metrics = self.profile.validate_output(output=pred)
        trace.tag(pred_metrics)
        return pred


#%%


#%%
# Oracle
class Feedback:
    def process_feedback(self, trace_id, feedback):
        jcr = {
            "timestamp": str(pendulum.now("utc")),
            "trace_id": str(trace_id),
            "ref": "actual",
            "data": feedback,
            "project_id": self.project_id,
        }
        requests.post(
            f"{self.url}/projects/{self.project_id}/ingest",
            json=jcr,
            headers=self.headers,
        )


#%%
