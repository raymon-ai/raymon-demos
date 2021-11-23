#%%
from pathlib import Path
from raymon.profiling import (
    ModelProfile,
    InputComponent,
    OutputComponent,
    ActualComponent,
    EvalComponent,
    MeanReducer,
)
from raymon.profiling.extractors.structured import generate_components, ElementExtractor
from raymon.profiling.extractors.structured.scoring import (
    AbsoluteRegressionError,
)
from houseprices.io import load_data

ROOT = Path("..")


def train(X_train, y_train):
    pass


#%%

# Load data
X_train, y_train = load_data(ROOT / "data/subset-cheap-train.csv")
X_test, y_test = load_data(ROOT / "data/subset-cheap-test.csv")


model = train(X_train=X_train, y_train=y_train)
y_pred = model.predict(X_test)

# Construct profile

profile = ModelProfile(
    name="housepricescheap",
    version="2.0.0",
    components=generate_components(X_test.dtypes, complass=InputComponent)
    + [
        OutputComponent(name="prediction", extractor=ElementExtractor()),
        ActualComponent(name="actual", extractor=ElementExtractor()),
        EvalComponent(name="abs_error", extractor=AbsoluteRegressionError()),
    ],
    reducers=[
        MeanReducer(
            name="MAE",
            inputs=["abs_error"],
            preferences={"abs_error": "low"},
            results=None,
        )
    ],
)
profile.build(input=X, output=y_pred[None, :], actual=y_test[None, :])
profile.save(ROOT / "models")
