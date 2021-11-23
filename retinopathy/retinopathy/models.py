import pandas as pd
import random


class RetinopathyMockModel:
    def __init__(self, oracle):
        self.oracle = oracle

    def predict(self, data, metadata, p_corr):
        iscorr = random.random() <= p_corr
        target = self.oracle.get_target(metadata)
        if iscorr:
            pred = target
        else:
            choices = {0, 1, 2, 3, 4}
            choices = choices.difference({target})
            pred = random.choice(list(choices))
        return pred

    def train(self, data):
        pass


class ModelOracle:
    def __init__(self, labelpath):
        self.labels = pd.read_csv(labelpath)

    def get_src(self, metadata):
        for tag in metadata:
            if tag["name"] == "srcfile":
                return tag["value"]

    def get_target(self, metadata):
        srcfile = self.get_src(metadata)
        return int(self.labels.loc[self.labels["image"] == srcfile, "level"].values[0])
