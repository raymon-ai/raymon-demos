import pandas as pd
import random


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


class RetinopathyMockModel:
    def __init__(self, oracle, bad_machines):
        self.oracle = oracle
        self.bad_machines = bad_machines

    def get_machine(self, metadata):
        for tag in metadata:
            if tag["name"] == "machine_id":
                return tag["value"]

    def predict(self, data, metadata):
        if self.get_machine(metadata) in self.bad_machines:
            p_corr = 0.2
        else:
            p_corr = 0.95
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
