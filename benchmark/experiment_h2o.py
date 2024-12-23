from typing import List
import h2o
from h2o.automl import H2OAutoML
import pandas as pd
from utils import BaseExperiment, Task, ClassificationFamily

# Its not really working. Errors because number of rows is too small


class H2OExperiment(BaseExperiment):
    def __init__(self):
        super().__init__()
        h2o.init()

    def rank_families(self, dataset: pd.DataFrame, _, task: Task, seed, duration: int) -> List[ClassificationFamily]:
        if task is not Task.BINARY and task is not Task.MULTICLASS:
            raise NotImplementedError(
                "Non classification task is not implemented")
        dataset = h2o.H2OFrame(dataset)
        x = dataset.columns
        y = dataset.columns[-1]
        x.remove(y)
        if task is Task.BINARY:
            dataset[y] = dataset[y].asfactor()
        aml = H2OAutoML(max_runtime_secs=duration,
                        sort_metric="accuracy", seed=seed)
        aml.train(x=x, y=y, training_frame=dataset)
        lb = aml.leaderboard
        print(lb)
        for model in lb["model_id"]:
            if model := self.find_model_in_string(model):
                print(model)
                return model


if __name__ == "__main__":
    experiment = H2OExperiment()
    experiment.perform_experiments([1], [15*60])
