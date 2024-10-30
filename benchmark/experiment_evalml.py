import pandas as pd
from evalml.automl import AutoMLSearch

from utils import BaseExperiment, Task

class EvalMlExperiment(BaseExperiment):
    def __init__(self):
        self.name = "EvalML"
        self.task_mapping = {Task.MULTICLASS: "multiclass", Task.BINARY: "binary"}

    
    def find_best_model(self, dataset_path, task, duration, train_ratio=0.8):
        if task not in self.task_mapping:
            raise NotImplementedError("Non classification task is not implemented")
        task = self.task_mapping[task]
        df = pd.read_csv(dataset_path)
        last_col = df.columns[-1]
        y = df[last_col]
        X = df.drop(last_col, axis=1)
        # cant choose accuracy as eval metric
        automl = AutoMLSearch(X_train=X, y_train=y, problem_type=task, max_time=duration)
        automl.search()
        for pipeline in automl.rankings['pipeline_name']:
            if model := self.find_model_in_string(pipeline):
                return model
        

if __name__ == "__main__":
    experiment = EvalMlExperiment()
    experiment.perform_experiment()