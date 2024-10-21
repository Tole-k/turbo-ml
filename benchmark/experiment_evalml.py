from evalml.automl import AutoMLSearch
import pandas
from utils import BaseExperiment, MODEL_NAMES

class EvalML(BaseExperiment):
    def __init__(self):
        self.name = self.__class__.__name__
        self.task_mapping = {"multiclass_classification": "multiclass", "binary_classification": "binary"}
    
    def find_model_in_string(self, string):
        for model in MODEL_NAMES:
            if model.lower() in string.lower().replace(" ", ""):
                return model
        return None
    
    def find_best_models(self, dataset_path, task, duration):
        df = pandas.read_csv(dataset_path)
        last_col = df.columns[-1]
        y = df[last_col]
        X = df.drop(last_col, axis=1)
        automl = AutoMLSearch(X_train=X, y_train=y, problem_type=task, max_time=duration)
        automl.search()
        for pipeline in automl.rankings['pipeline_name']:
            if model := self.find_model_in_string(pipeline):
                return model
        

if __name__ == "__main__":
    experiment = EvalML()
    experiment.perform_experiment()