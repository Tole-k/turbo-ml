from turbo_ml.utils import options
from turbo_ml.base import __ALL_MODELS__
from turbo_ml.meta_learning import MetaModelGuesser, sota_dataset_parameters
from turbo_ml.preprocessing import sota_preprocessor
from turbo_ml.meta_learning.meta_model import as_meta_model
import pandas as pd


from utils import BaseExperiment, _FAMILY_MAPPING

import sys
sys.path.append('.')


class TurboMLExperiment(BaseExperiment):
    def __init__(self):
        self.name = self.__class__.__name__
        family_scores = pd.read_csv("data/family_scores.csv")
        parameters = pd.read_csv("data/parameters.csv")
        self.data = pd.merge(family_scores, parameters, on="name")

    def rank_families(self, dataset, dataset_name, *_):
        training_frame = self.data[self.data["name"] != dataset_name].copy()
        model, preprocessor_dataset = as_meta_model.train_meta_model(
            save_model=False, device="cpu", save_path=None, frame=training_frame)
        options.device = "cpu"
        options.threads = 1
        target = dataset.columns[-1]
        target_data = dataset[target]
        data = dataset.drop(columns=[target])
        preprocessor = sota_preprocessor()
        data = preprocessor.fit_transform(data)
        target_data = preprocessor.fit_transform_target(target_data)
        dataset_params = sota_dataset_parameters(
            data, target_data, as_dict=True, old=False)
        guesser = MetaModelGuesser(
            model=model, preprocessors=preprocessor_dataset)
        models = guesser.predict(dataset_params)
        return [_FAMILY_MAPPING[model] for model in [models]]


if __name__ == "__main__":
    experiment = TurboMLExperiment()
    experiment.perform_experiments(durations=[60])
