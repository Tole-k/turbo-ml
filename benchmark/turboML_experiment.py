import pandas as pd

from collections import defaultdict

from utils import BaseExperiment, _FAMILY_MAPPING

from turbo_ml.meta_learning.meta_model import as_meta_model
from turbo_ml.preprocessing import sota_preprocessor
from turbo_ml.meta_learning import MetaModelGuesser, sota_dataset_parameters
from turbo_ml.algorithms import RandomGuesser as DummyModel
from turbo_ml.base import Model, __ALL_MODELS__
from turbo_ml.utils import options


class TurboMLExperiment(BaseExperiment):
    def __init__(self):
        self.name = self.__class__.__name__
        family_scores = pd.read_csv("data/family_scores.csv")
        parameters = pd.read_csv("data/parameters.csv")
        self.data = pd.merge(family_scores, parameters, on="name")


    def rank_families(self, dataset, dataset_name, *_):
        training_frame = self.data[self.data["name"] != dataset_name].copy()
        as_meta_model.train_meta_model(save_model=True, device="cpu",
            save_path="turbo_ml/meta_learning/meta_model", frame=training_frame
        )
        options.device = "cpu"
        options.threads = 1
        target = dataset.columns[-1]
        target_data = dataset[target]
        data = dataset.drop(columns=[target])
        try:
            preprocessor = sota_preprocessor()
            data = preprocessor.fit_transform(data)
            target_data = preprocessor.fit_transform_target(target_data)
        except Exception:
            return None
        try:
            dataset_params = sota_dataset_parameters(
                data, target_data, as_dict=True, old=True)
        except Exception:
            return None
        try:
            guesser = MetaModelGuesser()
            models = guesser.predict(dataset_params)
            if not models:
                return None
        except Exception  as e:
            print(e)
            return None
        return [_FAMILY_MAPPING[model] for model in models]

if __name__ == "__main__":
    experiment = TurboMLExperiment()
    experiment.perform_experiments([0], [60])
