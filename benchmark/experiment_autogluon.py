import logging
from autogluon.tabular import TabularDataset, TabularPredictor
from utils import BaseExperiment, Task, ClassificationFamily


FAMILIES_MAPPING = {
    "RandomForest": ClassificationFamily.RANDOM_FOREST,
    "ExtraTrees": ClassificationFamily.RANDOM_FOREST,
    "KNeighbors": ClassificationFamily.NEAREST_NEIGHBOR_METHOD,
    "LightGBM": ClassificationFamily.BOOSTING,
    "CatBoost": ClassificationFamily.BOOSTING,
    "XGBoost": ClassificationFamily.BOOSTING,
    "NeuralNetTorch": ClassificationFamily.NEURAL_NETWORK,
    "LinearModel": ClassificationFamily.GENERALIZED_LINEAR_MODEL,
    "NeuralNetFastAI": ClassificationFamily.NEURAL_NETWORK,
    "Transformer": ClassificationFamily.NEURAL_NETWORK,
    "FTTransformer": ClassificationFamily.NEURAL_NETWORK,
    "TabPFN": ClassificationFamily.NEURAL_NETWORK,
    "VowpalWabbit": ClassificationFamily.NEURAL_NETWORK,
    "WeightedEnsemble": ClassificationFamily.OTHER_ENSEMBLE,
}


# There is not way to pass the seed to AutoGluon
class AutoGluonExperiment(BaseExperiment):
    def __init__(self):
        super().__init__()
        self.task_mapping = {Task.MULTICLASS: "multiclass", Task.BINARY: "binary"}

    def find_family_in_string(self, string: str) -> ClassificationFamily:
        for model, family in FAMILIES_MAPPING.items():
            if model in string:
                return family
        logging.warning(f"Model not found in string: {string}")
        return None

    def rank_families(self, dataset, _, task, seed, duration):
        if task not in self.task_mapping:
            raise NotImplementedError("Non classification task is not implemented")
        task = self.task_mapping[task]
        dataset = TabularDataset(dataset)
        predictor = TabularPredictor(
            label=dataset.columns[-1],
            path="AutoGluon-outputs",
            problem_type=task,
            eval_metric="accuracy",
        ).fit(dataset, time_limit=duration)
        families = []
        for model in predictor.leaderboard()["model"]:
            if family := self.find_family_in_string(model):
                families.append(family)
        return families


if __name__ == "__main__":
    experiment = AutoGluonExperiment()
    experiment.perform_experiments([1])
