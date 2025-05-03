from sageml.utils import options
from sageml.meta_learning import MetaModelGuesser, get_sota_meta_features
from sageml.preprocessing import sota_preprocessor
from sageml.workflow import train_meta_model
from .utils import BaseExperiment, _FAMILY_MAPPING, ClassificationFamily
import pandas as pd
import sys
sys.path.append('.')


_ALGO_2_FAMILY_MAPPING = {
    "RadiusNeighborsClassifier": ClassificationFamily.NEAREST_NEIGHBOR_METHOD,
    "CategoricalNB": ClassificationFamily.BAYESIAN_METHOD,
    "RidgeClassifier": ClassificationFamily.LOGISTIC_AND_MULTINOMINAL_REGRESSION,
    "GradientBoostingClassifier": ClassificationFamily.BOOSTING,
    "QuadraticDiscriminantAnalysis": ClassificationFamily.DISCRIMINANT_ANALYSIS,
    "XGBoostClassifier": ClassificationFamily.BOOSTING,
    "SGDClassifier": ClassificationFamily.SVM,  # default but depends on loss
    "AdaBoostClassifier": ClassificationFamily.BOOSTING,
    "GaussianNB": ClassificationFamily.BAYESIAN_METHOD,
    "RandomForestClassifier": ClassificationFamily.RANDOM_FOREST,
    "ExtraTreesClassifier": ClassificationFamily.RANDOM_FOREST,
    "RandomGuesser": ClassificationFamily.OTHER_METHOD,
    "BernoulliNB": ClassificationFamily.BAYESIAN_METHOD,
    "PassiveAggressiveClassifier": ClassificationFamily.LOGISTIC_AND_MULTINOMINAL_REGRESSION,  # ?
    "SVC": ClassificationFamily.SVM,
    "DummyClassifier": ClassificationFamily.OTHER_METHOD,
    "HistGradientBoostingClassifier": ClassificationFamily.BOOSTING,
    "GaussianProcessClassifier": ClassificationFamily.OTHER_METHOD,  # ?
    "LabelPropagation": ClassificationFamily.NEAREST_NEIGHBOR_METHOD,
    "LinearSVC": ClassificationFamily.SVM,
    "BaggingClassifier": ClassificationFamily.BAGGING,
    "NuSVC": ClassificationFamily.SVM,
    "LogisticRegression": ClassificationFamily.LOGISTIC_AND_MULTINOMINAL_REGRESSION,
    "ComplementNB": ClassificationFamily.BAYESIAN_METHOD,
    "LabelSpreading": ClassificationFamily.NEAREST_NEIGHBOR_METHOD,
    "DecisionTreeClassifier": ClassificationFamily.DECISION_TREE,
    "Perceptron": ClassificationFamily.LOGISTIC_AND_MULTINOMINAL_REGRESSION,  # wrapper on SGDClassifier
    "AdaBoostRegressor": ClassificationFamily.BOOSTING,
    "CalibratedClassifierCV": ClassificationFamily.OTHER_ENSEMBLE,  # ?
    "ExtraTreeClassifier": ClassificationFamily.DECISION_TREE,
    "KNeighborsClassifier": ClassificationFamily.NEAREST_NEIGHBOR_METHOD,
    "LinearDiscriminantAnalysis": ClassificationFamily.DISCRIMINANT_ANALYSIS,
    "LogisticRegressionCV": ClassificationFamily.LOGISTIC_AND_MULTINOMINAL_REGRESSION,
    "MLPClassifier": ClassificationFamily.NEURAL_NETWORK,
    "MultinomialNB": ClassificationFamily.BAYESIAN_METHOD,
    "NearestCentroid": ClassificationFamily.NEAREST_NEIGHBOR_METHOD,
    "RidgeClassifierCV": ClassificationFamily.LOGISTIC_AND_MULTINOMINAL_REGRESSION,
    "NeuralNetworkModel": ClassificationFamily.NEURAL_NETWORK,
}


class SageMLExperiment(BaseExperiment):
    def __init__(self, mode='algorithm'):
        self.name = self.__class__.__name__
        self.parameters = self._get_parameters()
        self.data = pd.read_csv("data/family_scores.csv")
        self.mode = mode

    def rank_families(self, dataset, dataset_name, *_):
        training_frame = self.data[self.data["name"] != dataset_name].copy()
        model, preprocessor_dataset = train_meta_model(evaluations_frame=training_frame, feature_frame=self.parameters)
        # required on macbook
        # options.device = "mps"
        options.threads = 1
        target = dataset.columns[-1]
        target_data = dataset[target]
        data = dataset.drop(columns=[target])
        preprocessor = sota_preprocessor()
        data = preprocessor.fit_transform(data)
        target_data = preprocessor.fit_transform_target(target_data)
        dataset_params = get_sota_meta_features(options.meta_features)(
            data, target_data, as_dict=True)
        guesser = MetaModelGuesser(
            model=model, preprocessors=preprocessor_dataset, mode=self.mode)
        models = guesser.predict(dataset_params)
        result = [_ALGO_2_FAMILY_MAPPING[model.__name__]
                  for model in [models]] if self.mode == 'algorithm' else [_FAMILY_MAPPING[model] for model in [models]]
        return result


if __name__ == "__main__":
    experiment = SageMLExperiment(mode='family')
    experiment.perform_experiments(durations=[30], seeds=[0])
