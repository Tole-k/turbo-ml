from functools import cache
import pickle
from typing import Literal

from turbo_ml.base.model import Model
from ..model_prediction.model_prediction import Predictor
from turbo_ml.base import __ALL_MODELS__
from ..dataset_parameters.dataset_characteristics import DatasetDescription
import pandas as pd
import torch
import torch.nn as nn
from turbo_ml.utils import device_detector

__MODELS__ = ["NeuralNetworkModel", "XGBoostClassifier", "AdaBoostClassifier", "BaggingClassifier", "BernoulliNB", "CalibratedClassifierCV", "CategoricalNB", "ComplementNB", "DecisionTreeClassifier", "DummyClassifier", "ExtraTreeClassifier", "ExtraTreesClassifier", "GaussianNB", "GaussianProcessClassifier", "GradientBoostingClassifier", "HistGradientBoostingClassifier", "KNeighborsClassifier",
              "LabelPropagation", "LabelSpreading", "LinearDiscriminantAnalysis", "LinearSVC", "LogisticRegression", "LogisticRegressionCV", "MLPClassifier", "MultinomialNB", "NearestCentroid", "NuSVC", "PassiveAggressiveClassifier", "Perceptron", "QuadraticDiscriminantAnalysis", "RadiusNeighborsClassifier", "RandomForestClassifier", "RidgeClassifier", "RidgeClassifierCV", "SGDClassifier", "SVC"]


class Best_Model(nn.Module):
    def __init__(self, num_features: int, num_classes: int):
        super(Best_Model, self).__init__()
        self.fc1 = nn.Linear(num_features, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x


class MetaModelGuesser(Predictor):
    """ Search for the best meta model for a given dataset and model """

    def __init__(self, device: Literal['cpu', 'cuda', 'mps'] = 'auto'):
        self.device = device_detector(device)
        self._path = str(__file__)[:-20] + 'model/'
        # Do not rename this file (-20 is length of file name, model.pth is expected to be in the same directory)
        # in order to not exclude windows \ options
        self._meta_model = self._load_meta_model()
        self._preprocessor = self._load_preprocessor()

    def predict(self, dataset_params: dict) -> Model:
        frame = pd.DataFrame([dataset_params])
        frame.drop(columns=['task'], axis=1, inplace=True)
        pre_frame = self._preprocessor.transform(frame)
        train = torch.tensor(pre_frame.values.astype(
            'float32')).to(self.device)

        with torch.inference_mode():
            model_values = self._meta_model(train).cpu()[0]
        models = self._find_models(model_values, 2)
        return models[0]

    def _find_models(self, model_values: list, n: int = 1) -> list:
        model_list = [(idx, float(i)) for idx, i in enumerate(model_values)]
        model_list.sort(key=lambda x: x[1], reverse=True)
        best_models = model_list[:n]
        models_names = [__MODELS__[idx] for idx, _ in best_models]
        translate = MetaModelGuesser._get_str_to_model_dict()
        return list(map(lambda x: translate[x], models_names))

    def _load_meta_model(self):
        model = Best_Model(15, 36).to(self.device)
        model.load_state_dict(torch.load(self._path + 'model.pth'))
        return model.eval()

    def _load_preprocessor(self):
        with open(self._path + 'preprocessor.pkl', 'rb') as f:
            preprocessor = pickle.load(f)
        return preprocessor

    @cache
    @staticmethod
    def _get_str_to_model_dict():
        return {model.__name__: model for model in __ALL_MODELS__}
