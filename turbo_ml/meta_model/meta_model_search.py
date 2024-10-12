import os
from functools import cache
from ..model_prediction.model_prediction import Predictor
from ..preprocessing import sota_preprocessor
from ..base import __ALL_MODELS__
from ..model_prediction.dataset_characteristics import DatasetDescription
import pandas as pd
import torch
import torch.nn as nn

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

    def __init__(self, device='cpu'):
        self.device = device
        self._meta_model = self._load_meta_model()

    def predict(self, dataset_params: DatasetDescription):
        frame = pd.DataFrame([dataset_params.dict()])
        frame.drop(columns=['task'], axis=1, inplace=True)
        preprocessor = sota_preprocessor()
        pre_frame = preprocessor.fit_transform(frame)

        train = torch.tensor(pre_frame.values.astype(
            'float32')).to(self.device)
        self._meta_model.eval()
        with torch.inference_mode():
            model_values = self._meta_model(train).cpu()[0]
        model_list = [float(i) for i in model_values]
        best = model_list.index(max(model_list))
        model_name = __MODELS__[best]
        best_model = MetaModelGuesser._get_str_to_model_dict()[model_name]

        return best_model()

    def _load_meta_model(self):
        model = Best_Model(15, 36).to(self.device)
        model.load_state_dict(torch.load(
            str(__file__)[:-20] + 'model.pth'))
        return model

    @cache
    @staticmethod
    def _get_str_to_model_dict():
        return {model.__name__: model for model in __ALL_MODELS__}
