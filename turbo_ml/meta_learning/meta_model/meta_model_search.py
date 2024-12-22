from functools import cache
import pickle

from turbo_ml.base.model import Model
from turbo_ml.utils import options
from ..model_prediction.model_prediction import Predictor
from turbo_ml.base import get_models_list
import pandas as pd
import torch
from .model_architecture import ModelArchitecture

__MODELS_NAMES__ = ["NeuralNetworkModel", "XGBoostClassifier", "AdaBoostClassifier", "BaggingClassifier", "BernoulliNB", "CalibratedClassifierCV", "CategoricalNB", "ComplementNB", "DecisionTreeClassifier", "DummyClassifier", "ExtraTreeClassifier", "ExtraTreesClassifier", "GaussianNB", "GaussianProcessClassifier", "GradientBoostingClassifier", "HistGradientBoostingClassifier", "KNeighborsClassifier",
                    "LabelPropagation", "LabelSpreading", "LinearDiscriminantAnalysis", "LinearSVC", "LogisticRegression", "LogisticRegressionCV", "MLPClassifier", "MultinomialNB", "NearestCentroid", "NuSVC", "PassiveAggressiveClassifier", "Perceptron", "QuadraticDiscriminantAnalysis", "RadiusNeighborsClassifier", "RandomForestClassifier", "RidgeClassifier", "RidgeClassifierCV", "SGDClassifier", "SVC"]

__GROUP_NAMES__ = ["Bagging_(BAG)", "Bayesian_Methods_(BY)", "Boosting_(BST)", "Decision_Trees_(DT)", "Discriminant_Analysis_(DA)", "Generalized_Linear_Models_(GLM)", "Logistic_and_Multinomial_Regression_(LMR)", "Multivariate_Adaptive_Regression_Splines_(MARS)",
                   "Nearest_Neighbor_Methods_(NN)", "Neural_Networks_(NNET)", "Other_Ensembles_(OEN)", "Other_Methods_(OM)", "Partial_Least_Squares_and_Principal_Component_Regression_(PLSR)", "Random_Forests_(RF)", "Rule-Based_Methods_(RL)", "Stacking_(STC)", "Support_Vector_Machines_(SVM)"]


class MetaModelGuesser(Predictor):
    """ Search for the best meta model for a given dataset and model """

    def __init__(self, device='cpu', model=None, preprocessors=None, path: str = None):
        self.device = device
        self._path = path if path is not None else str(__file__)[
            :-20] + 'model/'
        # Do not rename this file (-20 is length of file name, model.pth is expected to be in the same directory)
        # in order to not exclude windows \ options
        self.device = options.device

        with open(self._path + 'model_params.pkl', 'rb') as f:
            self._config = pickle.load(f)
        if model is not None:
            self._meta_model = model
        else:
            self._meta_model = self._load_meta_model()
        if preprocessors is not None:
            self._preprocessor = preprocessors
        else:
            self._preprocessor = self._load_preprocessor()

    def predict(self, dataset_params: dict) -> Model:
        frame = pd.DataFrame([dataset_params])
        if 'task' in frame.columns:
            frame.drop(columns=['task'], axis=1, inplace=True)
        pre_frame = self._preprocessor.transform(frame)
        train = torch.tensor(pre_frame.values.astype(
            'float32')).to(self.device)

        with torch.inference_mode():
            model_values = self._meta_model(train).cpu()[0]
        models = self._find_models(model_values, 1)
        return models[0]

    def _find_models(self, model_values: list, n: int = 1) -> list:
        model_list = [(idx, float(i)) for idx, i in enumerate(model_values)]
        model_list.sort(key=lambda x: x[1], reverse=True)
        best_models = model_list[:n]
        models_names = [__MODELS_NAMES__[idx] for idx, _ in best_models]
        translate = MetaModelGuesser._get_str_to_model_dict()
        return list(map(lambda x: translate[x], models_names))

    def _load_meta_model(self):
        model = ModelArchitecture(self._config['input_size'],
                           self._config['output_size']).to(self.device)
        model.load_state_dict(torch.load(self._path + 'model.pth'))
        return model.eval()

    def _load_preprocessor(self):
        with open(self._path + 'preprocessor.pkl', 'rb') as f:
            preprocessor = pickle.load(f)
        return preprocessor

    @cache
    @staticmethod
    def _get_str_to_model_dict():
        return {model.__name__: model for model in get_models_list()}
