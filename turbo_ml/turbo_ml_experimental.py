""" Experimental Turbo-ML pipeline, newer but not fully tested version.
If something from here is not working, or you're getting some errors DO NOT make github issues.
This version is made mainly for testing new pipelines for newer version.
As this version does not provide logging or error handling it may be faster
"""
import pandas as pd

from typing import Literal, Optional

from turbo_ml.preprocessing import sota_preprocessor
from turbo_ml.meta_learning import MetaModelGuesser, get_sota_meta_features
from turbo_ml.hpo import HyperTuner
from turbo_ml.meta_learning.dataset_parameters import SimpleMetaFeatures
from turbo_ml.algorithms import RandomGuesser as DummyModel
from turbo_ml.base import Model
from turbo_ml.utils import options


class TurboML_Experimental:
    def __init__(self, dataset: pd.DataFrame, target: Optional[str] = None, device: Literal['cpu', 'cuda', 'mps', 'auto'] = 'auto', threads: int = 1, hpo_enabled: bool = False,
                 guesser: MetaModelGuesser = None, tuner: HyperTuner = None, param_function: Optional[callable] = None):
        if guesser is None:
            guesser = MetaModelGuesser()
        if tuner is None:
            tuner = HyperTuner()
        if param_function is None:
            param_function = get_sota_meta_features()
        options.device = device
        options.threads = threads
        self._algorithm = DummyModel
        self.model: Model
        self.hyperparameters = {}
        self._input_check(dataset, target)
        target_data = dataset[target]
        data = dataset.drop(columns=[target])

        self.preprocessor = sota_preprocessor()
        data = self.preprocessor.fit_transform(data)
        target_data = self.preprocessor.fit_transform_target(target_data)

        dataset_params = param_function(data, target_data, as_dict=True)

        self._algorithm = guesser.predict(dataset_params)

        if hpo_enabled:
            self.hyperparameters = tuner.optimize_hyperparameters(
                self._algorithm, (data, target_data), dataset_params['task'], dataset_params['num_classes'], dataset_params['target_features'])
        self.model = self._algorithm(**self.hyperparameters)

        self.model.train(data, target_data)

    def _input_check(self, dataset: pd.DataFrame, target: str):
        assert dataset is not None and isinstance(dataset, pd.DataFrame)
        assert len(dataset) > 0
        assert target is not None and isinstance(target, str)
        assert target in dataset.columns

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """
        Generates predictions using the trained model.

        Args:
            X (pd.DataFrame): A DataFrame containing the input features for prediction.

        Returns:
            pd.Series: A Series containing the predicted values.
        """
        X = self.preprocessor.transform(X)
        result = self.model.predict(X)
        return result  # TODO: inverse transform target

    def __call__(self, X: pd.DataFrame) -> pd.Series:
        """
        Generates predictions using the trained model. Call method is just wrapper for predict method.

        Args:
            X (pd.DataFrame): A DataFrame containing the input features for prediction.

        Returns:
            pd.Series: A Series containing the predicted values.
        """
        return self.predict(X)
