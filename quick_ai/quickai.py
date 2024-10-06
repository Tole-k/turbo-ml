"""
quickai.py

This module provides the `QuickAI` class, our main class for out-of-the-box autoML solution.
It does not provide additional functionalities but it combines other modules to provide a complete solution.
"""
import pandas as pd
from .base import Model, __ALL_MODELS__
from .algorithms import RandomGuesser as DummyModel
from .forecast import StatisticalParametersExtractor, ExhaustiveSearch, HyperTuner
from .forecast.as_meta_model import main as meta_model_search
from .preprocessing import sota_preprocessor
from typing import Optional
import time
import logging
logging.basicConfig(level=logging.INFO)


class QuickAI:
    """
    The `QuickAI` class provides an out-of-the-box AutoML solution that automatically
    selects and trains the best machine learning model for a given dataset. It handles
    data validation, statistical parameter extraction, model selection, hyperparameter
    optimization, and model training.

    **Example:**

    ```python
    from quick_ai import QuickAI
    import pandas as pd

    # Load your dataset
    df = pd.read_csv('your_dataset.csv')

    # Initialize QuickAI with the dataset and target column
    quick_ai = QuickAI(dataset=df, target='target_column_name')

    # Prepare new data for prediction
    new_data = pd.read_csv('new_data.csv')

    # Make predictions
    predictions = quick_ai.predict(new_data)
    ```

    **Attributes:**
        model (Model): The machine learning model selected and trained on the dataset.
    """
    logger = logging.getLogger()
    logger.setLevel('INFO')

    def __init__(self, dataset: pd.DataFrame, target: Optional[str] = None, verbose: bool = True):
        """
        Initializes the `QuickAI` instance by performing the following steps:

        - Validates the input dataset and target column.
        - Extracts statistical parameters from the dataset.
        - Selects the best machine learning model based on dataset characteristics.
        - Optimizes hyperparameters (to be implemented).
        - Trains the selected model on the dataset.

        Args:
            dataset (pd.DataFrame): The input dataset containing features and the target variable.
            target (Optional[str]): The name of the target column in the dataset.

        Raises:
            NotImplementedError: If the target column is not provided.
            Exception: If dataset description, model optimization, or model training fails.

        Notes:
            - The `target` parameter is currently required. Automatic target detection is not yet implemented.
            - Model selection and hyperparameter optimization functionalities are placeholders and should be implemented.
        """
        self.logger.info("Initializing QuickAI...")
        self.model: Model = DummyModel()
        start_time = time.time()
        if target is None:
            # target = find_target() TODO: to be implemented
            raise NotImplementedError(
                "Target automatic detection is not implemented yet, provide target column name")
        self._input_check(dataset, target)
        target_data = dataset[target]
        data = dataset.drop(columns=[target])
        try:
            self.preprocessor = sota_preprocessor()
            data = self.preprocessor.fit_transform(data)
            target_data = self.preprocessor.fit_transform_target(target_data)
        except Exception:
            raise Exception("Preprocessing failed")
        if verbose:
            self.logger.info('Preprocessing completed')
        try:
            extractor = StatisticalParametersExtractor(data, target_data)
            dataset_params = extractor.describe_dataset()
        except Exception:
            raise Exception("Dataset description failed")
        if verbose:
            self.logger.info(
                'Dataset parameters found, trying to guess best model')
        data_operations = time.time()

        # try:
        # TODO implement model guessing based on dataset parameters
        import torch
        meta_model = meta_model_search()
        frame = pd.DataFrame([dataset_params.dict()])
        PARAMETERS = ["name", "task", "task_detailed", "target_features", "target_nans", "num_columns", "num_rows", "number_of_highly_correlated_features", "highest_correlation",
                      "number_of_lowly_correlated_features", "lowest_correlation", "highest_eigenvalue", "lowest_eigenvalue", "share_of_numerical_features", "num_classes", "biggest_class_freq", "smallest_class_freq"]

        Models = ["NeuralNetworkModel", "XGBoostClassifier", "AdaBoostClassifier", "BaggingClassifier", "BernoulliNB", "CalibratedClassifierCV", "CategoricalNB", "ComplementNB", "DecisionTreeClassifier", "DummyClassifier", "ExtraTreeClassifier", "ExtraTreesClassifier", "GaussianNB", "GaussianProcessClassifier", "GradientBoostingClassifier", "HistGradientBoostingClassifier", "KNeighborsClassifier",
                  "LabelPropagation", "LabelSpreading", "LinearDiscriminantAnalysis", "LinearSVC", "LogisticRegression", "LogisticRegressionCV", "MLPClassifier", "MultinomialNB", "NearestCentroid", "NuSVC", "PassiveAggressiveClassifier", "Perceptron", "QuadraticDiscriminantAnalysis", "RadiusNeighborsClassifier", "RandomForestClassifier", "RidgeClassifier", "RidgeClassifierCV", "SGDClassifier", "SVC"]
        frame.drop(columns=['task'], axis=1, inplace=True)
        preprocessor = sota_preprocessor()
        pre_frame = preprocessor.fit_transform(frame)
        import torch.utils.data as data_utils

        train = torch.tensor(pre_frame.values.astype(
            'float32'))
        with torch.no_grad():
            model_values = meta_model(train)[0]
        model_list = [float(i) for i in model_values]
        best = model_list.index(max(model_list))
        model_name = Models[best]
        str_to_model = {model.__name__: model for model in __ALL_MODELS__}
        best_model = str_to_model[model_name]

        self.model = best_model()
        # except Exception:
        #     raise Exception('Model optimization failed')
        model_guessing_time = time.time()

        model_name = self.model.__class__.__name__
        if verbose:
            self.logger.info(f'''Model guessed: {
                model_name}, searching for better model''')
        try:
            # search = ExhaustiveSearch()  # TODO split search engine into guessing and selection
            # self.model = search.predict(data, target_data)
            # if verbose:
            #     self.logger.info(f'Looked at {search.counter} models')
            pass
        except Exception:
            self.logger.info('Trying to find better model failed')
        model_selection_time = time.time()

        try:
            tuner = HyperTuner()
            hyperparameters = tuner.optimize_hyperparameters(
                self.model.__class__, (data, target_data), dataset_params.task)
            self.model = self.model.__class__(**hyperparameters)
        except Exception:
            self.logger.info('Hyperparameter optimization failed')
        hpo_time = time.time()

        model_name = self.model.__class__.__name__
        if verbose:
            self.logger.info(f"Training {model_name} model")
        try:
            self.model.train(data, target_data)
        except Exception:
            raise Exception('Model training failed')
        end_time = time.time()
        self.times = {
            'total': end_time - start_time,
            'data_ops': data_operations - start_time,
            'guessing': model_guessing_time - data_operations,
            'AS': model_selection_time - model_guessing_time,
            'HPO': hpo_time - model_selection_time,
            'training': end_time - hpo_time
        }
        if verbose:
            self.logger.info(f"{model_name} model trained successfully")
            self.logger.info(f"Data operations time: {self.times['data_ops']}")
            self.logger.info(f"Model guessing time: {self.times['guessing']}")
            self.logger.info(f"Model selection time: {self.times['AS']}")
            self.logger.info(f"Model HPO time: {self.times['HPO']}")
            self.logger.info(f"Model training time: {self.times['training']}")
            self.logger.info(f"Total time: {self.times['total']}")

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
        self.preprocessor.transform(X)
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
