"""
quickai.py

This module provides the `QuickAI` class, our main class for out-of-the-box autoML solution.
It does not provide additional functionalities but it combines other modules to provide a complete solution.
"""
from typing import Any, Optional
import pandas as pd
from .base import Model
from .forecast import StatisticalParametersExtractor


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

    def __init__(self, dataset: pd.DataFrame, target: Optional[str] = None):
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
        if target is None:
            # target = find_target()
            raise NotImplementedError(
                "Target automatic detection is not implemented yet, provide target column name")
        self._input_check(dataset, target)
        self.model: Model = None
        target_data = dataset[target]
        data = dataset.drop(columns=[target])
        try:
            extractor = StatisticalParametersExtractor(data, target_data)
            dataset_params = extractor.describe_dataset()
        except Exception:
            raise Exception("Dataset description failed")

        try:
            # = forecast.guess_best_model(dataset_params)
            self.model = None
        except Exception:
            raise Exception("Model optimization failed")

        try:
            # = forecast.find_best_model(dataset, target, dataset_params)
            self.model = self.model
        except Exception:
            print("Trying to find better model failed")

        try:
            self.model.train(data, target_data)
        except Exception:
            raise Exception("Model training failed")

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
        return self.model.predict(X)

    def __call__(self, X: pd.DataFrame) -> pd.Series:
        """
        Generates predictions using the trained model. Call methd is just wrapper for predict method.

        Args:
            X (pd.DataFrame): A DataFrame containing the input features for prediction.

        Returns:
            pd.Series: A Series containing the predicted values.
        """
        return self.predict(X)
