""" Main training model loop """
from typing import Any
import torch
import torch.nn as nn
from torch.utils import data as data_utils
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sageml.meta_learning.model_architecture import ModelArchitecture
from sageml.preprocessing import sota_preprocessor
from sageml.utils import options


def train_meta_model(score_dataframe: pd.DataFrame, param_dataframe: pd.DataFrame,
                     epochs: int = 7000, batch_size: int = 32) -> tuple[ModelArchitecture, Any, dict]:
    """Train meta model.

    Args:
        score_dataframe (pd.DataFrame): Dataframe with scores.
        param_dataframe (pd.DataFrame): Dataframe with parameters.
        epochs (int, optional): Num of epochs. Defaults to 7000.

    Returns:
        tuple[ModelArchitecture, Any]: Model and preprocessing.
    """
    common_names = set(param_dataframe['name']) & set(score_dataframe['name'])
    param_dataframe = param_dataframe[param_dataframe['name'].isin(common_names)].sort_values('name').reset_index(drop=True)
    score_dataframe = score_dataframe[score_dataframe['name'].isin(common_names)].sort_values('name').reset_index(drop=True)

    param_dataframe.drop(columns=['name'], axis=1, inplace=True)
    score_dataframe.drop(columns=['name'], axis=1, inplace=True)
    preprocessor = sota_preprocessor()
    param_dataframe = preprocessor.fit_transform(param_dataframe)
    preprocessor2 = sota_preprocessor()
    score_dataframe = preprocessor2.fit_transform(score_dataframe)

    values = []
    model = ModelArchitecture(len(param_dataframe.columns),
                              len(score_dataframe.columns)).to(options.device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    criterion = nn.MSELoss()

    x_train, x_test, y_train, y_test = train_test_split(
        param_dataframe, score_dataframe, test_size=0.2)
    train = data_utils.TensorDataset(torch.tensor(x_train.values.astype(
        'float32')).to(options.device), torch.tensor(y_train.values.astype
                                                     ('float32')).to(options.device))
    test = data_utils.TensorDataset(torch.tensor(x_test.values.astype(
        'float32')).to(options.device), torch.tensor(y_test.values.astype
                                                     ('float32')).to(options.device))

    train_loader = data_utils.DataLoader(train, batch_size=batch_size)
    test_loader = data_utils.DataLoader(test, batch_size=batch_size)

    loss = float('inf')
    pbar = tqdm(range(epochs), total=epochs,
                desc='Training model, loss: ...', unit='epoch')
    for epoch in pbar:
        model.train()
        for x, y in train_loader:
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.inference_mode():
            for x, y in test_loader:
                output = model(x)
                loss = criterion(output, y)
                if epoch % 100 == 0:
                    pbar.set_description(f'Training model, loss: {loss:.2f}')
                values.append(float(loss))
    return model, preprocessor, {'input_size': len(param_dataframe.columns),
                                 'output_size': len(score_dataframe.columns)}
