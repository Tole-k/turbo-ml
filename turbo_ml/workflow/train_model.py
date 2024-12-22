import os
import pickle
from typing import Any, Tuple
from prefect import flow, task
from turbo_ml.meta_learning.model_architecture import ModelArchitecture
from turbo_ml.preprocessing import sota_preprocessor
from turbo_ml.utils import options
import torch
import torch.nn as nn
from torch.utils import data as data_utils
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split


@flow(name='Train Meta Model')
def train_meta_model(feature_frame: pd.DataFrame | str | None = None, evaluations_frame: pd.DataFrame | str | None = None,
                     epochs: int = 7000) -> Tuple[ModelArchitecture, Any]:
    if feature_frame is None:
        feature_frame = 'parameters.csv'
    if isinstance(feature_frame, str):
        feature_frame = pd.read_csv(feature_frame)

    if evaluations_frame is None:
        evaluations_frame = os.path.join('datasets', 'results_algorithms.csv')

    if isinstance(evaluations_frame, str):
        evaluations_frame = pd.read_csv(evaluations_frame)

    common_names = set(feature_frame['name']).intersection(
        set(evaluations_frame['name']))
    print(f'Number of matching datasets: {len(common_names)}')
    feature_frame = feature_frame[feature_frame['name'].isin(
        common_names)].sort_values('name').reset_index(drop=True)
    evaluations_frame = evaluations_frame[evaluations_frame['name'].isin(
        common_names)].sort_values('name').reset_index(drop=True)

    feature_frame.drop(columns=['name'], axis=1, inplace=True)
    evaluations_frame.drop(columns=['name'], axis=1, inplace=True)
    preprocessor = sota_preprocessor()
    feature_frame = preprocessor.fit_transform(feature_frame)
    preprocessor2 = sota_preprocessor()
    evaluations_frame = preprocessor2.fit_transform(evaluations_frame)

    values = []
    model = ModelArchitecture(len(feature_frame.columns),
                       len(evaluations_frame.columns)).to(options.device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    criterion = nn.MSELoss()


    x_train, x_test, y_train, y_test = train_test_split(
        feature_frame, evaluations_frame, test_size=0.2)
    train = data_utils.TensorDataset(torch.tensor(x_train.values.astype(
        'float32')).to(options.device), torch.tensor(y_train.values.astype
                                                     ('float32')).to(options.device))
    test = data_utils.TensorDataset(torch.tensor(x_test.values.astype(
        'float32')).to(options.device), torch.tensor(y_test.values.astype
                                                     ('float32')).to(options.device))

    train_loader = data_utils.DataLoader(train, batch_size=32)
    test_loader = data_utils.DataLoader(test, batch_size=32)

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
    return model, preprocessor


@task(name="Save Meta Model")
def save_meta_model(model: ModelArchitecture, preprocessor: Any, save_path: str):
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    torch.save(model.state_dict(), save_path + '/model.pth')
    with open(save_path + '/model_params.pkl', 'wb') as f:
        pickle.dump({'input_size': model.fc1.in_features,
                     'output_size': model.fc4.out_features}, f)
    with open(save_path + '/preprocessor.pkl', 'wb') as f:
        pickle.dump(preprocessor, f)


if __name__ == '__main__':
    train_meta_model(save_model=True,
                     save_path='turbo_ml/meta_learning/meta_model/model')
