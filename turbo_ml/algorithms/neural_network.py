import numpy as np
from typing import Any, Dict
from typing import Literal
import pandas as pd
from ..base import Model
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from typing import List, Tuple
import logging
from ..utils import options
import optuna as opt
logging.basicConfig(
    level=options.dev_mode_logging if options.dev_mode else options.user_mode_logging)


class NeuralNetworkBase(Model):
    input_formats = {pd.DataFrame}
    output_formats = {pd.DataFrame | pd.Series}

    def __init__(self, model: nn.Module, loss, optimizer, device, batch_size, epochs) -> None:
        super().__init__()
        self.model = model
        self.criterion = loss
        self.optimizer = optimizer
        self.device = device
        self.batch_size = batch_size
        self.epochs = epochs

    class DataTargetDataset(torch.utils.data.Dataset):
        def __init__(self, data, target, device, task):
            self.device = device
            self.data = torch.tensor(data.values).float().to(self.device)
            self.target = torch.tensor(target.values).long().to(self.device) if task == 'classification' else torch.tensor(
                target.values).float().to(self.device) if task == 'regression' else None

        def __getitem__(self, index):
            return self.data[index], self.target[index]

        def __len__(self):
            return len(self.data)

    def setup_loaders(self, data: pd.DataFrame, target: pd.DataFrame | pd.Series, task: str) -> Tuple[DataLoader, DataLoader]:
        train_data, test_data, train_target, test_target = train_test_split(
            data, target, test_size=0.2)

        train_dataset = self.DataTargetDataset(
            train_data, train_target, self.device, self.task)
        test_dataset = self.DataTargetDataset(
            test_data, test_target, self.device, self.task)

        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size)
        test_loader = DataLoader(
            test_dataset, batch_size=self.batch_size)
        return train_loader, test_loader


class NeuralNetworkClassifier(NeuralNetworkBase):
    task = 'classification'

    def train(self, data: pd.DataFrame, target: pd.DataFrame | pd.Series) -> None:
        train_loader, test_loader = self.setup_loaders(data, target, self.task)
        logger = logging.getLogger(self.__class__.__name__)
        logger.setLevel(
            options.dev_mode_logging if options.dev_mode else options.user_mode_logging)
        for epoch in range(self.epochs):
            self.model.train()
            for data, target in train_loader:
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()
            self.model.eval()
            with torch.inference_mode():
                for i, (data, target) in enumerate(test_loader):
                    output = self.model(data)
                    loss = self.criterion(output, target)
                    if epoch % 100 == 0:
                        logger.info(' [%d, %5d] loss: %.3f' %
                                    (epoch + 1, i + 1, loss))

    def predict(self, guess: pd.DataFrame) -> pd.DataFrame | pd.Series:
        result = None
        self.model.eval()
        with torch.inference_mode():
            result = torch.argmax(self.model(torch.tensor(
                guess.values).float().to(self.device)), dim=1)
        if result.ndim == 1:
            return pd.Series(result.cpu().numpy())
        else:
            return pd.DataFrame(result.cpu().numpy())


class NeuralNetworkRegressor(NeuralNetworkBase):
    task = 'regression'

    def train(self, data: pd.DataFrame, target: pd.DataFrame | pd.Series) -> None:
        train_loader, test_loader = self.setup_loaders(data, target, self.task)
        logger = logging.getLogger(self.__class__.__name__)
        logger.setLevel(
            options.dev_mode_logging if options.dev_mode else options.user_mode_logging)
        for epoch in range(self.epochs):
            self.model.train()
            for data, target in train_loader:
                self.optimizer.zero_grad()
                if target.ndimension() == 1:
                    target = target.view(len(target), 1)
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()
            self.model.eval()
            with torch.inference_mode():
                for i, (data, target) in enumerate(test_loader):
                    output = self.model(data)
                    loss = self.criterion(output, target)
                    if epoch % 100 == 0:
                        logger.info(' [%d, %5d] loss: %.3f' %
                                    (epoch + 1, i + 1, loss))

    def predict(self, guess: pd.DataFrame) -> pd.DataFrame | pd.Series:
        result = None
        self.model.eval()
        with torch.inference_mode():
            result = self.model(torch.tensor(
                guess.values).float().to(self.device))
        if result.ndim == 1:
            return pd.Series(result.cpu().numpy())
        else:
            return pd.DataFrame(result.cpu().numpy())


class NNFactory:
    activations = [nn.ReLU, nn.Sigmoid, nn.Tanh, nn.Softmax, nn.LeakyReLU, nn.ELU, nn.SELU, nn.GELU, nn.Hardtanh, nn.LogSigmoid, nn.Softplus,
                   nn.Softshrink, nn.Softsign, nn.Tanhshrink, nn.RReLU, nn.CELU, nn.GLU, nn.SiLU, nn.Mish, nn.ReLU6, nn.PReLU, nn.Hardsigmoid, nn.Hardshrink]
    loss_functions = [nn.CrossEntropyLoss, nn.MSELoss, nn.L1Loss, nn.NLLLoss, nn.PoissonNLLLoss, nn.KLDivLoss, nn.BCELoss, nn.BCEWithLogitsLoss, nn.MarginRankingLoss,
                      nn.HingeEmbeddingLoss, nn.MultiMarginLoss, nn.SmoothL1Loss, nn.HuberLoss, nn.CosineEmbeddingLoss, nn.MultiLabelSoftMarginLoss, nn.TripletMarginLoss, nn.CTCLoss]
    optimizers = [optim.Adam, optim.SGD, optim.Adadelta, optim.Adagrad,
                  optim.Adamax, optim.RMSprop, optim.Rprop, optim.LBFGS]

    def _add_activation(self, activation: str) -> nn.Module:
        return {activation.__name__.lower(): activation() for activation in self.activations}[activation.lower()]

    def _add_loss(self, loss: str) -> nn.Module:
        return {loss.__name__.lower(): loss() for loss in self.loss_functions}[loss.lower()]

    def _add_optimizer(self, optimizer: str, learning_rate) -> optim.Optimizer:
        return {optimizer.__name__.lower(): optimizer(self.model.parameters(), lr=learning_rate)
                for optimizer in self.optimizers}[optimizer.lower()]

    @staticmethod
    def optimize_hyperparameters(dataset: Tuple[pd.DataFrame, pd.DataFrame], task: Literal['classification', 'regression'] = 'classification', no_classes: int = None, no_variables: int = None, device='cpu', trials: int = 100) -> Dict[str, Any]:
        def objective(trial, dataset, task, device):
            x_train, x_test, y_train, y_test = train_test_split(
                *dataset, test_size=0.2)
            trial.set_user_attr('input_size', x_train.shape[1])

            trial.set_user_attr('task', task)
            trial.set_user_attr('device', device)
            trial.set_user_attr('epochs', 1000)
            params = {}
            params['input_size'] = trial.user_attrs['input_size']
            if task == 'classification':
                params['output_size'] = no_classes
            else:
                params['output_size'] = no_variables
            trial.set_user_attr('output_size', params['output_size'])
            num_hidden_layers = trial.suggest_int(
                'num_hidden_layers', 1, 8)
            hidden_sizes = []
            activations = []
            for i in range(num_hidden_layers):
                hidden_sizes.append(trial.suggest_categorical(
                    f'hidden_size_{i}', [2**i for i in range(2, 11)]))
                activations.append(trial.suggest_categorical(
                    f'activation_{i}', ['relu', 'leakyrelu', 'elu', 'selu', 'gelu', 'rrelu', 'celu', 'silu', 'mish', 'relu6', 'prelu']))
            trial.set_user_attr('hidden_sizes', hidden_sizes)
            trial.set_user_attr('activations', activations)
            params['hidden_sizes'] = trial.user_attrs['hidden_sizes']
            params['activations'] = trial.user_attrs['activations']
            params['task'] = trial.user_attrs['task']
            if task == 'classification':  # Loss function optimization removed because it introduced to many incompatible combinations with activation functions and overall nn architecture
                params['loss'] = 'crossentropyloss'
            else:
                params['loss'] = 'mseloss'
            trial.set_user_attr('loss', params['loss'])
            params['optimizer'] = trial.suggest_categorical(
                'optimizer', ['adam', 'sgd', 'adadelta', 'adagrad', 'adamax', 'rmsprop', 'rprop'])
            params['batch_size'] = trial.suggest_int(
                'batch_size', np.log2(len(x_train)), np.sqrt(len(x_train)))
            params['epochs'] = trial.user_attrs['epochs']
            params['learning_rate'] = trial.suggest_float(
                'learning_rate', 0.0001, 0.01)
            params['device'] = trial.user_attrs['device']
            model = NNFactory().create_neural_network(**params)
            model.train(x_train, y_train)
            if task == 'classification':
                return sum(model.predict(x_test) == y_test.reset_index(drop=True))/len(y_test)
            else:
                results = model.predict(x_test)
                if not isinstance(y_test, pd.Series):
                    results.columns = y_test.columns
                    results.index = y_test.index
                    return np.sum((results-y_test).values**2)/len(y_test)
                return sum((results-y_test)**2)/len(y_test)

        study = opt.create_study(
            direction='maximize' if task == 'classification' else 'minimize', study_name='Neural Network Hyperparameter Optimization')
        study.optimize(lambda trial: objective(
            trial, dataset, task, device), n_trials=trials)
        return study.best_params | study.best_trial.user_attrs

    def create_neural_network(self, input_size: int, output_size: int, hidden_sizes: List[int], task: str = 'classification', activations: List[str] = ['relu'], loss: str = 'crossentropyloss', optimizer: str = 'adam', batch_size: int = 64, epochs: int = 1000, learning_rate=0.001, device='cpu') -> NeuralNetworkBase:
        layers = []
        if len(activations) != len(hidden_sizes):
            raise ValueError(
                'Number of activations must be equal to the number of hidden layers')
        if len(hidden_sizes) == 0:
            layers.append(nn.Linear(input_size, output_size))
        else:
            for i, (hidden_size, activation) in enumerate(zip(hidden_sizes, activations)):
                if i == 0:
                    layers.append(nn.Linear(input_size, hidden_size))
                else:
                    layers.append(nn.Linear(hidden_sizes[i-1], hidden_size))
                layers.append(self._add_activation(activation))
            layers.append(nn.Linear(hidden_sizes[-1], output_size))
        self.task = task
        self.model = nn.Sequential(
            *layers).to(device)
        self.criterion = self._add_loss(loss)
        self.optimizer = self._add_optimizer(optimizer, learning_rate)
        self.batch_size = batch_size
        self.epochs = epochs
        self.device = device
        if task == 'classification':
            return NeuralNetworkClassifier(self.model, self.criterion, self.optimizer, self.device, self.batch_size, self.epochs)
        elif task == 'regression':
            return NeuralNetworkRegressor(self.model, self.criterion, self.optimizer, self.device, self.batch_size, self.epochs)
        else:
            raise ValueError('Invalid task type')


# For compatibility with general Model class for the sake of hyperparameter tuning


class NeuralNetworkModel(Model):
    input_formats = {pd.DataFrame}
    output_formats = {pd.DataFrame | pd.Series}
    factory = NNFactory()

    @staticmethod
    def optimize_hyperparameters(dataset: Tuple[pd.DataFrame, pd.DataFrame], task: Literal['classification', 'regression'] = 'classification', no_classes: int = None, no_variables: int = None, device='cpu', trials=100) -> Dict[str, Any]:
        params = NNFactory.optimize_hyperparameters(
            dataset, task, no_classes, no_variables, device, trials)
        hidden_sizes = []
        activations = []
        for i in range(params['num_hidden_layers']):
            hidden_sizes.append(params[f'hidden_size_{i}'])
            activations.append(params[f'activation_{i}'])
            del params[f'hidden_size_{i}']
            del params[f'activation_{i}']
        del params['num_hidden_layers']
        params['hidden_sizes'] = hidden_sizes
        params['activations'] = activations
        return params

    def __init__(self, input_size: int, output_size: int, hidden_sizes: List[int], task: str = 'classification', activations: List[str] = ['relu'], loss: str = 'crossentropyloss', optimizer: str = 'adam', batch_size: int = 64, epochs: int = 1000, learning_rate=0.001, device='cpu') -> None:
        super().__init__()
        self.model = self.factory.create_neural_network(
            input_size, output_size, hidden_sizes, task, activations, loss, optimizer, batch_size, epochs, learning_rate, device)

    def train(self, data: pd.DataFrame, target: pd.DataFrame | pd.Series) -> None:
        self.model.train(data, target)

    def predict(self, guess: pd.DataFrame) -> pd.DataFrame | pd.Series:
        return self.model.predict(guess)


def __main__imports__():
    from datasets import get_iris
    return get_iris


if __name__ == '__main__':
    get_iris = __main__imports__()
    params = NeuralNetworkModel.optimize_hyperparameters(
        dataset=get_iris(), task='classification', no_classes=3, no_variables=1, device=options.device)
    print(params)
    model = NeuralNetworkModel(**params)
    model.train(*get_iris())
    print(model.predict(get_iris()[0]))
