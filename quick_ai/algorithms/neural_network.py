from datasets import *
import pandas as pd
from ..base import Model
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from typing import List, Tuple
import logging
from ..utils import option

logging.basicConfig(level=option.log_level)


class NeuralNetwork(Model):
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


class NeuralNetworkClassifier(NeuralNetwork):
    task = 'classification'

    def train(self, data: pd.DataFrame, target: pd.DataFrame | pd.Series) -> None:
        train_loader, test_loader = self.setup_loaders(data, target, self.task)
        logger = logging.getLogger(self.__class__.__name__)
        for epoch in range(self.epochs):
            for i, (data, target) in enumerate(train_loader):
                self.model.train()
                output = self.model(data)
                loss = self.criterion(output, target)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                if epoch % 10 == 0:
                    logger.info('[%d, %5d] loss: %.3f' %
                                (epoch + 1, i + 1, loss / 100))
        self.model.eval()
        with torch.inference_mode():
            correct = 0
            total = 0
            for data, target in test_loader:
                output = self.model(data)
                predicted = torch.argmax(output.data, dim=1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
            acc = 100.0 * correct / total
            logger.info('Accuracy on test set:', acc)

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


class NeuralNetworkRegressor(NeuralNetwork):
    task = 'regression'

    def train(self, data: pd.DataFrame, target: pd.DataFrame | pd.Series) -> None:
        train_loader, test_loader = self.setup_loaders(data, target, self.task)
        logger = logging.getLogger(self.__class__.__name__)
        for epoch in range(self.epochs):
            for i, (data, target) in enumerate(train_loader):
                if target.ndimension() == 1:
                    target = target.view(len(target), 1)
                self.model.train()
                output = self.model(data)
                loss = self.criterion(output, target)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                if epoch % 10 == 0:
                    logger.info('[%d, %5d] loss: %.3f' %
                                (epoch + 1, i + 1, loss / 100))
        self.model.eval()
        with torch.inference_mode():
            total_loss = 0
            for data, target in test_loader:
                if target.ndimension() == 1:
                    target = target.view(len(target), 1)
                output = self.model(data)
                loss = self.criterion(output, target)
                total_loss += loss
            total_loss /= len(test_loader)
            logger.info('Error on test set:', total_loss.item())

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

    def create_neural_network(self, input_size: int, output_size: int, hidden_sizes: List[int], task: str = 'classification', activations: List[str] = ['relu'], loss: str = 'crossentropyloss', optimizer: str = 'adam', batch_size: int = 64, epochs: int = 1000, learning_rate=0.001) -> NeuralNetwork:
        layers = []
        if len(activations) != len(hidden_sizes):
            raise ValueError(
                'Number of activations must be equal to the number of hidden layers')
        if len(hidden_sizes) == 0:
            layers.append(nn.Linear(input_size, output_size))
            layers.append(self._add_activation(activations[0]))
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
            *layers).cuda() if torch.cuda.is_available() else nn.Sequential(*layers)
        self.criterion = self._add_loss(loss)
        self.optimizer = self._add_optimizer(optimizer, learning_rate)
        self.batch_size = batch_size
        self.epochs = epochs
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        if task == 'classification':
            return NeuralNetworkClassifier(self.model, self.criterion, self.optimizer, self.device, self.batch_size, self.epochs)
        elif task == 'regression':
            return NeuralNetworkRegressor(self.model, self.criterion, self.optimizer, self.device, self.batch_size, self.epochs)
        else:
            raise ValueError('Invalid task type')


print('Iris')
data, target = get_iris()
model = NNFactory().create_neural_network(4, 3, [128, 64], 'classification', ['relu', 'relu'],
                                          'crossentropyloss', 'adam', 32, 100)
model.train(data, target)
print(model.predict(data))
print('Wine')
data, target = get_wine()
model = NNFactory().create_neural_network(13, 3, [128, 64], 'classification', ['relu', 'relu'],
                                          'crossentropyloss', 'adam', 32, 100)
model.train(data, target)
print('Breast Cancer')
data, target = get_breast_cancer()
model = NNFactory().create_neural_network(30, 2, [128, 64], 'classification', ['relu', 'relu'],
                                          'crossentropyloss', 'adam', 32, 100)
model.train(data, target)
print(model.predict(data))
print('Diabetes')
data, target = get_diabetes()
model = NNFactory().create_neural_network(10, 1, [128, 64], 'regression', [
    'relu', 'relu'], 'mseloss', 'adam', 32, 100)
model.train(data, target)
print(model.predict(data))
print('Linnerud')
data, target = get_linnerud()
model = NNFactory().create_neural_network(3, 3, [128, 64], 'regression', [
    'relu', 'relu'], 'mseloss', 'adam', 32, 100)
model.train(data, target)
print(model.predict(data))
