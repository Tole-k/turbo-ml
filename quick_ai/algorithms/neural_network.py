import pandas as pd
from datasets import *
from ..base import Model
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split


class NeuralNetwork(Model):
    input_formats = {pd.DataFrame}
    output_formats = {pd.DataFrame | pd.Series}

    def addActivation(self, activation: str) -> nn.Module:
        match activation:
            case 'relu':
                return nn.ReLU()
            case 'sigmoid':
                return nn.Sigmoid()
            case 'tanh':
                return nn.Tanh()
            case 'softmax':
                return nn.Softmax()
            case _:
                raise ValueError(f'Activation {activation} not supported')

    def addLoss(self, loss: str) -> nn.Module:
        match loss:
            case 'cross-entropy':
                return nn.CrossEntropyLoss()
            case 'mse':
                return nn.MSELoss()
            case _:
                raise ValueError(f'Loss {loss} not supported')

    def addOptimizer(self, optimizer: str, learning_rate) -> optim.Optimizer:
        match optimizer:
            case 'adam':
                return optim.Adam(self.model.parameters(), lr=learning_rate)
            case 'sgd':
                return optim.SGD(self.model.parameters(), lr=learning_rate)
            case _:
                raise ValueError(f'Optimizer {optimizer} not supported')

    def __init__(self, input_size: int, output_size: int, hidden_sizes: list[int], task: str = 'classification', activations: list[str] = ['relu'], loss: str = 'cross-entropy', optimizer: str = 'adam', batch_size: int = 64, epochs: int = 1000, learning_rate=0.001) -> None:
        super().__init__()
        layers = []
        if len(activations) != len(hidden_sizes):
            raise ValueError(
                'Number of activations must be equal to the number of hidden layers')
        if len(hidden_sizes) == 0:
            layers.append(nn.Linear(input_size, output_size))
            layers.append(self.addActivation(activations[0]))
        else:
            for i, (hidden_size, activation) in enumerate(zip(hidden_sizes, activations)):
                if i == 0:
                    layers.append(nn.Linear(input_size, hidden_size))
                else:
                    layers.append(nn.Linear(hidden_sizes[i-1], hidden_size))
                layers.append(self.addActivation(activation))
            layers.append(nn.Linear(hidden_sizes[-1], output_size))
        self.task = task
        self.model = nn.Sequential(*layers).cuda()
        self.criterion = self.addLoss(loss)
        self.optimizer = self.addOptimizer(optimizer, learning_rate)
        self.batch_size = batch_size
        self.epochs = epochs
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

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

    def train(self, data: pd.DataFrame, target: pd.DataFrame | pd.Series) -> None:
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
        for epoch in range(self.epochs):
            for i, (data, target) in enumerate(train_loader):
                if self.task == 'regression' and target.ndimension() == 1:
                    target = target.view(len(target), 1)
                self.model.train()
                output = self.model(data)
                loss = self.criterion(output, target)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                if epoch % 10 == 0:
                    print('[%d, %5d] loss: %.3f' %
                          (epoch + 1, i + 1, loss / 100))
        self.model.eval()
        with torch.inference_mode():
            if self.task == 'classification':
                correct = 0
                total = 0
                for data, target in test_loader:
                    output = self.model(data)
                    predicted = torch.argmax(output.data, dim=1)
                    total += target.size(0)
                    correct += (predicted == target).sum().item()
                acc = 100.0 * correct / total
                print('Accuracy on test set:', acc)
            else:
                # breakpoint()
                total_loss = 0
                for data, target in test_loader:
                    if target.ndimension() == 1:
                        target = target.view(len(target), 1)
                    output = self.model(data)
                    loss = self.criterion(output, target)
                    total_loss += loss
                total_loss /= len(test_loader)
                print('Error on test set:', total_loss.item())

    def predict(self, guess: pd.DataFrame | pd.Series) -> torch.Tensor:
        result = None
        self.model.eval()
        with torch.inference_mode():
            if self.task == 'classification':
                result = torch.argmax(self.model(torch.tensor(
                    guess.values).float().to(self.device)), dim=1)
            else:
                result = self.model(torch.tensor(
                    guess.values).float().to(self.device))
        if result.ndim == 1:
            return pd.Series(result.cpu().numpy())
        else:
            return pd.DataFrame(result.cpu().numpy())

# 
# print('Iris')
# data, target = get_iris()
# target.map({0: '0', 1: '1', 2: '2'})
# model = NeuralNetwork(4, 3, [128, 64], 'classification', ['relu', 'relu'],
#                       'cross-entropy', 'adam', 32, 100)
# model.train(data, target)
# print(model.predict(data))
#
# print('Wine')
# data, target = get_wine()
# model = NeuralNetwork(13, 3, [128, 64], 'classification', ['relu', 'relu'],
#                       'cross-entropy', 'adam', 32, 100)
# model.train(data, target)
#
# print('Breast Cancer')
# data, target = get_breast_cancer()
# model = NeuralNetwork(30, 2, [128, 64], 'classification', ['relu', 'relu'],
#                       'cross-entropy', 'adam', 32, 100)
# model.train(data, target)
# print(model.predict(data))
#
# print('Diabetes')
# data, target = get_diabetes()
# model = NeuralNetwork(10, 1, [128, 64], 'regression', [
#     'relu', 'relu'], 'mse', 'adam', 32, 100)
# model.train(data, target)
# print(model.predict(data))

#
# print('Linnerud')
# data, target = get_linnerud()
# model = NeuralNetwork(3, 3, [128, 64], 'regression', [
#     'relu', 'relu'], 'mse', 'adam', 32, 100)
# model.train(data, target)
# print(model.predict(data))
