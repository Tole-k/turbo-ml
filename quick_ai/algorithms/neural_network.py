import pandas as pd
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
            case 'leaky_relu':
                return nn.LeakyReLU()
            case 'elu':
                return nn.ELU()
            case 'selu':
                return nn.SELU()
            case 'gelu':
                return nn.GELU()
            case 'threshold':
                return nn.Threshold()
            case 'hardtanh':
                return nn.Hardtanh()
            case 'log_sigmoid':
                return nn.LogSigmoid()
            case 'softplus':
                return nn.Softplus()
            case 'softshrink':
                return nn.Softshrink()
            case 'softsign':
                return nn.Softsign()
            case 'tanhshrink':
                return nn.Tanhshrink()
            case 'rrelu':
                return nn.RReLU()
            case 'celu':
                return nn.CELU()
            case 'glu':
                return nn.GLU()
            case 'silu':
                return nn.SiLU()
            case 'mish':
                return nn.Mish()
            case 'relu6':
                return nn.ReLU6()
            case 'prelu':
                return nn.PReLU()
            case 'hardsigmoid':
                return nn.Hardsigmoid()
            case 'hardshrink':
                return nn.Hardshrink()
            case _:
                raise ValueError(f'Activation {activation} not supported')

    def addLoss(self, loss: str) -> nn.Module:
        match loss:
            case 'cross-entropy':
                return nn.CrossEntropyLoss()
            case 'mse':
                return nn.MSELoss()
            case 'l1':
                return nn.L1Loss()
            case 'nll':
                return nn.NLLLoss()
            case 'poisson':
                return nn.PoissonNLLLoss()
            case 'kld':
                return nn.KLDivLoss()
            case 'bce':
                return nn.BCELoss()
            case 'bce_with_logits':
                return nn.BCEWithLogitsLoss()
            case 'margin_ranking':
                return nn.MarginRankingLoss()
            case 'hinge':
                return nn.HingeEmbeddingLoss()
            case 'multi_margin':
                return nn.MultiMarginLoss()
            case 'smooth_l1':
                return nn.SmoothL1Loss()
            case 'huber':
                return nn.HuberLoss()
            case 'cosine':
                return nn.CosineEmbeddingLoss()
            case 'multi_label_soft_margin':
                return nn.MultiLabelSoftMarginLoss()
            case 'triplet_margin':
                return nn.TripletMarginLoss()
            case 'ctc':
                return nn.CTCLoss()
            case 'nll_loss2d':
                return nn.NLLLoss2d()
            case 'poisson_nll':
                return nn.PoissonNLLLoss()
            case _:
                raise ValueError(f'Loss {loss} not supported')

    def addOptimizer(self, optimizer: str, learning_rate) -> optim.Optimizer:
        match optimizer:
            case 'adam':
                return optim.Adam(self.model.parameters(), lr=learning_rate)
            case 'sgd':
                return optim.SGD(self.model.parameters(), lr=learning_rate)
            case 'adadelta':
                return optim.Adadelta(self.model.parameters(), lr=learning_rate)
            case 'adagrad':
                return optim.Adagrad(self.model.parameters(), lr=learning_rate)
            case 'adamax':
                return optim.Adamax(self.model.parameters(), lr=learning_rate)
            case 'rmsprop':
                return optim.RMSprop(self.model.parameters(), lr=learning_rate)
            case 'rprop':
                return optim.Rprop(self.model.parameters(), lr=learning_rate)
            case 'lbfgs':
                return optim.LBFGS(self.model.parameters(), lr=learning_rate)
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
                loss = self.criterion(outppip3 install torch torchvision torchaudiout, target)
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

# from datasets import *
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
