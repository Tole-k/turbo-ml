from ..base import Model
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader


class NeuralNetwork(Model):
    input_formats = {torch.Tensor}
    output_formats = {torch.Tensor}

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

    def addOptmiizer(self, optimizer: str) -> optim.Optimizer:
        match optimizer:
            case 'adam':
                return optim.Adam(self.model.parameters())
            case 'sgd':
                return optim.SGD(self.model.parameters())
            case _:
                raise ValueError(f'Optimizer {optimizer} not supported')

    def __init__(self, input_size: int, output_size: int, hidden_sizes: list[int] = [1], activations: list[str] = ['relu'], loss: str = 'cross-entropy', optimizer: str = 'adam', batch_size: int = 64, epochs: int = 1000) -> None:
        super().__init__()
        layers = []
        if len(activations) != len(hidden_sizes)+1:
            raise ValueError(
                'Number of activations must be equal to the number of hidden layers + 1')
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
            layers.append(self.addActivation(activations[-1]))
        self.model = nn.Sequential(*layers)
        self.criterion = self.addLoss(loss)
        self.optimizer = self.addOptmiizer(optimizer)
        self.batch_size = batch_size
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

    def df_to_tensor(self, df):
        return torch.from_numpy(df.values).float().to(self.device)

    def train(self, data: torch.Tensor, target: torch.Tensor) -> None:
        train_dataset = self.df_to_tensor(data)
        test_dataset = self.df_to_tensor(target)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=self.batch_size, shuffle=False)
        for epoch in range(10):
            for i, (data, target) in enumerate(train_loader):
                output = self.model(data)
                loss = self.criterion(output, target)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                if epoch % 10 == 0:
                    print('[%d, %5d] loss: %.3f' %
                          (epoch + 1, i + 1, loss / 100))
        with torch.no_grad():
            correct = 0
            total = 0
            for data, target in test_loader:
                output = self.model(data)
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
            acc = 100.0 * correct / total
            print('Accuracy on test set:', acc)

    def predict(self, guess: torch.Tensor) -> torch.Tensor:
        return self.model(guess)
