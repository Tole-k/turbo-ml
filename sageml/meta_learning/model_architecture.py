import torch.nn as nn


# class ModelArchitecture(nn.Module):
#     def __init__(self, in_features: int, out_features: int):
#         super(ModelArchitecture, self).__init__()
#         self.fc1 = nn.Linear(in_features, 256)
#         self.fc2 = nn.Linear(256, 128)
#         self.fc3 = nn.Linear(128, 64)
#         self.fc4 = nn.Linear(64, out_features)

#     def forward(self, x):
#         x = torch.relu(self.fc1(x))
#         x = torch.relu(self.fc2(x))
#         x = torch.relu(self.fc3(x))
#         x = self.fc4(x)
#         return x

class ModelArchitecture(nn.Sequential):
    def __init__(self, in_features: int, out_features: int):
        super(ModelArchitecture, self).__init__()
        self.append(nn.Linear(in_features, 1024))
        self.append(nn.BatchNorm1d(1024))
        self.append(nn.ReLU())
        self.append(nn.Dropout1d(0.2))
        self.append(nn.Linear(1024, 512))
        self.append(nn.BatchNorm1d(512))
        self.append(nn.ReLU())
        self.append(nn.Dropout1d(0.2))
        self.append(nn.Linear(512, 256))
        self.append(nn.BatchNorm1d(256))
        self.append(nn.ReLU())
        self.append(nn.Dropout1d(0.2))
        self.append(nn.Linear(256, 64))
        self.append(nn.BatchNorm1d(64))
        self.append(nn.ReLU())
        self.append(nn.Dropout1d(0.2))
        self.append(nn.Linear(64, 32))
        self.append(nn.BatchNorm1d(32))
        self.append(nn.ReLU())
        self.append(nn.Dropout1d(0.2))
        self.append(nn.Linear(32, out_features))
        self.append(nn.ReLU())

    def forward(self, x):
        for layer in self:
            x = layer(x)
        return x
