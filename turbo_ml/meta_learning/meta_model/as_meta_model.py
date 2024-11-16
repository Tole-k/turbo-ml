import pickle
import matplotlib.pyplot as plt
import torch.utils.data as data_utils
import pandas as pd
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from turbo_ml.meta_learning.dataset_parameters.dataset_characteristics import StatisticalParametersExtractor
from turbo_ml.algorithms.neural_network import NeuralNetworkModel
from turbo_ml.preprocessing import sota_preprocessor
from turbo_ml.utils import options


class Best_Model(nn.Module):
    def __init__(self, num_features: int, num_classes: int):
        super(Best_Model, self).__init__()
        self.fc1 = nn.Linear(num_features, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc25 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        x = torch.relu(self.fc25(x))
        x = torch.sigmoid(self.fc3(x))
        x = self.fc4(x)
        return x


def train_meta_model(device='cpu', save_model=False, save_path='model.pth', frame=None):
    if frame is None:
        frame = pd.read_csv('results.csv')
    # change parameters
    PARAMETERS = ["name", "task", "task_detailed", "target_features", "target_nans", "num_columns", "num_rows", "number_of_highly_correlated_features", "highest_correlation",
                  "number_of_lowly_correlated_features", "lowest_correlation", "highest_eigenvalue", "lowest_eigenvalue", "share_of_numerical_features", "num_classes", "biggest_class_freq", "smallest_class_freq"]
    Families = ["Bagging_(BAG)", "Bayesian_Methods_(BY)", "Boosting_(BST)", "Decision_Trees_(DT)", "Discriminant_Analysis_(DA)", "Generalized_Linear_Models_(GLM)", "Logistic_and_Multinomial_Regression_(LMR)", "Multivariate_Adaptive_Regression_Splines_(MARS)",
                "Nearest_Neighbor_Methods_(NN)", "Neural_Networks_(NNET)", "Other_Ensembles_(OEN)", "Other_Methods_(OM)", "Partial_Least_Squares_and_Principal_Component_Regression_(PLSR)", "Random_Forests_(RF)", "Rule-Based_Methods_(RL)", "Stacking_(STC)", "Support_Vector_Machines_(SVM)"]
    if 'task' in frame.columns:
        frame.drop(columns=['task'], axis=1, inplace=True)
    if 'name' in frame.columns:
        frame.drop(columns=['name'], axis=1, inplace=True)

    target = frame[Families]
    frame.drop(Families, axis=1, inplace=True)
    preprocessor = sota_preprocessor()
    pre_frame = preprocessor.fit_transform(frame)
    if save_path:
        with open(save_path + '/preprocessor.pkl', 'wb') as f:
            pickle.dump(preprocessor, f)
    # dataset = pd.concat([pre_frame, target], axis=1)
    # print(dataset)

    values = []
    model = Best_Model(len(pre_frame.columns),
                       len(target.columns)).to(device)

    x_train, x_test, y_train, y_test = train_test_split(
        pre_frame, target, test_size=0.2)

    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    criterion = nn.MSELoss()

    train = data_utils.TensorDataset(torch.tensor(x_train.values.astype(
        'float32')).to(device), torch.tensor(y_train.values.astype
                                             ('float32')).to(device))
    test = data_utils.TensorDataset(torch.tensor(x_test.values.astype(
        'float32')).to(device), torch.tensor(y_test.values.astype
                                             ('float32')).to(device))

    train_loader = DataLoader(train, batch_size=32)
    test_loader = DataLoader(test, batch_size=32)

    epochs = 7000

    for epoch in range(epochs):
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
                    print(f'Epoch: {epoch}, Loss: {loss}')
                values.append(float(loss))
    if save_model:
        torch.save(model.state_dict(), save_path + '/model.pth')
    return model, preprocessor


if __name__ == '__main__':
    train_meta_model(device=options.device, save_model=True,
                     save_path='turbo_ml/meta_learning/meta_model/model')
