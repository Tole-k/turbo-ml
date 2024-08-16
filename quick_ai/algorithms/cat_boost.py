# from sklearn.model_selection import train_test_split
# from datasets import get_iris
# import catboost as cb
# from ..base import Model
# from collections.abc import Iterable
# from typing import List, Literal


# class CatBoostClassifier(Model):
#     input_formats = {Iterable[int | float | bool | str]}
#     output_formats = {list[int], list[str], list[bool]}
#     hyperparameters = [
#         {
#             "name": "iterations",
#             "type": "int",
#             "min": 1,
#             "max": 1000,
#             "optional": False
#         },
#         {
#             "name": "learning_rate",
#             "type": "float",
#             "min": 0.0,
#             "max": 1.0,
#             "optional": False
#         },
#         {
#             "name": "depth",
#             "type": "int",
#             "min": 1,
#             "max": 16,
#             "optional": False
#         },
#         {
#             "name": "l2_leaf_reg",
#             "type": "float",
#             "min": 0.0,
#             "max": 10.0,
#             "optional": False
#         },
#         {
#             "name": "model_size_reg",
#             "type": "float",
#             "min": 0.0,
#             "max": 10.0,
#             "optional": True
#         },
#         {
#             "conditional": True,
#             "condition": "binary/multi",
#             "variants": [{
#                 "name": "loss_function",
#                 "type": "categorical",
#                 "choices": ['Logloss', 'CrossEntropy'],
#                 "optional": False
#             },
#                 {
#                 "name": "loss_function",
#                 "type": "categorical",
#                 "choices": ['MultiClass', 'MultiClassOneVsAll'],
#                 "optional": False
#             }]
#         },
#         {
#             "name": "feature_border_type",
#             "type": "categorical",
#             "choices": ['Median', 'Uniform', 'UniformAndQuantiles', 'GreedyLogSum', 'MaxLogSum', 'MinEntropy'],
#             "optional": False
#         },
#         {
#             "name": "min_data_in_leaf",
#             "type": "int",
#             "min": 1,
#             "max": 10,
#             "optional": False
#         },
#         {
#             "conditional": True,
#             "condition": "cpu/cuda",
#             "variants": [{
#                 "name": "device",
#                 "type": "no_choice",
#                 "choices": ['CPU'],
#                 "optional": False
#             }, {
#                 "name": "device",
#                 "type": "no_choice",
#                 "choices": ['GPU'],
#                 "optional": False
#             }]
#         }
#     ]

#     def __init__(self, iterations: int = 500,
#                  learning_rate: float = 0.03,
#                  depth: int = 6,
#                  l2_leaf_reg: float = 3.0,
#                  model_size_reg: float = None,
#                  loss_function: str = 'Logloss',
#                  feature_border_type: Literal['Median', 'Uniform', 'UniformAndQuantiles',
#                                               'GreedyLogSum', 'MaxLogSum', 'MinEntropy'] = 'GreedyLogSum',
#                  min_data_in_leaf: int = 1,
#                  device: Literal['CPU', 'GPU'] = 'CPU',
#                  ** options) -> None:
#         super().__init__()
#         self.clf = cb.CatBoostClassifier(
#             iterations=iterations,
#             learning_rate=learning_rate,
#             depth=depth,
#             l2_leaf_reg=l2_leaf_reg,
#             model_size_reg=model_size_reg,
#             loss_function=loss_function,
#             feature_border_type=feature_border_type,
#             min_data_in_leaf=min_data_in_leaf,
#             task_type=device,
#             **options
#         )

#     def train(self, data: Iterable[int | float | bool | str], target: Iterable) -> None:
#         self.clf = self.clf.fit(data, target, verbose=False)

#     def predict(self, guess: Iterable[int | float | bool | str]) -> list[int] | list[bool] | list[str]:
#         return [x[0] for x in self.clf.predict(guess)]


# class CatBoostRegressor(Model):
#     input_formats = {Iterable[int | float | bool]}
#     output_formats = {List[float]}
#     hyperparameters = [
#         {
#             "name": "iterations",
#             "type": "int",
#             "min": 1,
#             "max": 1000,
#             "optional": False
#         },
#         {
#             "name": "learning_rate",
#             "type": "float",
#             "min": 0.0,
#             "max": 1.0,
#             "optional": False
#         },
#         {
#             "name": "depth",
#             "type": "int",
#             "min": 1,
#             "max": 16,
#             "optional": False
#         },
#         {
#             "name": "l2_leaf_reg",
#             "type": "float",
#             "min": 0.0,
#             "max": 10.0,
#             "optional": False
#         },
#         {
#             "name": "model_size_reg",
#             "type": "float",
#             "min": 0.0,
#             "max": 10.0,
#             "optional": True
#         },
#         {
#             "conditional": True,
#             "condition": "single/multi",
#             "variants": [{
#                 "name": "loss_function",
#                 "type": "categorical",
#                 "choices": ['MAE', 'MAPE', 'Poisson', 'Quantile', 'RMSE', 'LogLinQuantile'],
#                 "optional": False
#             },
#                 {
#                 "name": "loss_function",
#                 "type": "no_choice",
#                 "choices": ['MultiRMSE'],
#                 "optional": False
#             }]
#         },
#         {
#             "name": "feature_border_type",
#             "type": "categorical",
#             "choices": ['Median', 'Uniform', 'UniformAndQuantiles', 'GreedyLogSum', 'MaxLogSum', 'MinEntropy'],
#             "optional": False
#         },
#         {
#             "name": "min_data_in_leaf",
#             "type": "int",
#             "min": 1,
#             "max": 10,
#             "optional": False
#         },
#         {
#             "conditional": True,
#             "condition": "cpu/cuda",
#             "variants": [{
#                 "name": "device",
#                 "type": "no_choice",
#                 "choices": ['CPU'],
#                 "optional": False
#             }, {
#                 "name": "device",
#                 "type": "no_choice",
#                 "choices": ['GPU'],
#                 "optional": False
#             }]
#         }
#     ]

#     def __init__(self, iterations: int = 500,
#                  learning_rate: float = 0.03,
#                  depth: int = 6,
#                  l2_leaf_reg: float = 3.0,
#                  model_size_reg: float = None,
#                  loss_function: str = 'MAE',
#                  feature_border_type: Literal['Median', 'Uniform', 'UniformAndQuantiles',
#                                               'GreedyLogSum', 'MaxLogSum', 'MinEntropy'] = 'GreedyLogSum',
#                  min_data_in_leaf: int = 1,
#                  device: Literal['CPU', 'GPU'] = 'CPU',
#                  devices: List[int] = [0],
#                  ** options) -> None:
#         super().__init__()
#         self.reg = cb.CatBoostRegressor(
#             iterations=iterations,
#             learning_rate=learning_rate,
#             depth=depth,
#             l2_leaf_reg=l2_leaf_reg,
#             model_size_reg=model_size_reg,
#             loss_function=loss_function,
#             feature_border_type=feature_border_type,
#             min_data_in_leaf=min_data_in_leaf,
#             task_type=device,
#             devices=devices,
#             **options
#         )

#     def train(self, data: Iterable[int | float | bool], target: Iterable) -> None:
#         self.reg = self.reg.fit(data, target, verbose=False)

#     def predict(self, guess: Iterable[int | float | bool]) -> list[float]:
#         return self.reg.predict(guess)


# if __name__ == "__main__":
#     train_data, test_data, train_labels, test_labels = train_test_split(
#         *get_iris())
#     model = CatBoostClassifier(loss_function='MultiClass')
#     model.train(train_data, train_labels)
#     preds_class = model.predict(test_data)
#     print("class = ", preds_class)
