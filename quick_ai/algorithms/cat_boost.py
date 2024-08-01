import catboost as cb
from ..base import Model
from collections.abc import Iterable


class CatBoostClassifier(Model):
    input_formats = {Iterable[int | float | bool | str]}
    output_formats = {list[int], list[str], list[bool]}

    def __init__(self, iterations=None,
                 learning_rate=None,
                 depth=None,
                 l2_leaf_reg=None,
                 model_size_reg=None,
                 rsm=None,
                 loss_function=None,
                 eval_metric=None,
                 bootstrap_type=None,
                 min_data_in_leaf=None,
                 **options) -> None:
        super().__init__()
        self.clf = cb.CatBoostClassifier(
            iterations=iterations,
            learning_rate=learning_rate,
            depth=depth,
            l2_leaf_reg=l2_leaf_reg,
            model_size_reg=model_size_reg,
            rsm=rsm,
            loss_function=loss_function,
            min_data_in_leaf=min_data_in_leaf,
            eval_metric=eval_metric,
            bootstrap_type=bootstrap_type,
            **options
        )

    def train(self, data: Iterable[int | float | bool | str], target: Iterable) -> None:
        self.clf = self.clf.fit(data, target)

    def predict(self, guess: Iterable[int | float | bool | str]) -> list[int] | list[bool] | list[str]:
        return self.clf.predict(guess)


class CatBoostRegressor(Model):
    input_formats = {Iterable[int | float | bool]}
    output_formats = {List[float]}

    def __init__(self, iterations=None,
                 learning_rate=None,
                 depth=None,
                 l2_leaf_reg=None,
                 model_size_reg=None,
                 rsm=None,
                 loss_function='RMSE',
                 eval_metric=None,
                 bootstrap_type=None,
                 min_data_in_leaf=None,
                 **options) -> None:
        super().__init__()
        self.reg = cb.CatBoostRegressor(
            iterations=iterations,
            learning_rate=learning_rate,
            depth=depth,
            l2_leaf_reg=l2_leaf_reg,
            model_size_reg=model_size_reg,
            rsm=rsm,
            loss_function=loss_function,
            eval_metric=eval_metric,
            bootstrap_type=bootstrap_type,
            min_data_in_leaf=min_data_in_leaf,
            **options
        )

    def train(self, data: Iterable[int | float | bool], target: Iterable) -> None:
        self.reg = self.reg.fit(data, target)

    def predict(self, guess: Iterable[int | float | bool]) -> list[float]:
        return self.reg.predict(guess)


# train_data = np.random.choice([True, False], size=(100, 10))
# train_labels = np.random.choice([False, True],
#                                 size=(100))
# test_data = catboost_pool = cb.Pool(train_data,
#                                     train_labels)
# model = CatBoostRegressor(iterations=2,
#                           depth=2,
#                           learning_rate=1,
#                           loss_function='RMSE',
#                           logging_level='Silent',
#                           allow_writing_files=False)
# # train the model
# model.train(train_data, train_labels)
# # make the prediction using the resulting model
# preds_class = model.predict(test_data)
# print("class = ", preds_class)
