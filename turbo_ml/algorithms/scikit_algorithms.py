from turbo_ml.base import Model
from sklearn.utils import all_estimators
from typing import Dict, Type
SCIKIT_MODELS: Dict[str, Type[Model]] = {}


def _train(self: Model, data, target):
    self.model.fit(data, target)


def _predict(self: Model, data):
    return self.model.predict(data)


def _classifier_init(self: Model, **kwargs):
    self.model = self.classifier(**kwargs)


for name, classifier in all_estimators(type_filter='classifier'):
    try:
        classifier_obj = classifier()
        model = type(name, (Model,),
                     {'classifier': classifier, 'train': _train, 'predict': _predict, '__init__': _classifier_init})
        SCIKIT_MODELS[name] = model
    except TypeError as e:
        continue

if __name__ == '__main__':
    for i, model in enumerate(SCIKIT_MODELS):
        print(i, model)