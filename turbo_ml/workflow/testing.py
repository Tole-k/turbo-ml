from prefect import flow
from turbo_ml.turbo_ml_experimental import TurboML_Experimental
from turbo_ml.meta_learning import MetaModelGuesser
from datasets import get_iris

@flow(name='Test TurboML')
def test_TurboML(path:str, param_function):
    dataset, y = get_iris()
    dataset['species'] = y
    turboml = TurboML_Experimental(dataset, target='species', hpo_enabled=False,
                                 guesser=MetaModelGuesser(path=path), param_function=param_function)
    results = turboml.predict(dataset)
    return results