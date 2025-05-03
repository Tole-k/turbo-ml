""" Tests the model if got trained and saved properly """
from sageml.sageml_experimental import SageML_Experimental
from sageml.meta_learning import MetaModelGuesser
from datasets import get_iris


def test_SageML(path: str, param_function):
    dataset, y = get_iris()
    dataset['species'] = y
    sageml = SageML_Experimental(dataset, target='species', hpo_enabled=False,
                                 guesser=MetaModelGuesser(path=path), param_function=param_function)
    sample = dataset.sample(frac=0.1, random_state=42)
    results = sageml.predict(sample.drop(columns=['species']))
    return results
