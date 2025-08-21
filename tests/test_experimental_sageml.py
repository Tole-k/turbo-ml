from sageml.sageml_experimental import SageML_Experimental
from sageml.utils import options
from datasets import get_iris


def test_happy_path():
    dataset, target = get_iris()
    dataset['target'] = target
    random = dataset.sample(n=6)
    dataset.drop(random.index, inplace=True)
    test = random['target']
    random.drop('target', axis=1, inplace=True)
    sml = SageML_Experimental(dataset=dataset, target='target',
                              device=options.device, threads=options.threads)
    result = sml(random)
    assert result is not None
    assert len(result) == len(test)
    assert all(i in target for i in result)
    assert sml.model.__class__.__name__ != 'RandomGuesser'


if __name__ == '__main__':
    test_happy_path()
