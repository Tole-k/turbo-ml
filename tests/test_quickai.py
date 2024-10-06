from turbo_ml import TurboML
from datasets import get_iris


def test_happypath():
    dataset, target = get_iris()
    dataset['target'] = target
    random = dataset.sample(n=6)
    dataset.drop(random.index, inplace=True)
    test = random['target']
    random.drop('target', axis=1, inplace=True)
    truboml = TurboML(dataset=dataset, target='target')
    result = truboml(random)
    assert result is not None
    assert len(result) == len(test)
    assert all(i in target for i in result)
    assert truboml.model.__class__.__name__ != 'RandomGuesser'


if __name__ == '__main__':
    test_happypath()
