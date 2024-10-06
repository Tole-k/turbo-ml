from quick_ai import QuickAI
from datasets import get_iris

import pandas as pd


import pandas as pd
import numpy as np


def test_happypath():
    dataset, target = get_iris()
    dataset['target'] = target
    random = dataset.sample(n=6)
    dataset.drop(random.index, inplace=True)
    test = random['target']
    random.drop('target', axis=1, inplace=True)
    quickai = QuickAI(dataset=dataset, target='target')
    result = quickai(random)
    assert result is not None
    assert len(result) == len(test)
    assert all(i in target for i in result)
    assert quickai.model.__class__.__name__ != 'RandomGuesser'


if __name__ == '__main__':
    test_happypath()