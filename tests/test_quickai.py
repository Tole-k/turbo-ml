from quick_ai import QuickAI
from datasets import get_iris

import pandas as pd


import pandas as pd
import numpy as np


def auto_convert(df):
    for col in df.columns:
        # Attempt to convert to numeric with errors='coerce'
        converted = pd.to_numeric(df[col], errors='coerce')
        # Check if all non-null values are numeric
        if converted.notnull().all() or converted.dropna().shape[0] > 0:
            # All data is numeric or convertible to numeric (excluding NaNs)
            df[col] = converted.astype(float)
        else:
            # Non-numeric data present, convert to category
            df[col] = df[col].astype('category')
    return df

# Usage:
# df = auto_convert(df)

# Usage:
# df = auto_convert(df)


def test_happypath():
    dataset, target = get_iris()
    dataset['target'] = target
    # dataset = auto_convert(dataset)
    random = dataset.sample(n=5)
    dataset.drop(random.index, inplace=True)
    test = random['target']
    random.drop('target', axis=1, inplace=True)
    print(random)
    print('-'*100)
    print(dataset)
    quickai = QuickAI(dataset=dataset, target='target')
    print('-'*100)
    print(quickai(random))
    print(test)


test_happypath()
