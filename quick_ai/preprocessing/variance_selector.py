import pandas as pd
import numpy as np
from ..base.preprocess import Preprocessor
from sklearn.feature_selection import VarianceThreshold

class VarianceSelector(Preprocessor):
    
    def fit_transform(self, data:pd.DataFrame, threshold1:float=0.1,threshold2:float=0.9)->pd.DataFrame:
        data=data.apply(lambda x: x.astype(bool) if x.isin([0,1]).all() else x)
        self.variance_threshold = VarianceThreshold(threshold=threshold1)
        cols = data.columns
        numeric_cols = data.select_dtypes(include=[np.number])
        categorical_cols = data.select_dtypes(include=["category",object,"string",np.bool_]).map(lambda x: str(x))
        cat_cols_left = []
        for col in categorical_cols:
            if categorical_cols[col].value_counts(sort=1)[0]/len(categorical_cols[col])<threshold2:
                cat_cols_left.append(col)
        self.variance_threshold.fit(numeric_cols)
        num_cols_left = self.variance_threshold.get_feature_names_out()
        self.cols_to_rem = set(cols) - set(cat_cols_left) - set(num_cols_left)
        data.drop(columns=self.cols_to_rem,inplace=True)
        return data
    
    def transform(self, data:pd.DataFrame)->pd.DataFrame:
        data.drop(columns=self.cols_to_rem,inplace=True)
        return data


dataset = pd.DataFrame({
    'A': [0.5, 0.5, 0.5, 0.5],
    'B': [0.25, 0.25, 0.25, 1],
    'C': ["c", "c", "c", "d"],
    'D': [0, 1, 0, 1],
    'E': ["c","c","c","c"],
})
selector = VarianceSelector()
print(selector.fit_transform(dataset))
dataset = pd.DataFrame({
    'A': [0.1, 0.2, 0.3, 0.4],
    'B': [0.25, 0.25, 0.25, 0.25],
    'C': ["c", "c", "c", "c"],
    'D': [0, 1, 0, 1],
    'E': ["c","c","c","d"],
})
print(selector.transform(dataset))