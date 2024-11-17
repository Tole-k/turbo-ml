
from typing import List
from benchmark.utils import TEST_DURATIONS, _FAMILY_MAPPING
from scipy.stats import friedmanchisquare
import pandas as pd
import numpy as np
import os
import scikit_posthocs as sp
import yaml


def friedman_test(experiments: List[str], datasets: List[str], timestamps: List[int]) -> pd.DataFrame:
    inv_family_map = {v.name: k for k, v in _FAMILY_MAPPING.items()}

    family_scores = pd.read_csv(os.path.join(
        "data", "family_scores.csv"), index_col=0)

    scores = {timestamp: {experiment: {dataset: [] for dataset in datasets}
                          for experiment in experiments} for timestamp in timestamps}
    for file in os.listdir(os.path.join('benchmark', 'outputs')):
        experiment = file.split('-')[0]
        if experiment not in experiments:
            continue
        with open(os.path.join('benchmark', 'outputs', file), 'r') as f:
            data = yaml.safe_load(f)
            for seed in data:
                for timestamp in data[seed]:
                    for dataset in data[seed][timestamp]:

                        family = inv_family_map[list(dataset.values())[0][0]]
                        dataset = list(dataset.keys())[0]

                        score = family_scores.loc[dataset, family]

                        scores[timestamp][experiment][dataset].append(score)
            for timestamp in scores:
                for dataset in scores[timestamp][experiment]:
                    scores[timestamp][experiment][dataset] = np.mean(
                        scores[timestamp][experiment][dataset])

    p_significance = 0.05
    ranks_list = []
    is_significant = []
    for timestamp in scores:
        df = pd.DataFrame(scores[timestamp])
        # print(f"Timestamp: {timestamp}")
        # print(df, '\n')
        stat, p_value = friedmanchisquare(
            *[df[col] for col in df.columns], nan_policy='omit')
        # print("P-value:", p_value)
        # print(f"Friedman Test Statistic: {stat}")
        # print(f"p-value: {p_value}")

        if p_value > p_significance:
            # print("No significant difference between the models")
            is_significant.append(False)
        else:
            # print("Significant difference between the models")
            is_significant.append(True)

        ranks = df.rank(axis=1, method="average", ascending=False)
        average_ranks = ranks.mean()
        # print("\nAverage Ranks:")
        # print(average_ranks)
        ranks_list.append(average_ranks)
        nemenyi_result = sp.posthoc_nemenyi_friedman(df.to_numpy())
        # print(nemenyi_result)
        # print("\nNemenyi Post Hoc Results:")
        ranking = pd.DataFrame(nemenyi_result)
        valid_experiemnts = [
            experiment for experiment in experiments if not np.all(np.isnan(np.array(list(scores[timestamp][experiment].values()))))]
        ranking.columns = valid_experiemnts
        ranking.index = valid_experiemnts

        for i in range(len(ranking)):
            for j in range(len(ranking)):
                if ranking.iloc[i, j] < p_significance:
                    if average_ranks.iloc[i] < average_ranks.iloc[j]:
                        ranking.iloc[i, j] = 1
                    else:
                        ranking.iloc[i, j] = -1
                else:
                    ranking.iloc[i, j] = 0
        print(ranking)
        path = os.path.join('benchmark', 'friedman_results',
                            "pairwise-comparisons")
        if not os.path.exists(path):
            os.makedirs(path)
        ranking.to_csv(os.path.join(path, f"after_{timestamp}s.csv"))
        # print("\n")
    ranks = pd.DataFrame(ranks_list)
    ranks.insert(0, "Timestamp", timestamps)
    ranks['is_significant'] = is_significant
    ranks.set_index("Timestamp", inplace=True)
    ranks.to_csv(os.path.join(
        'benchmark', 'friedman_results', 'average_ranks.csv'))
    return ranks


if __name__ == "__main__":
    experiments = [
        "AutoGluonExperiment",
        "EvalMlExperiment",
        # "H2OExperiment",
        # "PyCaretExperiment",
        # "SklearnExperiment",
        "TPotExperiment",
        # "TurboMLExperiment",

    ]
    datasets = pd.read_csv(os.path.join("data", "parameters.csv"))
    datasets = datasets["name"]
    datasets = datasets.to_list()
    datasets = datasets[:5]

    friedman_test(experiments, datasets, TEST_DURATIONS)
