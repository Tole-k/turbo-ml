import re
import random
import json
from openml import runs


def download_scores(set_names, set_openml_ids, inv_map, families):
    scores = {name: {family: [] for family in families}
              for name in missing_sets}
    for id, name in zip(set_openml_ids[:10], set_names[:10]):
        print(name)
        runs_df = runs.list_runs(task=[id], output_format='dataframe')
        run_ids = runs_df['run_id'].tolist()
        random.shuffle(run_ids)
        for run_id in run_ids[:100]:
            run = runs.get_run(run_id)
            desc = run.__str__()
            try:
                metric = re.search(
                    r'Metric.{10}: ([^\s]+)', desc).group(0).split(': ')[-1]
            except:
                continue
            if metric != 'predictive_accuracy':
                continue
            flow_name = re.search(
                r'Flow Name[.]+: ([^\s]+)', desc).group(0).split(': ')[-1].split('(')[0]
            result = re.search(
                r'Result.{10}: 0.[0-9]+', desc).group(0).split(': ')[-1]
            if flow_name[:5] == 'weka.':
                flow_name = ('_').join([flow_name[5:], 'w'])
            if flow_name in inv_map:
                family = inv_map[flow_name]
            else:
                continue
            scores[name][family].append(float(result))
        for family in families:
            if len(scores[name][family]) == 0:
                scores[name][family] = 0
            else:
                scores[name][family] = max(scores[name][family])


if __name__ == "__main__":
    missing_sets = ['analcatdata_asbestos', 'analcatdata_boxing1', 'analcatdata_broadwaymult', 'analcatdata_germangss', 'analcatdata_lawsuit', 'ar4', 'autos', 'baseball', 'bodyfat', 'braziltourism', 'chatfield_4', 'chscase_vine1', 'cloud', 'diabetes', 'diggle_table_a2', 'disclosure_z', 'elusage', 'fri_c0_250_5', 'kc3', 'kidney', 'labor', 'lowbwt', 'lupus',
                    'meta', 'mfeat-karhunen', 'mfeat-morphological', 'newton_hema', 'no2', 'plasma_retinol', 'pm10', 'prnn_synth', 'rabe_131', 'rmftsa_sleepdata', 'schizo', 'schlvote', 'sleuth_case2002', 'socmob', 'solar-flare', 'squash-stored', 'squash-unstored', 'tae', 'teachingAssistant', 'transplant', 'triazines', 'veteran', 'visualizing_livestock', 'vote', 'white-clover']

    missing_ids = [
        3550, 3540, 3824, 3887, 3542, 3911, 9, 2077, 3644, 2078, 3685, 3680, 3753, 37, 3683, 3794, 3655, 3642, 3915, 3808, 4, 3804, 3562, 3623, 16, 18, 3649, 3749, 3778, 3616, 3555, 3788, 3607, 3557, 3713, 3765, 3797, 2068, 3835, 3848, 47, 3949, 3748, 3653, 3585, 3731, 55, 3872
    ]

    with open("../turbo_ml/meta_learning/meta_model/algorthm_families.json", "r") as f:
        algorithm_families = json.load(f)
    inv_map = {}
    families = algorithm_families.keys()
    for k, v in algorithm_families.items():
        for i in v:
            inv_map[i] = k

    download_scores(missing_sets, missing_ids, inv_map, families)
