from turbo_ml.meta_learning.meta_model import generate_dataset, ALL_MODELS, ALL_DATASETS

chosen_sets = [
    "energy-y2_R.dat",
    "prnn_synth.csv",
    "pittsburg-bridges-T-OR-D_R.dat",
    "ar4.csv",
    "glass_R.dat",
    "plant-margin_R.dat",
    "kc3.csv",
    "teachingAssistant.csv",
]

ALL_DATASETS = [d for d in ALL_DATASETS if d[1] in chosen_sets]
generate_dataset(ALL_MODELS, ALL_DATASETS, device="cuda",
                 threads=-1, path="performance_results.csv")
