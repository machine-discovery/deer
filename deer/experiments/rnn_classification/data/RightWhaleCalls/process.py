from typing import Optional, Dict
from scipy.io import arff
import pandas as pd
import numpy as np
import pickle


def arff_to_np(arff_file: str, target_map: Optional[Dict[str, int]] = None):
    data, meta = arff.loadarff(arff_file)
    print("Reading finished")
    data_lst = []
    targets = []
    for row in data:
        row_lst = []
        target = None
        for i, col in enumerate(row):
            if i < len(row) - 1:
                row_lst.append(col)
            else:
                target = col
        assert target is not None
        data_lst.append(row_lst)
        targets.append(target)
    print("Converting to list finished")

    data_np = np.array(data_lst)

    if target_map is None:
        targets_unique = list(set(targets))
        target_map = {target: i for i, target in enumerate(targets_unique)}
    targets_np = np.array([target_map[target] for target in targets])
    print("Processing data finished")

    return data_np, targets_np, target_map

prefix = "RightWhaleCalls"
print("Processing train data")
train_data, train_target, target_map = arff_to_np(f"{prefix}_TRAIN.arff")

print("Processing test data")
test_data, test_target, _ = arff_to_np(f"{prefix}_TEST.arff", target_map)

print("Saving data")
with open(f"{prefix}_TRAIN.pkl", "wb") as f:
    pickle.dump((train_data, train_target, target_map), f)

with open(f"{prefix}_TEST.pkl", "wb") as f:
    pickle.dump((test_data, test_target, target_map), f)
