import pickle

with open("RightWhaleCalls_TRAIN.pkl", "rb") as f:
    train_data, train_target, target_map = pickle.load(f)

with open("RightWhaleCalls_TEST.pkl", "rb") as f:
    test_data, test_target, target_map = pickle.load(f)

print(train_data.shape, train_data.dtype)
print(test_data.shape, test_data.dtype)
print(train_target.shape, train_target.dtype)
print(target_map)
