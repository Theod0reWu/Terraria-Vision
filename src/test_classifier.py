from base_dataset import TestDataset
from base_classifier import TileClassifier
import os
from tqdm import tqdm
from torchvision.transforms import Resize

import pickle
import numpy as np


td = TestDataset()
print(len(td))
model_path = os.path.join("..", "models", "model.pkl")

class_names = []
with open("TileID.txt", "r") as f:
      for line in f:
          class_names.append(line.split("\t")[0])

with open(model_path, 'rb') as f:
    clf = pickle.load(f)

clf.classes_ = np.array(class_names)
predictions = []
TOP_K = 5

transform = Resize(size=(8, 8), antialias=True)

predictions = []
for idx, x in enumerate(tqdm(td)):
    if idx > 10:
        break
    x = x[0].permute(2, 0, 1)
    x = transform(x)
    x = x.numpy()
    img = np.reshape(x, (1, -1))/255.0
    predictions.append(clf.predict(img))

print(predictions)