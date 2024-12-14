from base_dataset import TestDataset
from base_classifier import TileClassifier
import os
import cv2
from tqdm import tqdm

td = TestDataset()
print(len(td))
model_path = os.path.join("..", "models", "model.pkl")
tile_classifier = TileClassifier(model_path)
predictions = []
TOP_K = 5

predictions = []
for idx, x in enumerate(tqdm(td)):
    if idx > 10:
        break
    x = x.numpy()
    img = cv2.resize(x, (8, 8), interpolation=cv2.INTER_LINEAR)/255.0
    predictions.append(tile_classifier(img, TOP_K)[0])

print(predictions)