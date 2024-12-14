from base_dataset import TestDataset
from base_classifier import TileClassifier
import os
import cv2
from tqdm import tqdm
import torchvision.transforms as transforms
import pickle
import numpy as np

td = TestDataset()
print(len(td))
model_path = os.path.join("..", "models", "model.pkl")

class_names = []
with open("TileID.txt", "r") as f:
      for line in f:
          class_names.append(line.split("\t")[0])

#tile_classifier = TileClassifier(model_path)

with open(model_path, 'rb') as f:
    clf = pickle.load(f)

clf.classes_ = np.array(class_names)
predictions = []
TOP_K = 5

#grayscale_transform = transforms.Grayscale(num_output_channels=1)

predictions = []
for idx, x in enumerate(tqdm(td)):
    if idx > 10:
        break
    
    #x = grayscale_transform(x)
    x = x[0].numpy()
    img = cv2.resize(x, (8, 8), interpolation=cv2.INTER_LINEAR)
    img = np.reshape(img, (1, -1))/255.0
    #cv2.imshow("Window", img)
    predictions.append(clf.predict(img))
    #predictions.append(tile_classifier(img, TOP_K)[0])


print(predictions)