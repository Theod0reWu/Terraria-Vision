from base_dataset import TestDataset
from base_classifier import TileClassifier
import os
import cv2
from tqdm import tqdm

td = TestDataset()
model_path = os.path.join("..", "models", "model.pkl")
tile_classifier = TileClassifier(model_path)
predictions = []
TOP_K = 5
"""
def p(i):
    x = td[i]
    x = Image.fromarray(x.numpy())
    #x.show()
    x = x.resize((8,8))
    #x.show()
    print(x)
    x = np.asarray(x) / 255.0
    print(x.shape)
    prediction = tile_classifier(x, TOP_K)
    print(prediction)
"""


for x in tqdm(td):
    x = x.numpy()
    img = cv2.resize(x, (8, 8), interpolation=cv2.INTER_LINEAR)/255.0
    predictions.append(tile_classifier(img, TOP_K)[0][0])

print(predictions)