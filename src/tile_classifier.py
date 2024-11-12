from sklearn.svm import LinearSVC
import numpy as np
from sklearn.model_selection import train_test_split
from dataset import TileDataset
from tqdm import tqdm
import torch
import os

model_path_name = os.path.join("..", "models", "embedding_network.pth")

model = torch.load(model_path_name)

td = TileDataset("../dataset")


x_train = []
y_train = []

model.eval()

for i in tqdm(range(len(td))):
    img, label = td.__getitem__(i)
    embeddings = model(img.unsqueeze(0)).detach().numpy()

    x_train.append(embeddings)
    y_train.append(label)

x_train = np.reshape(x_train, (-1, 128))
np.save(os.path.join("..", "embeddings", "embeddings.npy"), x_train)
np.save(os.path.join("..", "embeddings", "labels.py"), y_train)

x_train, x_test, y_train, y_test = train_test_split(x_train,y_train, test_size=.2)

clf = LinearSVC()
clf.fit(x_train, y_train)

print(clf.score(x_test, y_test))


"""
from sklearn.neighbors import KNeighborsClassifier
for i in range(1, 100):
    clf = KNeighborsClassifier(n_neighbors=i)
    clf.fit(x_train, y_train)
    print(clf.score(x_test, y_test))
"""
"""
from sklearn.linear_model import LogisticRegression

clf = LogisticRegression()
clf.fit(x_train, y_train)
print(clf.score(x_test, y_test))
"""

