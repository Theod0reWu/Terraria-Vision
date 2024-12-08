from sklearn.svm import LinearSVC
import numpy as np
from sklearn.model_selection import train_test_split
from dataset import TileDataset
from tqdm import tqdm
import torch
import os
from siamese_model import EmbeddingNetwork

"""
model = EmbeddingNetwork()

model_path_name = os.path.join("..", "models", "embedding_network.pth")

weights = torch.load(model_path_name)
model.load_state_dict(weights)
model.eval()

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
"""

x_train = np.load(os.path.join("..", "embeddings", "embeddings.npy"))
y_train = np.load(os.path.join("..", "embeddings", "labels.npy"))
x_train, x_test, y_train, y_test = train_test_split(x_train,y_train, test_size=.2)
print(x_train.shape)

"""
clf = LinearSVC()
clf.fit(x_train, y_train)

print(clf.score(x_test, y_test))
"""

from sklearn.ensemble import RandomForestClassifier


clf = RandomForestClassifier(max_depth=25, random_state=0)
clf.fit(x_train, y_train)
print(clf.score(x_test, y_test))



"""
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA

for d in range(20, 100, 10):
    pca = PCA(n_components=d)
    X_train_reduced = pca.fit_transform(x_train)
    X_test_reduced = pca.transform(x_test)


    clf = KNeighborsClassifier(n_neighbors=10, weights="distance", metric="cosine")
    clf.fit(X_train_reduced, y_train)
    print(f"---------------N_Components = {d}------------")
    print(clf.score(X_train_reduced, y_train))
    print(clf.score(X_test_reduced, y_test))
    print("\n\n")

import pickle
clf_model_path = os.path.join("..", "models", "model.pkl")
with open(clf_model_path ,'wb') as f:
    pickle.dump(clf,f)

"""

"""
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