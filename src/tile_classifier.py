from sklearn.svm import LinearSVC
import numpy as np
from sklearn.model_selection import train_test_split
from dataset import TileDataset
from tqdm import tqdm
import torch
import os
from siamese_model import EmbeddingNetwork


model = EmbeddingNetwork()

model_path_name = os.path.join("..", "models", "embedding_network.pth")

weights = torch.load(model_path_name)
model.load_state_dict(weights)
model.eval()

td = TileDataset("../dataset")

"""
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
import matplotlib.pyplot as plt

vals, freqs = np.unique(y_train, return_counts=True)

x, counts = np.unique(freqs, return_counts=True)
cusum = np.cumsum(counts)
plt.plot(x, cusum/ cusum[-1])
#plt.hist(freqs, bins=[0, 5, 5, 10, 15, 100] + list(range(200, max(freqs), 100)))
plt.show()

"""
from sklearn.ensemble import GradientBoostingClassifier
clf = GradientBoostingClassifier(n_estimators=10, learning_rate=1.0,
    max_depth=2, max_features ="log2", random_state=0).fit(x_train, y_train)
score = clf.score(x_test, y_test)
print(score)
"""

"""
clf = LinearSVC()
clf.fit(x_train, y_train)

print(clf.score(x_test, y_test))
"""

"""

from sklearn.ensemble import RandomForestClassifier


clf = RandomForestClassifier(max_depth=25, random_state=0)
clf.fit(x_train, y_train)
print(clf.score(x_test, y_test))
"""

from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier(n_neighbors=10, weights="distance", metric="cosine")
clf.fit(x_train, y_train)
print(clf.score(x_train, y_train))
print(clf.score(x_test, y_test))

def get_top_k_acc(k):
    num_correct = 0
    for feat, label in tqdm(zip(x_test, y_test)):
        probs = clf.predict_proba(feat.reshape(1, -1))[0]
        indices = np.argsort(probs)[::-1]
        top_k_labels = clf.classes_[indices[:k]]
        if label in top_k_labels:
            num_correct += 1
    return num_correct / len(x_test)

print(get_top_k_acc(5))

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