import torch
import os
from sklearn.decomposition import PCA
import numpy as np
from dataset import PairDataset
import matplotlib.pyplot as plt
from siamese_model import EmbeddingNetwork

model = EmbeddingNetwork()

model_path_name = os.path.join("..", "models", "embedding_network.pth")

weights = torch.load(model_path_name)
model.load_state_dict(weights)
model.eval()
pd = PairDataset(os.path.join("..", "dataset"))
embeddings = []
class_labels = []

for i in range(5):

    dirname = os.path.join("..", "dataset", pd.dirnames[i])
    num_tiles = len(os.listdir(dirname))
    for j in range(num_tiles):
        img = pd.load_image(i, j).unsqueeze(0)
        embeddings.append(model(img).detach().numpy())
        class_labels.append(i)

embeddings = np.array(embeddings)
embeddings = np.reshape(embeddings, (-1, 128))

column_mean = np.mean(embeddings, axis=0)
column_std = np.std(embeddings, axis=0)

embeddings = (embeddings - column_mean) / column_std

pca = PCA(n_components=2)
projected_embeddings = pca.fit_transform(embeddings)

plt.figure(figsize=(8, 6))
for label in np.unique(class_labels):
    indices = np.where(class_labels == label)
    plt.scatter(projected_embeddings[indices, 0], projected_embeddings[indices, 1], label=f"Tile {label}")

plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.legend()
plt.title("PCA Projection of Tile Embeddings")
plt.savefig(os.path.join("..", "artifacts", "Embedding_Visualization.png"))
plt.show()