from dataset import RandomPairDataset
from siamese_model import EmbeddingNetwork, ContrastiveLoss
import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = RandomPairDataset(os.path.join("..", "dataset"))
data_loader = DataLoader(dataset, batch_size=256)

model = EmbeddingNetwork(embedding_dim=128).to(device)
contrastive_loss = ContrastiveLoss(margin=1.0).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)

losses = []
num_epochs = 10
num_iter = int(1e7 // 256)
train_loss = []
iterations = []

data_loader_iter = iter(data_loader)

for train_iter in tqdm(range(num_iter)):
    model.train()
    img1, img2, labels = next(data_loader_iter)
    
    img1, img2, labels = img1.to(device), img2.to(device), labels.to(device)

    embedding1 = model(img1)
    embedding2 = model(img2)
    
    optimizer.zero_grad()
    loss = contrastive_loss(embedding1, embedding2, labels.float())
    
    loss.backward()
    optimizer.step()
    
    if train_iter % 1000 == 0:
        train_loss.append(loss.item())
        iterations.append(train_iter)

model_path_name = os.path.join("..", "models", "embedding_network.pth")
torch.save(model.state_dict(), model_path_name)

plt.plot(iterations, train_loss)
plt.xlabel("Number of Training Iterations")
plt.ylabel("Contrastive Loss")
plt.title("Loss of Embedding Model Against Number of Iterations")
plt.savefig(os.path.join("..", "artifacts", "Embedding_Model_Performance.png"))
