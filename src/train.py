from dataset import RandomPairDataset
from siamese_model import EmbeddingNetwork, ContrastiveLoss
import os
import torch.optim as optim
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt


dataset = RandomPairDataset(os.path.join("..", "dataset"))
data_loader = DataLoader(dataset, batch_size=256)

model = EmbeddingNetwork(embedding_dim=128)
contrastive_loss = ContrastiveLoss(margin=1.0)
optimizer = optim.Adam(model.parameters(), lr=0.001)
losses = []

num_epochs = 10
num_iter = int(1e8 // 256)
train_loss = []
iterations = []

data_loader_iter = iter(data_loader)

for train_iter in tqdm(range(num_iter)):
    model.train()
    img1, img2, labels = next(data_loader_iter)
    embedding1 = model(img1)
    embedding2 = model(img2)
    optimizer.zero_grad()
    loss = contrastive_loss(embedding1, embedding2, labels.float())
    loss.backward()
    optimizer.step()
    if train_iter % 100 == 0:
        train_loss.append(loss.item())
        iterations.append(train_iter)

"""

while(num_iter != 0):
    model.train()
    for batch_idx, (img1, img2, labels) in tqdm(enumerate(data_loader), total=len(dataset) // 32):

        embedding1 = model(img1)
        embedding2 = model(img2)

        loss = contrastive_loss(embedding1, embedding2, labels.float())
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        #epoch_loss += loss.item()
        num_iter -= 32
        if num_iter % 100 == 0:
            train_loss.append(loss.item())
    
    avg_loss = epoch_loss / len(data_loader)
    losses.append(avg_loss)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")
"""


model_path_name = os.path.join("..", "models", "embedding_network.pth")
torch.save(model, model_path_name)

plt.plot(iterations, train_loss)
plt.xlabel("Number of Training Iterations")
plt.ylabel("Contrastive Loss")
plt.title("Loss of Embedding Model Against Number of Iterations")
plt.savefig(os.path.join("..", "artifacts", "Embedding_Model_Performance_Performance.png"))