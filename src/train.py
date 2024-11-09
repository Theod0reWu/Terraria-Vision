from dataset import PairDataset
from siamese_model import EmbeddingNetwork, ContrastiveLoss
import os
import torch.optim as optim
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm


dataset = PairDataset(os.path.join("..", "dataset"))
data_loader = DataLoader(dataset, batch_size=32)

model = EmbeddingNetwork(embedding_dim=128)
contrastive_loss = ContrastiveLoss(margin=1.0)
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 10
for epoch in range(num_epochs):

    model.train()
    epoch_loss = 0.0
    for batch_idx, (img1, img2, labels) in tqdm(enumerate(data_loader), total=len(dataset) // 32):

        embedding1 = model(img1)
        embedding2 = model(img2)

        loss = contrastive_loss(embedding1, embedding2, labels.float())
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        
    avg_loss = epoch_loss / len(data_loader)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

model_path_name = os.path.join("..", "models", "embedding_network.pth")
torch.save(model, model_path_name)