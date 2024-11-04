import torch






class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, embedding1, embedding2, label):
        distances = torch.norm(embedding1 - embedding2, p=2, dim=1)
        loss_similar = label * distances**2
        loss_dissimilar = (1 - label) * torch.clamp(self.margin - distances, min=0)**2
        return torch.mean(loss_similar + loss_dissimilar) / 2
