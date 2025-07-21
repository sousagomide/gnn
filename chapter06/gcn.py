from numpy import number
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import degree
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from collections import Counter
from typing import cast
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt


dataset = Planetoid(root='.', name='Cora')
data = cast(Data, dataset[0])

def accuracy(pred_y, y):
    return ((pred_y == y).sum()/len(y)).item()

# Calcula o número de vizinhos de cada nó do grafo
degrees = degree(data.edge_index[0]).numpy()
# # Conta o número de nós para cada grau
# numbers = Counter(degrees)
# # Plota o resultado
# fig, ax = plt.subplots()
# ax.set_xlabel('Node degree')
# ax.set_ylabel('Number of nodes')
# plt.bar(numbers.keys(), numbers.values())
# plt.show()
class GCN(torch.nn.Module):
    """Graph Convolutional Network"""
    def __init__(self, dim_in, dim_h, dim_out):
        super().__init__()
        self.gcn1 = GCNConv(dim_in, dim_h)
        self.gcn2 = GCNConv(dim_h, dim_out)

    def forward(self, x, edge_index):
        h = self.gcn1(x, edge_index)
        h = torch.relu(h)
        h = self.gcn2(h, edge_index)
        return F.log_softmax(h, dim=1)
    
    def fit(self, data, epochs):
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=0.01, weight_decay=5e-4)
        # Um laço de treinamento PyTorch é implementado. É usado a função accuracy() 
        # para função de perda
        self.train()
        for epoch in range(epochs + 1):
            optimizer.zero_grad()
            out = self(data.x, data.edge_index)
            loss = criterion(out[data.train_mask], data.y[data.train_mask])
            acc = accuracy(out[data.train_mask].argmax(dim=1), data.y[data.train_mask])
            loss.backward()
            optimizer.step()
            if epoch % 20 == 0:
                val_loss = criterion(out[data.val_mask], data.y[data.val_mask])
                val_acc = accuracy(out[data.val_mask].argmax(dim=1), data.y[data.val_mask])
                print(f'Epoch {epoch:>3} | Train Loss: {loss:.3f} | Train Acc: {acc*100:>5.2f}% | Val Loss: {val_loss:.2f} | Val Acc: {val_acc*100:.2f}%')

    @torch.no_grad()
    def test(self, data):
        self.eval()
        out = self(data.x, data.edge_index)
        acc = accuracy(out.argmax(dim=1)[data.test_mask], data.y[data.test_mask])
        return acc

gcn = GCN(dataset.num_features, 16, dataset.num_classes)
print(gcn)
gcn.fit(data, epochs=100)
acc = gcn.test(data)
print(f'GCN test accuracy: {acc*100:.2f}%')