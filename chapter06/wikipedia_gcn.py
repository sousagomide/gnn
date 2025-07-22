# A regressão refere-se a predição de valores contínuos. Esse código é 
# escrito para predizer valor contínuos ao invés de variáveis 
# categóricas. Para isso será utilizado o dataset Wikipedia Network
from collections import Counter
from scipy.stats import  norm
from sklearn.metrics import mean_squared_error, mean_absolute_error
from torch_geometric.datasets import WikipediaNetwork
from torch_geometric.nn import GCNConv
from torch_geometric.utils import degree
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn.functional as F
import torch_geometric.transforms as T

dataset = WikipediaNetwork(root='.', name='chameleon', transform=T.RandomNodeSplit(num_val=200, num_test=500))
data = dataset[0]

print(f'Dataset: {dataset}')
print('-' * 30)
print(f'Number of graphs: {len(dataset)}')
print(f'Number of nodes: {data.x.shape[0]}')
print(f'Number of unique features: {dataset.num_features}')
print(f'Number of classes: {dataset.num_classes}')

# # Trecho do código usado para verificar a distribuição dos dados normalizados
# df = pd.read_csv('wikipedia/chameleon/musae_chameleon_target.csv')
# values = np.log10(df['target'])
# data.y = torch.tensor(values)
# degrees = degree(data.edge_index[0]).numpy()
# # Conta o número de nós para cada grau
# numbers = Counter(degrees)
# # Plota o resultado
# fig, ax = plt.subplots()
# ax.set_xlabel('Node degree')
# ax.set_ylabel('Number of nodes')
# plt.bar(numbers.keys(), numbers.values())
# plt.show()
# # Plota o histograma dos dados normalizados
# df['target'] = values
# sns.histplot(df['target'], kde=True, stat='density', bins=30, color='skyblue')
# mu, std = norm.fit(df['target'])
# xmin, xmax = plt.xlim()
# x = np.linspace(xmin, xmax, 100)
# p = norm.pdf(x, mu, std)
# plt.plot(x, p, 'r', linewidth=2)
# plt.title(f'Ajuste Normal: μ = {mu:.2f}, σ = {std:.2f}')
# plt.xlabel('Target')
# plt.ylabel('Densidade')
# plt.show()
class GCN(torch.nn.Module):

    def __init__(self, dim_in, dim_h, dim_out):
        super().__init__()
        self.gcn1 = GCNConv(dim_in, dim_h*4)
        self.gcn2 = GCNConv(dim_h*4, dim_h*2)
        self.gcn3 = GCNConv(dim_h*2, dim_h)
        self.linear = torch.nn.Linear(dim_h, dim_out)

    # A função forward não usa o log_softmax por não se tratar de uma 
    # classe de predição
    def forward(self, x, edge_index):
        h = self.gcn1(x, edge_index)
        h = torch.relu(h)
        h = F.dropout(h, p=0.5, training=self.training)
        h = self.gcn2(h, edge_index)
        h = torch.relu(h)
        h = F.dropout(h, p=0.5, training=self.training)
        h = self.gcn3(h, edge_index)
        h = torch.relu(h)
        h = self.linear(h)
        return h
    
    # O método fit agora passa a usar o Mean Squared Error (MSE)
    def fit(self, data, epochs):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.02, weight_decay=5e-4)
        self.train()
        for epoch in range(epochs+1):
            optimizer.zero_grad()
            out = self(data.x, data.edge_index)
            loss = F.mse_loss(out.squeeze()[data.train_mask], data.y[data.train_mask].float())
            loss.backward()
            optimizer.step()
            if epoch % 20 == 0:
                val_loss = F.mse_loss(out.squeeze()[data.val_mask], data.y[data.val_mask])
                print(f'Epoch {epoch:>3} | Train Loss: {loss:.5f} | Val Loss: {val_loss:.5f}')
    
    @torch.no_grad()
    def test(self, data):
        self.eval()
        out = self(data.x, data.edge_index)
        return F.mse_loss(out.squeeze()[data.test_mask], data.y[data.test_mask].float())

gcn = GCN(dataset.num_features, 128, 1)        
print(gcn)
gcn.fit(data, epochs=200)
loss = gcn.test(data)
print(f'GCN test loss: {loss:.5f}')
# O MSE loss não é uma métrica interpretável. Para atingir melhores 
# resultados podemos utilizar duas métricas: RMSE; e o Mean Absolute 
# Error (MAE)
# Converte o PyTorch tensor para predições em um vetor de NumPy 
# usando .detach().numpy()
out = gcn(data.x, data.edge_index)
y_pred = out.squeeze()[data.test_mask].detach().numpy()
mse = mean_squared_error(data.y[data.test_mask], y_pred)
mae = mean_absolute_error(data.y[data.test_mask], y_pred)
print('-'*50)
print(f'MSE = {mse:.4f} | RMSE = {np.sqrt(mse):.4f} | MAE = {mae:.4f}')
print('-'*50)
fig = sns.regplot(x=data.y[data.test_mask].numpy(), y=y_pred)
plt.xlabel('Ground truth')
plt.ylabel('Predicted Values')
plt.show()


