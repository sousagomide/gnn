# Introdução: Dataset criado por Sen et al. 2008. É um dataset para 
# classificação de nó em literatura científica. Representa uma rede 
# de 2708 publicações. Cada publicação é descrita como um vetor 
# binário de 1433 palavras únicas, onde 0 e 1 indicam a ausência 
# ou presença de palavras correspondentes.
import pandas as pd
import torch
import torch.nn.functional as F
from typing import cast
from torch_geometric.datasets import Planetoid
from torch_geometric.data import Data
from torch.nn import Linear


dataset = Planetoid(root='.', name='Cora')
data = cast(Data, dataset[0])
print(f'Dataset: {dataset}')
print('-' * 30)
print(f'Number of graphs: {len(dataset)}')
print(f'Number of nodes: {dataset.x.shape[0]}')
print(f'Number of features: {dataset.num_features}')
print(f'Number of classes: {dataset.num_classes}')
# Detalhes sobre o grafo
print(f'Graph:')
print('-' * 30)
print(f'Edges are directed: {data.is_directed()}')
print(f'Graph has isolated nodes: {data.has_isolated_nodes()}')
print(f'Graph has loops: {data.has_self_loops()}')
# Acessando os dados no formato de tabela
# df_x = pd.DataFrame(data.x.numpy())
# df_x['label'] = pd.DataFrame(data.y)
# print(df_x.head())
# A seguir será desenvolvido um Multilayer Perceptron (MLP) e treinado o data.x 
# com os labels obtidos do data.y. Serão criados 4 métodos: 
# __init__(): inicializar a instância; 
# forward(): fazer o forward pass; 
# fit(): modelo de treinamento; 
# test(): para avaliação
# Antes do treinamento é necessário definir a métrica principal. Há muitas métricas 
# para o problema de classificação multiclasse: accuracy, F1 score, ROC AUC score. 
def accuracy(y_pred, y_true):
    return torch.sum(y_pred == y_true) / len(y_true)

class MLP(torch.nn.Module):
    # dim_in: número de neurônios de entrda
    # dim_h: número de neurônios da camada oculta
    # dim_out: número de neurônios da camada de saída
    def __init__(self, dim_in, dim_h, dim_out):
        super().__init__()
        self.linear1 = Linear(dim_in, dim_h)
        self.linear2 = Linear(dim_h, dim_out)
    # Esse método faz um forward pass. A entrada é alimentada para primeira camada 
    # lienar uma funão de ativação ReLU. O resultado passa para a segunda camada 
    # linear. É retornado o log softmax do resultado final da classificação.
    def forward(self, x):
        x = self.linear1(x)
        x = torch.relu(x)
        x = self.linear2(x)
        return F.log_softmax(x, dim=1)
    # O método fit é carregado para o treinamento. Primeiro, é inicializado uma 
    # função de perda e um otimizador que poderá ser usado durante o processo de 
    # treinamento.
    def fit(self, data, epochs):
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=0.01, weight_decay=5e-4)
        # Um laço de treinamento PyTorch é implementado. É usado a função accuracy() 
        # para função de perda
        self.train()
        for epoch in range(epochs + 1):
            optimizer.zero_grad()
            out = self(data.x)
            loss = criterion(out[data.train_mask], data.y[data.train_mask])
            acc = accuracy(out[data.train_mask].argmax(dim=1), data.y[data.train_mask])
            loss.backward()
            optimizer.step()
            if epoch % 20 == 0:
                val_loss = criterion(out[data.val_mask], data.y[data.val_mask])
                val_acc = accuracy(out[data.val_mask].argmax(dim=1), data.y[data.val_mask])
                print(f'Epoch {epoch:>3} | Train Loss: {loss:.3f} | Train Acc: {acc*100:>5.2f}% | Val Loss: {val_loss:.2f} | Val Acc: {val_acc*100:.2f}%')
    # O método test() avalia o modelo com um conjunto de testes e retorna a accuracy
    def test(self, data):
        self.eval()
        out = self(data.x)
        acc = accuracy(out.argmax(dim=1)[data.test_mask], data.y[data.test_mask])
        return acc
# Testando a MLP
mlp = MLP(dataset.num_features, 16, dataset.num_classes)
print(mlp)
# Pega o número de características e treina em 100 épocas
mlp.fit(data, epochs=100)
acc = mlp.test(data)
print(f'MLP test accuracy: {acc*100:.2f}%')

