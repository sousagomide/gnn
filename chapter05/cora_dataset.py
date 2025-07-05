# Introdução: Dataset criado por Sen et al. 2008. É um dataset para 
# classificação de nó em literatura científica. Representa uma rede 
# de 2708 publicações. Cada publicação é descrita como um vetor 
# binário de 1433 palavras únicas, onde 0 e 1 indicam a ausência 
# ou presença de palavras correspondentes.
from typing import cast
from torch_geometric.datasets import Planetoid
from torch_geometric.data import Data

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

