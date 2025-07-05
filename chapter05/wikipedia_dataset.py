# Introdução: Um dataset que representa redes de links entre páginas da 
# Wikipedia, focado em artigos relacionados a temas específicos. Um 
# dataset que representa redes de links entre páginas da Wikipedia, 
# focado em artigos relacionados a temas específicos. Cada aresta 
# representa um link entre duas páginas. O dataset é utilizado para 
# tarefas de classificação de nós, onde o objetivo é prever o tópico de 
# cada página.
from typing import cast
from torch_geometric.datasets import WikipediaNetwork
from torch_geometric.data import Data

dataset = WikipediaNetwork(root='.', name='chameleon')
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

