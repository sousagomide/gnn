# A regressão refere-se a predição de valores contínuos. Esse código é 
# escrito para predizer valor contínuos ao invés de variáveis 
# categóricas. Para isso será utilizado o dataset Wikipedia Network
from torch_geometric.datasets import WikipediaNetwork
import torch_geometric.transforms as T

dataset = WikipediaNetwork(root='.', name='chameleon', transform=T.RandomNodeSplit(num_val=200, num_test=500))
data = dataset[0]

print(f'Dataset: {dataset}')
print('-' * 30)
print(f'Number of graphs: {len(dataset)}')
print(f'Number of nodes: {data.x.shape[0]}')
print(f'Number of unique features: {dataset.num_features}')
print(f'Number of classes: {dataset.num_classes}')