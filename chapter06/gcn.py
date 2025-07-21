from numpy import number
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import degree
from torch_geometric.data import Data
from collections import Counter
from typing import cast
import matplotlib.pyplot as plt


dataset = Planetoid(root='.', name='Cora')
data = cast(Data, dataset[0])

# Calcula o número de vizinhos de cada nó do grafo
degrees = degree(data.edge_index[0]).numpy()
# Conta o número de nós para cada grau
numbers = Counter(degrees)
# Plota o resultado
fig, ax = plt.subplots()
ax.set_xlabel('Node degree')
ax.set_ylabel('Number of nodes')
plt.bar(numbers.keys(), numbers.values())
plt.show()