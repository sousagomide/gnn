# Nesse código é computado o GAT do dataset CiteSeer para classificação de nó.
# Segundo autores o GAT é significativamente melhor que o GCN
# Também é constatado que nós com poucos vizinho são mais difíceis de ser 
# classificado
from collections import Counter
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import degree
from gat_pytorch_geometric import GAT
import matplotlib.pyplot as plt


dataset = Planetoid(root='.', name='CiteSeer')
data = dataset[0]
print(data)

# degrees = degree(data.edge_index[0]).numpy()
# numbers = Counter(degrees)
# fig, ax = plt.subplots()
# ax.set_xlabel('Node degree')
# ax.set_ylabel('Number of nodes')
# plt.bar(numbers.keys(), numbers.values())
# plt.show()

gat = GAT(dataset.num_features, 16, dataset.num_classes)
gat.fit(data, epochs=100)
acc = gat.test(data)
print(f'GAT test accuracy: {acc*100:.2f}%')
