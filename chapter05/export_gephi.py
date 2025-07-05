# Introdução: Código usado para exportar o dataset WikipediaNetwork para 
# o formato graphml

from torch_geometric.datasets import WikipediaNetwork
from torch_geometric.utils import to_networkx
import networkx as nx

# Carregue o dataset
dataset = WikipediaNetwork(root='.', name='chameleon')
data = dataset[0]

# Converta para NetworkX (grafo não-direcionado, pois Gephi funciona melhor assim)
G = to_networkx(data, to_undirected=True)
nx.write_graphml(G, 'chameleon/wikipedia_network.graphml')
