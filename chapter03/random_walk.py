# Introdução: Random Walks são sequências de nós produzidas aleatoriamente escolhendo 
# o nó vizinho a cada passo.
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import random

random.seed(0)
# A função erdos_renyi_graph fixa 10 nós e define a probabilidade de criar uma aresta 
# entre dois nós (0.3)
G = nx.erdos_renyi_graph(10, 0.3, seed=1, directed=False)
# Mostra o Random Graph
plt.figure(dpi=300)
plt.axis('off')
nx.draw_networkx(G, 
                 pos=nx.spring_layout(G, seed=0), 
                 node_size=600, 
                 cmap='coolwarm', 
                 font_size=14, 
                 font_color='white')
plt.savefig('img/random_walk_img01.png', format='png', bbox_inches='tight')
# Essa função implementa o Random Walk
# Parâmetros:
#   start - o nó inicial
#   length - o tamanho do walk
def random_walk(start, length):
    walk = [str(start)]
    for i in range(length):
        neighbors = [node for node in G.neighbors(start)]
        # Escolhe aleatoriamente o próximo vizinho
        next_node = np.random.choice(neighbors, 1)[0]
        walk.append(str(next_node))
        start = next_node
    return walk
print(random_walk(0, 10))


