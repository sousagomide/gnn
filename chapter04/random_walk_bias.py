# Introdução: Nesse exemplo será apresentado o Random Walk com Bias e como funciona 
# os parâmetros p e q. Ao selecionar o próximo nó, haverá uma probabilidade de: 
# visitar o nó anterior; visitar um nó imediato; ou selecionar um outro nó qualquer.
import networkx as nx
import random
import numpy as np

# A função next_node seleciona qual o próximo nó a partir dos parâmetros p e q
def next_node(previous, current, p, q):
    neighbors = list(G.neighbors(current))
    alphas = []
    # Para cada vizinho será calculado o valor de alpha: 1/p se o vizinho é o nó 
    # anterior; 1 se o vizinho está conectado ao no anterior; e do contrário 1/q
    for neighbor in neighbors:
        if neighbor == previous:
            alpha = 1/p
        elif G.has_edge(neighbor, previous):
            alpha = 1
        else:
            alpha = 1/q
        alphas.append(alpha)
    # Normaliza os valores para criar as probabilidades
    probs = [alpha / sum(alphas) for alpha in alphas]
    # Seleciona o próximo nó baseado na transição de probabilidades
    next = np.random.choice(neighbors, size=1, p=probs)[0]
    return next

# A função random_walk é semelhante a criada no capítulo 03, só que agora o nó 
# escolhido será definido a partir da função next_node
def random_walk(start, length, p, q):
    walk = [start]
    for i in range(length):
        current = walk[-1]
        previous = walk[-2] if len(walk) > 1 else None
        next = next_node(previous, current, p, q)
        walk.append(next)
    return [str(x) for x in walk]

random.seed(0)
G = nx.erdos_renyi_graph(10, 0.3, seed=1, directed=False)
# Testando a função random_walk com: length=5, p=1 e q=1
print(random_walk(0, 8, p=1, q=1))
# Testando a função random_walk com: length=5, p=1 e q=10
print(random_walk(0, 8, p=1, q=10))
# Testando a função random_walk com: length=5, p=10 e q=1
print(random_walk(0, 8, p=10, q=1))