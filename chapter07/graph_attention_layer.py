# O objetivo do Graph Attention Layer é converter uma estrutura baseada 
# em grafos para o formato matricial
# O Graph Attention é um bloco essêncial para desenvolver uma GNN
import numpy as np

# O grafo precisa apresentar duas partes: a matriz de adjacência e o nó 
# de características
# A matriz de adjacência
np.random.seed(0)
A = np.array([
    [1, 1, 1, 1],
    [1, 1, 0, 0],
    [1, 0, 1, 1],
    [1, 0, 1, 1]
])
# Para o nó de características é utilizado a função np.random.uniform()
X = np.random.uniform(-1, 1, (4, 4))
# Agora é gerado o peso da matriz. A matriz W é projetada para dimensões: 
# nb de dimensões ocultas, nb de nós
# No caso o nb de nós = 4 pois temos quatro nós
# O nb de dimensões ocultas é arbitrário. No caso foi escolhido 2
W = np.random.uniform(-1, 1, (2, 4))
# A matriz a seguir concatena os vetores ocultos para produzir um valor 
# único. O tamanho precisa ser (1, dim_h*2)
W_att = np.random.uniform(-1, 1, (1, 4))
# Agora deseja-se concatenar os vetores ocultos da fonte com os nós de 
# destino
connections = np.where(A > 0)
np.concatenate([(X @ W.T)[connections[0]], (X @ W.T)[connections[1]]], axis=1)
# Apica-se a transformação linear
a = W_att @ np.concatenate([(X @ W.T) [connections[0]], (X @ W.T) [connections[1]]], axis=1).T
##################################################
##### O segundo passo é aplicar a função Leaky ReLU
##################################################
def leaky_relu(x, alpha=0.2):
    return np.maximum(alpha*x, x)
e = leaky_relu(a)
# Organiza a matrix
E = np.zeros(A.shape)
E[connections[0], connections[1]] = e[0]
# Utiliza a função softmax para normalizar os scores
def softmax2D(x, axis):
    e = np.exp(x - np.expand_dims(np.max(x, axis=axis), axis))
    sum = np.expand_dims(np.sum(e, axis=axis), axis)
    return e/sum
W_alpha = softmax2D(E, 1)
# W_alpha provê os pesos para todas possíveis conexões na rede. Ele é 
# usado para calcular nossa matriz de transformação
H = A.T @ W_alpha @ X @ W.T
print(H)