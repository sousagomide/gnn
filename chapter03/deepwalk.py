# Introdução: O algoritmo DeepWalk é uma combinação das técnicas Word2Vec e 
# Random Walk. No DeepWalk quando o nó está mais próximo de outro há uma alta 
# pontuação de similaridade. Do contrário obtém-se uma baixa pontuação.
# Nesse exemplo será usado o Zachary Karate Club (ZKC) que representa a relação 
# dos estudantes de uma escola de Karatê. O ZKC é tipo de rede social onde 
# cada nó é um membro e os membros que interagem fora do clube estão conectados.
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from gensim.models.word2vec import Word2Vec
from sklearn.manifold import TSNE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Função explicada no código random_walk.py
def random_walk(start, length):
    walk = [str(start)]
    for i in range(length):
        neighbors = [node for node in G.neighbors(start)]
        # Escolhe aleatoriamente o próximo vizinho
        next_node = np.random.choice(neighbors, 1)[0]
        walk.append(str(next_node))
        start = next_node
    return walk

G = nx.karate_club_graph()
# Convertendo labels em valores numéricos. Exemplo: Mr. Hi = 0, Officer = 1
labels = []
for node in G.nodes:
    label = G.nodes[node]['club']
    labels.append(1 if label == 'Officer' else 0)
# Plotar o grafo com os novos rótulos
plt.figure(figsize=(12, 12), dpi=300)
plt.axis('off')
nx.draw_networkx(G,
                 pos=nx.spring_layout(G, seed=0),
                 node_color=labels,
                 node_size=800,
                 cmap='coolwarm',
                 font_size=14,
                 font_color='white')
plt.savefig('img/deepwalk_img01.png', format='png', bbox_inches='tight')
# Gerar o dataset de random walks. Serão gerados 80 random walks de tamanho 
# 10 para cada nó do grafo
walks = []
for node in G.nodes:
    for _ in range(80):
        walks.append(random_walk(node, 10))
print(f'Length walks: {len(walks)}')
print(walks[0])
# Implementação do Word2Vec. Será usado o modelo skip-gram previamente visto 
# no código skip_gram.py
# Hierarchical softmax: hs=1
model = Word2Vec(walks, hs=1, sg=1, vector_size=100, window=10, workers=2, seed=0)
# Treinamento do modelo
model.train(walks, total_examples=model.corpus_count, epochs=30, report_delay=1)
# Exemplo de aplicação 01: Encontrar os nós mais similares dado um nó
print('Nodes that are the most similar to node 0: ')
for similarity in model.wv.most_similar(positive=['0']):
    print(f'{similarity}')
# Exemplo de aplicação 02: Calcular a pontuação de similaridade entre dois nós
print(f'Similarity between node 0 and 4: {model.wv.similarity('0', '4')}')
# Plot dos resultados usando t-distributed stochastic neighbor embedding (t-SNE)
# Cria dois vetores. Um para armazenar as palavras transformadas e outro para 
# armazenar os rótulos
nodes_wv = np.array([model.wv.get_vector(str(i)) for i in range(len(model.wv))])
labels = np.array(labels)
# Treinamento do modelo t-SNE com duas dimensões (n_components=2)
tsne = TSNE(n_components=2, learning_rate='auto', init='pca', random_state=0).fit_transform(nodes_wv)
# Plot do TSNE
plt.figure(figsize=(6, 6), dpi=300)
plt.scatter(tsne[:, 0], tsne[:, 1], s=100, c=labels, cmap='coolwarm')
plt.savefig('img/deepwalk_img02.png', format='png', bbox_inches='tight')
# Implementação de um classificador e treinamento dos nós
# Separando a base para dados de treinamento e teste.
nodes = np.array(G.nodes())
train_mask, test_mask = train_test_split(nodes, test_size=0.5, random_state=42)
print(f'Base de treinamento: {train_mask}')
print(f'Base de teste: {test_mask}')
# Aplicando o Random Forest
clf = RandomForestClassifier(random_state=0)
clf.fit(nodes_wv[train_mask], labels[train_mask])
# Avaliação do modelo treinado baseado nos dados de teste (Accuracy Score)
y_pred = clf.predict(nodes_wv[test_mask])
print(f'Acurácia: {accuracy_score(y_pred, labels[test_mask])}')

