# Introdução: A implementação do Node2Vec é muito similar a implementação do DeepWalk.
import networkx as nx
import pandas as pd
import numpy as np
from gensim.models.word2vec import Word2Vec
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

def next_node(previous, current, p, q):
    neighbors = list(G.neighbors(current))
    alphas = []
    for neighbor in neighbors:
        if neighbor == previous:
            alpha = 1/p
        elif G.has_edge(neighbor, previous):
            alpha = 1
        else:
            alpha = 1/q
        alphas.append(alpha)
    probs = [alpha / sum(alphas) for alpha in alphas]
    next = np.random.choice(neighbors, size=1, p=probs)[0]
    return next

def random_walk(start, length, p, q):
    walk = [start]
    for i in range(length):
        current = walk[-1]
        previous = walk[-2] if len(walk) > 1 else None
        next = next_node(previous, current, p, q)
        walk.append(next)
    return [str(x) for x in walk]

results = []
for p in range(1, 8):
    for q in range(1, 8):
        G = nx.karate_club_graph()
        labels = []
        for node in G.nodes:
            label = G.nodes[node]['club']
            labels.append(1 if label == 'Officer' else 0)
        walks = []
        for node in G.nodes:
            for _ in range(80):
                walks.append(random_walk(node, 10, p=p, q=q))
        node2vec = Word2Vec(walks, hs=1, sg=1, vector_size=100, window=10, workers=2, min_count=1, seed=0)
        node2vec.train(walks, total_examples=node2vec.corpus_count, epochs=30, report_delay=1)
        nodes = np.array(G.nodes())
        train_mask, test_mask = train_test_split(nodes, test_size=0.5, random_state=42)
        train_mask_str = [str(x) for x in train_mask]
        test_mask_str = [str(x) for x in test_mask]
        labels = np.array(labels)
        clf = RandomForestClassifier(random_state=0)
        clf.fit(node2vec.wv[train_mask_str], labels[train_mask])
        y_pred = clf.predict(node2vec.wv[test_mask_str])
        acc = accuracy_score(y_pred, labels[test_mask])
        results.append({'p': p, 'q': q, 'accuracy': acc})
df_results = pd.DataFrame(results)
tabela_accuracy = df_results.pivot(index='p', columns='q', values='accuracy')
print('Tabela de resultados:')
print(tabela_accuracy)