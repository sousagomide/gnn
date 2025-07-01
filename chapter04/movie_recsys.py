# Introdução: Esse exemplo codifica os filmes em vez de palavras, permitindo verificar a similaridade do filme 
# com base em uma entrada. Como um sistema de recomendação. Deseja criar um Random Walk Biased do filmes, porém 
# isso irá requerer um dataset de grafos onde os filmes similares são conectados com cada um outro.
# Será implementado uma abordagem simples e intuitiva: filmes são curtidos pelos mesmos usuários que estão conectados. 
# Será usado um grafo para aprender os filmes transformados usando Node2Vec.
import pandas as pd
import networkx as nx
from collections import defaultdict
from node2vec import Node2Vec
from zipfile import ZipFile

def load_dataset():
    zip_path = 'dataset/ml-100k.zip'
    with ZipFile(zip_path, 'r') as zfile:
        zfile.extractall('.')

# A função recommned pega um filme de entrada, converte o título em um Movie ID que é 
# usado para consultar o modelo
def recommend(movie):
    movie_id = str(movies[movies.title == movie].movie_id.values[0])
    for id in model.wv.most_similar(movie_id)[:5]:
        title = movies[movies.movie_id == int(id[0])].title.values[0]
        print(f'{title}: {id[1]:.2f}')

# load_dataset()
ratings = pd.read_csv('ml-100k/u.data', sep='\t', names=['user_id', 'movie_id', 'rating', 'unix_timestamp'])
# print(ratings)
movies = pd.read_csv('ml-100k/u.item', sep='|', usecols=range(2), names=['movie_id', 'title'], encoding='latin-1')
# print(movies)
# Listar apenas os filmes que receberam muito likes. Manter apenas os ratings mantendo 
# o score de 4 e 5
ratings = ratings[ratings.rating >= 4]
# print(ratings)
# Contar toda vez que dois filmes receberam like do mesmo usuário. Essa informação será 
# usada para construir arestas do nosso grafo
pairs = defaultdict(int)
for group in ratings.groupby('user_id'):
    user_movies = list(group[1]['movie_id'])
    for i in range(len(user_movies)):
        for j in range(i+1, len(user_movies)):
            pairs[(user_movies[i], user_movies[j])] += 1
# Criando o grafo
G = nx.Graph()
for pair in pairs:
    movie1, movie2 = pair
    score = pairs[pair]
    # Se o score for maior que 20 é feita a conexão entre os filmes
    if score >= 20:
        G.add_edge(movie1, movie2, weight=score)
# Realizando o treinamento do grafo usando Node2Vec
node2vec = Node2Vec(G, dimensions=64, walk_length=20, num_walks=200, p=2, q=1, workers=1)
# Treina o modelo com um Random Walk Biased com uma janela de 10 (5 nós antes e 5 nós depois)
model = node2vec.fit(window=10, min_count=1, batch_words=4)
recommend('Star Wars (1977)')