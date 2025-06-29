import numpy as np
from gensim.models.word2vec import Word2Vec

# Montar o skip-gram com duas palavras
CONTEXT_SIZE = 2
text = """
    Lorem ipsum dolor sit amet, consectetur adipiscing elit. Nunc eu sem scelerisque, dictum eros 
    aliquam, accumsan quam. Pellentesque tempus, lorem ut semper fermentum, ante turpis accumsan ex, 
    sit amet ultricies tortor erat quis nulla. Nunc consectetur ligula sit amet purus porttitor, vel 
    tempus tortor scelerisque. Vestibulum ante ipsum primis in faucibus orci luctus et ultrices 
    posuere cubilia curae; Quisque suscipit ligula nec faucibus accumsan. Duis vulputate massa sit 
    amet viverra hendrerit. Integer maximus quis sapien id convallis. Donec elementum placerat ex 
    laoreet gravida. Praesent quis enim facilisis, bibendum est nec, pharetra ex. Etiam pharetra 
    congue justo, eget imperdiet diam varius non. Mauris dolor lectus, interdum in laoreet quis, 
    faucibus vitae velit. Donec lacinia dui eget maximus cursus. Class aptent taciti sociosqu ad 
    litora torquent per conubia nostra, per inceptos himenaeos. Vivamus tincidunt velit eget nisi 
    ornare convallis. Pellentesque habitant morbi tristique senectus et netus et malesuada fames ac 
    turpis egestas. Donec tristique ultrices tortor at accumsan.
""".split()
skipgrams = []
for i in range(CONTEXT_SIZE, len(text) - CONTEXT_SIZE):
    array = [text[j] for j in np.arange(i-CONTEXT_SIZE, i+CONTEXT_SIZE+1) if j != i]
    skipgrams.append((text[i], array))
# Remover as palavras duplicadas
vocab = set(text)
VOCAB_SIZE = len(vocab)
# print(f'Length of text = {len(text)}')
# print(f'Length of vocabulary = {VOCAB_SIZE}')
# Definir: N -> a dimensionalidade do Word Vector. Padrão (100 a 1000)
# Criar o modelo: uma camada de projeção; uma camada completamente conectada
# Para criar o modelo será usada a biblioteca gensim
# skip-gram: sg = 1
model = Word2Vec([text], sg=1, vector_size=10, min_count=0, window=2, workers=2, seed=0)
print(f'Shape of W_embed: {model.wv.vectors.shape}')
# Treinar o modelo em 10 épocas
model.train([text], total_examples=model.corpus_count, epochs=10)
print('Word embedding = ')
print(model.wv[0])
# Conclusão: Essa abordagem funciona bem pequenos vocabularios. O custo computacional da função softmax
# para milhões de palavras é oneroso na maioria dos casos. Word2Vec e DeepWalk implementam a técnica
# chamada H-Softmax. Essa técnica usa uma árvore binária para conectar as palavras. O H-Softmax pode ser
# ativado no gensim usando hs = 1
