import pickle
import numpy as np
import time

f = open('nytimes_docs.pkl','rb')
data = pickle.load(f)
f.close()
vocab_size = data['vocab_size']
bag_of_docs = data['bag_of_docs']

def generate_matrix(n,d):
    matrix = np.zeros((n,d))
    for i in xrange(n):
        for j in xrange(d):
            matrix[i,j] = np.random.normal(0, 1/np.sqrt(d))
    return matrix

def generate_matrix_fast(n,d):
    return np.random.normal(0,1/np.sqrt(d),(n,d))

d = vocab_size
n = 4
iterations = 1

matrizes = []
time_creation = []

print('criando matrizes, metodo aula')

for i in xrange(iterations):
    start = time.time()
    print('iteracao ', i)
    matrizes.append(generate_matrix_fast(n,d))
    end = time.time()
    time_creation.append((end-start))

time_creations = np.array(time_creation)

print('tempo de criacao de matrizes')
print('min ', time_creations.min(), ' max ', time_creations.max(), ' mean ', time_creations.mean())

def projection(bag_of_words, random_base):
    n = random_base.shape[0]
    d = random_base.shape[1]
    #bag of
    used_words = list(bag_of_words.keys())
    proj_doc = np.sum(random_base[:,used_words],axis=1)

