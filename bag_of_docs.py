import time
import numpy as np
import pickle

f = open('docword.nytimes.txt','r')
content = f.read()
f.close()

#Bag of Words dos primeiros 3000 documentos

def readDocs(n):
    bag_of_docs = {}
    i = 0
    vocab_size = 0
    with open('docword.nytimes.txt','r') as fdocs:
        n_docs = int(fdocs.readline())
        vocab_size = int(fdocs.readline())
        n_words = fdocs.readline()

        for line in fdocs:
            i += 1
            if i > 3:
                doc, word, freq = line.split()
                doc = int(doc)
                word = int(word)
                freq = int(freq)
                if not doc in bag_of_docs and len(bag_of_docs) >= n:
                    return vocab_size, bag_of_docs
                elif not doc in bag_of_docs:
                    bag_of_docs[doc] = {}
                bag_of_docs[doc][word] = freq
    return vocab_size, bag_of_docs

print('Reading documents...')
start = time.time()
vocab_size, bag_of_docs = readDocs(3000)
end = time.time()
print('Para ler os documentos levou ', (end-start), ' segundos')

data = {
    'vocab_size':vocab_size,
    'bag_of_docs':bag_of_docs
}

f = open('nytimes_docs.pkl','wb')
pickle.dump(data, f)
f.close()

"""
def distance(doc1, doc2):
    words_2 = set(doc2.keys())
    words_1 = set(doc1.keys())

    #words at both documents
    words_both = words_1 & words_2

    words_1_only = words_1 - words_both
    words_2_only = words_2 - words_both

    distance = 0
    for word in words_1_only:
        distance += doc1[word]**2

    for word in words_2_only:
        distance += doc2[word]**2

    for word in words_both:
        distance += (doc2[word] - doc1[word])**2

    return np.sqrt(distance)


def distance_all(bag_of_docs, verbose=False):
    n = len(bag_of_docs)
    distances = np.zeros((n,n))
    for i in xrange(n):
        if verbose:
            print('linha ', i)
        for j in xrange(i+1,n,1):
            distances[i,j] = distance(bag_of_docs[i+1], bag_of_docs[j+1])
            distances[j,i] = distances[i,j]
    return distances

print('computing distances...')
start = time.time()
distances = distance_all(bag_of_docs, verbose=True)
end = time.time()

print('Computar a distancia entre os documentos levou ', (end-start), ' segundos')

f = open('doc_distances_3000.pkl','wb')
pickle.dump(distances, f)
f.close()"""
