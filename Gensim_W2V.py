import gensim
from gensim.models.keyedvectors import KeyedVectors
import numpy as np


#syn0 contains the word_embeddings
#reduction could be due to the fact that we may have removed the less frequent occuring words
def laodNCreateMatrix(emb_path,word_index,emb_dim):
	word_vectors = KeyedVectors.load_word2vec_format(emb_path, binary=False)
	rows=len(word_index)+1
	embedding_dict={}
	emb_word_index=dict([(k, v.index) for k, v in word_vectors.vocab.items()])
	for i in range(len(emb_word_index)):
		word=word_vectors.index2word[i]
		embedding_dict[word]=word_vectors.syn0[i]
	allembs=np.stack(embedding_dict.values())
	emb_mean,emb_std=allembs.mean(),allembs.std()
	embedding_matrix=np.random.normal(emb_mean,emb_std,(rows,emb_dim))
	#ii=0
	for word,index in word_index.items():
		#ii=ii+1
		try:
			gensim_index=emb_word_index[word]
		except:
			gensim_index=None
		if gensim_index is None:
			continue
		embedding_matrix[index]=word_vectors.syn0[gensim_index]
	return embedding_matrix




