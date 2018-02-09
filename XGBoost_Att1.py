import pandas as pd
import numpy as np


import os

import pickle

f=open('./Emb_Mat.mat','r')
embedding_matrix=pickle.load(f)
f.close()

f=open('./Tokenizer.tkn','r')
tokenizer=pickle.load(f)
f.close()

train_path='./train.csv'

dataframe=pd.read_csv(train_path)

X_train=dataframe['comment_text'].fillna('<UNKNOWN>').values

labels=['toxic','severe_toxic','obscene','threat','insult','identity_hate']

Y_train=dataframe[labels].values

maxlen=100

word_index=tokenizer.word_index



from gensim.models import KeyedVectors

glove=KeyedVectors.load_word2vec_format('./Glove840B.txt')


def getVectorForWord(word,model,we_dim):
	x=np.zeros(we_dim)
	try:
		x=model.wv[word]
	except:
		print('Not Found '+word)
	return x


def sen2Vec(sentence,maxlen,we,we_dim):
	x=np.zeros(we_dim*maxlen)
	end=0
	i=-1
	for words in sentence.split():
		words=words.lower()
		i=end
		end=i+we_dim
		x[i:end]=getVectorForWord(words,we,300)
	return x

def doc2Vec(doc,maxlen,we,we_dim):
	rows=doc.shape[0]
	#cols=maxlen
	x=np.zeros(shape=(rows,maxlen*we_dim))
	for i in range(rows):
		x[i]=sen2Vec(doc[i],maxlen,we,we_dim)

