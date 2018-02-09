#Score :.105

import pandas as pd
import numpy as np

import pickle

from keras import backend as K



import os

from Losses import *

train_path='./train.csv'

dataframe=pd.read_csv(train_path)

X_train=dataframe['comment_text'].fillna('<UNK>').values

labels=['toxic','severe_toxic','obscene','threat','insult','identity_hate']

Y_train=dataframe[labels].values

maxlen=50

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
'''
tokenizer=Tokenizer()
tokenizer.fit_on_texts(X_train)
word_index=tokenizer.word_index

from keras.preprocessing.sequence import pad_sequences
'''

f=open('./Emb_Mat.mat','r')
embedding_matrix=pickle.load(f)
f.close()

f=open('./tokenizer.tkn','r')
tokenizer=pickle.load(f)
f.close()


word_index=tokenizer.word_index


X_train=tokenizer.texts_to_sequences(X_train)

X_train=pad_sequences(X_train,maxlen=maxlen)
we_dim=300
'''
embedding_path='./../glove.840B.300d.txt'

embedding_dict={}

f=open(embedding_path,'r')

for line in f:
	fields=line.split()
	word=fields[0]
	w_e=np.asarray(fields[1:],dtype='float32')
	embedding_dict[word]=w_e

f.close()


embedding_matrix=np.zeros(shape=(len(word_index)+1,we_dim))


#Build the weight matrix for embeddings for initialization process :D
for word,index in word_index.items():
	vector=embedding_dict.get(word)
	if vector is not None:
		embedding_matrix[index]=vector

'''



from keras.models import *
from keras.layers import Dense,Dropout,Merge
from keras.layers.convolutional import *
from keras.layers.pooling import *
from keras.layers.embeddings import *
from keras.layers import LeakyReLU
submodels=[]
filter_size=[3,5]
for f_size in filter_size:
	sm=Sequential()
	sm.add(Embedding(len(word_index)+1,300,weights=[embedding_matrix],input_length=maxlen,trainable=False))
	sm.add(Conv1D(128,f_size,activation='relu',strides=1,padding='valid'))
	sm.add(MaxPooling1D(2))
	sm.add(Dropout(0.1))
	sm.add(Conv1D(64,2,strides=1,padding='valid'))
	sm.add(LeakyReLU())
	sm.add(Conv1D(32,2,padding='valid'))
	sm.add(LeakyReLU())
	sm.add(MaxPooling1D(2))
	sm.add(Dropout(0.1))
	sm.add(Conv1D(16,2,padding='valid'))
	sm.add(LeakyReLU())
	sm.add(GlobalMaxPooling1D())
	sm.add(Dropout(0.05))
	submodels.append(sm)



NUM_CLASSES=Y_train.shape[1]

cnn=Sequential()
cnn.add(Merge(submodels,mode='concat'))
cnn.add(Dense(32,activation='relu'))
cnn.add(Dense(NUM_CLASSES,activation='sigmoid'))
#cnn.compile(loss='binary_crossentropy',optimizer='nadam',metrics=['acc'])
cnn.compile(loss=[focal_loss(alpha=.5,gamma=1)],optimizer='nadam',metrics=['acc'])

from keras.callbacks import *
earlystopping=EarlyStopping(monitor='val_loss',patience=20)
os.mkdir('CNN_BASELINE_FL')
checkpoint=ModelCheckpoint(filepath='./CNN_BASELINE_FL'+os.sep+"weights.{epoch:03d}-{loss:.2f}-{val_loss:.2f}.h5",monitor='val_loss',verbose=0,save_best_only=True,save_weights_only=False,mode='auto',period=1)
history=cnn.fit([X_train,X_train],Y_train,epochs=151,verbose=1,callbacks=[earlystopping,checkpoint],batch_size=64,validation_split=.10)

'''
test_path='./test.csv'
dataframe=pd.read_csv(test_path)
X_test=dataframe['comment_text'].fillna('<UNK>').values


X_test=tokenizer.texts_to_sequences(X_test)
X_test=pad_sequences(X_test,maxlen=maxlen)
probabilities=cnn.predict([X_test,X_test,X_test])

submission_df = pd.DataFrame(columns=['id'] + labels)
submission_df['id'] = dataframe['id'].values 
submission_df[labels] = probabilities 
submission_df.to_csv("./cnn_multifilter_submission.csv", index=False)
'''