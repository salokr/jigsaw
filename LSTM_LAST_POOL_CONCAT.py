import pandas as pd
import numpy as np


import os

import pickle

f=open('./Emb_Mat.mat','r')
embedding_matrix=pickle.load(f)
f.close()

f=open('./New_Tokenizer.tkn','r')
tokenizer=pickle.load(f)
f.close()

train_path='./train.csv'

dataframe=pd.read_csv(train_path)

X_train=dataframe['comment_text'].fillna('<UNK>').values

labels=['toxic','severe_toxic','obscene','threat','insult','identity_hate']

Y_train=dataframe[labels].values

maxlen=100

word_index=tokenizer.word_index

X_train=tokenizer.texts_to_sequences(X_train)


from keras.preprocessing.sequence import pad_sequences
X_train=pad_sequences(X_train,maxlen=maxlen)

we_dim=300

NUM_CLASSES=Y_train.shape[1]

from keras.layers import *
from keras.models import *
from keras.layers.recurrent import *
from keras.layers.pooling import *
from keras.layers.embeddings import *
from keras.layers.wrappers import Bidirectional

import shutil

destination='./Bi_LSTM_CONCAT_POOL'
if os.path.isdir(destination):
	shutil.rmtree(destination, ignore_errors=True)

os.mkdir(destination)

from keras.callbacks import *
#checkpoint=ModelCheckpoint(filepath=destination+os.sep+"weights.FL.h5",monitor='val_loss',verbose=0,save_best_only=True,save_weights_only=False,mode='auto',period=1)
#earlystopping=EarlyStopping(monitor='val_loss',patience=3)
from keras.layers import LeakyReLU




input_layer=Input(shape=(maxlen,),dtype='int32')
embedding_layer=Embedding(len(word_index)+1,emb_dim,weights=[embedding_matrix],input_length=maxlen,trainable=False)
emb_seq=embedding_layer(input_layer)

#######################FIRST_LAYER_LSTM's#################################################
x=Bidirectional(LSTM(51,return_sequences=True,dropout=0.2,recurrent_dropout=0.2))(emb_seq)# # #
x=LeakyReLU()(x)

last_state=Lambda(lambda x: x[:,-1,:])(BatchNormalization()(x))

y=GlobalMaxPool1D()(x)
y=BatchNormalization()(y)#<- Contains Most Relevant Features

import keras
merged=keras.layers.concatenate([y,last_state], axis=-1)

#Concatenate Summary and most relevant features
#import keras
#smerged=keras.layers.concatenate([x,batch_normalized_last_state], axis=-1)

#y=Dense(64)([x,batch_normalized_last_state])
y=Dense(64)(merged)
y=LeakyReLU()(y)
y=Dropout(0.2)(y)
y=Dense(NUM_CLASSES,activation='sigmoid')(y)

model=Model(inputs=input_layer,outputs=y)


checkpoint=ModelCheckpoint(filepath=destination+os.sep+"weights.FL.h5",monitor='val_loss',verbose=0,save_best_only=True,save_weights_only=False,mode='auto',period=1)
earlystopping=EarlyStopping(monitor='val_loss',patience=5)
#model=getModel(word_index,embedding_matrix,maxlen,6)


model.compile(loss='binary_crossentropy',optimizer='nadam',metrics=['acc'])
model.fit(X_train,Y_train,epochs=51,verbose=1,callbacks=[earlystopping,checkpoint],batch_size=500,validation_split=.10)





'''
#X OUTPUT_SHAPE : (?,?,102), extract (?,-1,?, that's it !!! )
#pool_rnn=Lambda(lambda x: K.max(x,axis=1))
#last_state = x[:,-1,:]
#batch_normalized_last_state=BatchNormalization()(last_state)#<-Contains Summary of the network
'''
