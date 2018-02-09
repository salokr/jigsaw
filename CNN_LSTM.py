from Load_Jigsaw import *
import numpy as np
import os
import pandas as pd
from keras.layers import *
from keras.models import *
from keras.layers.recurrent import *
from keras.layers.pooling import *
from keras.layers.convolutional import *
from keras.layers.embeddings import *
from keras.callbacks import *
from keras.utils import *
import keras


embedding_matrix,tokenizer,dataframe,X_train,labels,Y_train,maxlen,word_index,we_dim,NUM_CLASSES,destination,checkpoint,earlystopping=load_jigsaw()
input_layer=Input(shape=(maxlen,),dtype='int32')
embedding_layer=Embedding(len(word_index)+1,we_dim,weights=[embedding_matrix],input_length=maxlen,trainable=False)
emb_seq=embedding_layer(input_layer)
#First The CNN's
submodels=[]
filter_size=[2,3,4]
lstm_= Bidirectional(LSTM(51,return_sequences=True,dropout=0.3,recurrent_dropout=0.2))
for f in filter_size:
	x=Conv1D(128,f,strides=1)(emb_seq)
	x=LeakyReLU()(x)
	x=BatchNormalization()(x)
	x=MaxPooling1D(2)(x)
	x=Dropout(0.1)(x)
	x=Conv1D(64,f)(x)
	x=LeakyReLU()(x)
	x=MaxPooling1D(3)(x)
	x=BatchNormalization()(x)
	x=Dropout(0.1)(x)
	x=lstm_(x)
	x=LeakyReLU()(x)
	x=BatchNormalization()(x)
	x=GlobalMaxPooling1D()(x)
	model=Model(inputs=input_layer,outputs=x)
	submodels.append(model)


model=Sequential()
model.add(Merge(submodels,mode='concat'))
model.add(Dense(64))
model.add(LeakyReLU())
model.add(Dense(NUM_CLASSES,activation='sigmoid'))
model.compile(loss='binary_crossentropy',optimizer='nadam',metrics=['acc'])
