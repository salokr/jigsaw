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


embedding_matrix,tokenizer,dataframe,X_train,labels,Y_train,maxlen,word_index,we_dim,NUM_CLASSES,destination,X_test=load_jigsaw()
input_layer=Input(shape=(maxlen,),dtype='int32')
embedding_layer=Embedding(len(word_index)+1,we_dim,weights=[embedding_matrix],input_length=maxlen,trainable=False)
emb_seq=embedding_layer(input_layer)
lstm_= Bidirectional(LSTM(51,return_sequences=True,dropout=0.3,recurrent_dropout=0.2))
#An implementation of Hierarchical ConvNet

x=Conv1D(128,2,strides=1)(emb_seq)
x=LeakyReLU()(x)
x=BatchNormalization()(x)
x=MaxPool1D(2)(x)
x=Dropout(0.1)(x)
lstm1=lstm_(x)
x=Conv1D(128,3,strides=1)(x)
x=LeakyReLU()(x)
x=BatchNormalization()(x)
x=MaxPool1D(2)(x)
x=Dropout(0.1)(x)
lstm2=lstm_(x)
x=Conv1D(128,4,strides=1)(x)
x=LeakyReLU()(x)
x=BatchNormalization()(x)
x=MaxPool1D(2)(x)
x=Dropout(0.1)(x)
lstm3=lstm_(x)
x=Conv1D(128,4,strides=1)(x)
x=LeakyReLU()(x)
x=BatchNormalization()(x)
x=MaxPool1D(2)(x)
x=Dropout(0.1)(x)
lstm4=lstm_(x)
shared_GlobalMaxPool=GlobalMaxPool1D()
u=keras.layers.concatenate([shared_GlobalMaxPool(lstm1),shared_GlobalMaxPool(lstm2),shared_GlobalMaxPool(lstm3),shared_GlobalMaxPool(lstm4)],axis=-1)
x=Dense(64)(u)
x=LeakyReLU()(x)
x=Dropout(0.2)(x)
x=Dense(NUM_CLASSES,activation='sigmoid')(x)

	

from keras.utils import plot_model
plot_model(model,'Hierarchical_Conv_Net.png')
destination='./Hierarchical_Conv_Net/'
os.mkdir(destination)
model.compile(loss='binary_crossentropy',optimizer='nadam',metrics=['acc'])
checkpoint=ModelCheckpoint(filepath=destination+os.sep+"weights.h5",monitor='val_loss',verbose=0,save_best_only=True,save_weights_only=False,mode='auto',period=1)
earlystopping=EarlyStopping(monitor='val_loss',patience=5)
model.fit(X_train,Y_train,epochs=51,verbose=1,callbacks=[earlystopping,checkpoint],batch_size=500,validation_split=.10)
