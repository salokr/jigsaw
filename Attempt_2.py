#Score :.051
import pandas as pd
import numpy as np


import os

import pickle

f=open('./Emb_Mat.mat','r')
embedding_matrix=pickle.load(f)
f.close()

f=open('./tokenizer.tkn','r')
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

destination='./Bi_LSTM_BASELINE'
if os.path.isdir(destination):
	shutil.rmtree(destination, ignore_errors=True)

os.mkdir(destination)

from keras.callbacks import *
checkpoint=ModelCheckpoint(filepath=destination+os.sep+"weights.{epoch:03d}-{loss:.2f}-{val_loss:.2f}.h5",monitor='val_loss',verbose=0,save_best_only=True,save_weights_only=False,mode='auto',period=1)
earlystopping=EarlyStopping(monitor='val_loss',patience=3)
from keras.layers import LeakyReLU

def getModel(word_index,embedding_matrix,maxlen,NUM_CLASSES):
	model=Sequential()
	model.add(Embedding(len(word_index)+1,300,weights=[embedding_matrix],input_length=maxlen))
	model.add(Bidirectional(LSTM(51,return_sequences=True,dropout=0.3,recurrent_dropout=0.2)))
	model.add(LeakyReLU())
	model.add(GlobalMaxPool1D())
	model.add(BatchNormalization())
	model.add(Dense(64))
	model.add(LeakyReLU())
	model.add(Dropout(0.2))
	model.add(Dense(NUM_CLASSES,activation='sigmoid'))
	model.compile(loss='binary_crossentropy',optimizer='nadam',metrics=['acc'])
	return model


checkpoint=ModelCheckpoint(filepath=destination+os.sep+"weights.{epoch:03d}-{loss:.2f}-{val_loss:.2f}.h5",monitor='val_loss',verbose=0,save_best_only=True,save_weights_only=False,mode='auto',period=1)
earlystopping=EarlyStopping(monitor='val_loss',patience=5)
history=getModel().fit(X_train,Y_train,epochs=51,verbose=1,callbacks=[earlystopping,checkpoint],batch_size=32,validation_split=.10)
#Load the best model

#
#performCrossValidation(X_train,Y_train,epochs=1,batch_size=1000,model_name='./Test')
'''
from keras.models import load_model
model=load_model('./LSTM_BASELINE/weights.000-0.16-0.14.h5')


test_path='./test.csv'
test_df=pd.read_csv(test_path)
X_test=test_df['comment_text'].fillna('<UNK>').values
X_test=tokenizer.texts_to_sequences(X_test)
X_test=pad_sequences(X_test,maxlen=maxlen)


probabilities=model.predict(X_test)

submission_df = pd.DataFrame(columns=['id'] + labels)
submission_df['id'] = dataframe['id'].values 
submission_df[labels] = probabilities
submission_df.to_csv("./cnn_multifilter_submission.csv", index=False)
'''
"""

#Even Worse, ensemble of lstm and cnn

cnn=pd.read_csv('cnn_multifilter_submission.csv')
lstm=pd.read_csv('lstm_submission.csv')
p_res=lstm.copy()


p_res[labels]=(lstm[labels]+cnn[labels])/2


p_res.to_csv('CNN_LSTM_Ensemble.csv',index=False)
"""


'''
max of two dataframes

cnn1=pd.read_csv('cnn_multifilter_submission.csv')
cnn2=pd.read_csv('cnn_sensible_submission.csv')
x=cnn1.where(cnn1>cnn2,cnn1)
x.to_csv('CNN1_CNN2_Ensemble_max.csv',index=False)
'''