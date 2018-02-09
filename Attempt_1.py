#Score :.055

import pandas as pd
import numpy as np
from Losses import *
import os
from New_Utils import *
#clean=dataframe[(dataframe['toxic']!=0) | (dataframe['severe_toxic']!=0) | (dataframe['obscene']!=0) | (dataframe['threat']!=0) | (dataframe['insult']!=0) | (dataframe['identity_hate']!=0) ]
datacolumns='comment_text'
maxlen=50
na='<UNK>'
train_path='./train.csv'

labels=['toxic','severe_toxic','obscene','threat','insult','identity_hate']
X_train,Y_train=loadCompleteCSV(train_path,datacolumns,labels,na,maxlen,'./tokenizer.tkn',False)


#Done : New_Utils##########################################################################################################################################################################################
tokenizer=readPickle('./tokenizer.tkn')
word_index=tokenizer.word_index
embedding_path='./../glove.840B.300d.txt'
embedding_matrix=load_create_embedding_matrix(None,None,None,None,'./Emb_Mat.mat')
#Done : New_Utils#############################################################################################
from keras.models import *
from keras.layers import *
from keras.layers.convolutional import *
from keras.layers.pooling import *
from keras.layers.embeddings import *

def getModel(word_index,embedding_matrix,maxlen,trainable,NUM_CLASSES):
	submodels=[]
	filter_size=[2,3,4]
	for f_size in filter_size:
		sm=Sequential()
		sm.add(Embedding(len(word_index)+1,300,weights=[embedding_matrix],input_length=maxlen,trainable=trainable))
		sm.add(Conv1D(128,f_size,strides=1,padding='valid'))
		sm.add(LeakyReLU())
		sm.add(MaxPooling1D(2))
		sm.add(Dropout(0.1))
		sm.add(Conv1D(128,5))
		sm.add(LeakyReLU())
		sm.add(BatchNormalization())
		sm.add(Conv1D(64,3))
		sm.add(MaxPooling1D(2))
		sm.add(Dropout(0.1))
		sm.add(Conv1D(64,f_size,strides=1,padding='valid'))
		sm.add(LeakyReLU())
		sm.add(GlobalMaxPooling1D())
		sm.add(BatchNormalization())
		sm.add(Dropout(0.1))
		submodels.append(sm)
	#NUM_CLASSES=Y_train.shape[1]
	cnn=Sequential()
	cnn.add(Merge(submodels,mode='concat'))
	cnn.add(Dense(32))
	cnn.add(LeakyReLU())
	cnn.add(Dense(NUM_CLASSES,activation='sigmoid'))
	cnn.compile(loss='binary_crossentropy',optimizer='nadam',metrics=['acc'])
	return cnn


#performCrossValidation([X_train,X_train,X_train],Y_train,epochs=35,batch_size=1000,model_name='./CNN_CV',word_index=word_index,embedding_matrix=embedding_matrix,maxlen=maxlen,trainable=False,NUM_CLASSES=6,getModel=getModel)
from keras.callbacks import *
earlystopping=EarlyStopping(monitor='val_loss',patience=20)
os.mkdir('CNN_BASELINE')
checkpoint=ModelCheckpoint(filepath='./CNN_BASELINE'+os.sep+"weights.{epoch:03d}-{loss:.2f}-{val_loss:.2f}.h5",monitor='val_loss',verbose=0,save_best_only=True,save_weights_only=False,mode='auto',period=1)
#history=getModel(word_index=word_index,embedding_matrix=embedding_matrix,maxlen=maxlen,trainable=False,NUM_CLASSES=6).fit([X_train,X_train,X_train],Y_train,epochs=151,verbose=1,callbacks=[earlystopping,checkpoint],batch_size=64,validation_split=.10)
history=getModel(word_index=word_index,embedding_matrix=embedding_matrix,maxlen=maxlen,trainable=False,NUM_CLASSES=6).fit([X_train,X_train,X_train],Y_train,epochs=151,verbose=1,callbacks=[earlystopping,checkpoint],batch_size=64,validation_split=.10)
##############################################################################################################


test_path='./test.csv'
dataframe=pd.read_csv(test_path)
evaluateTestData(model=cnn,testpath=test_path,datacolumns=datacolumns,fillna_vals=na,tokenizer_path='./tokenizer.tkn',maxlen=maxlen,columns=['id'] + labels,dataframe=dataframe,dataframe_labels=labels,pred_filename='New_Extended_CNN.csv')
##############################################################################################################