import pandas as pd
import numpy as np
import os
import pickle
import shutil
from keras.callbacks import *
from keras.preprocessing.sequence import pad_sequences


def load_jigsaw():
	f=open('./Emb_Mat.mat','r')
	embedding_matrix=pickle.load(f)
	print('Emb Matrix Done')
	f.close()
	f=open('./New_Tokenizer.tkn','r')
	tokenizer=pickle.load(f)
	print('Tok Done')
	f.close()
	train_path='./train.csv'
	dataframe=pd.read_csv(train_path)
	X_train=dataframe['comment_text'].fillna('<UNK>').values
	labels=['toxic','severe_toxic','obscene','threat','insult','identity_hate']
	Y_train=dataframe[labels].values
	maxlen=100
	word_index=tokenizer.word_index
	X_train=tokenizer.texts_to_sequences(X_train)
	X_train=pad_sequences(X_train,maxlen=maxlen)
	we_dim=300
	NUM_CLASSES=Y_train.shape[1]
	print('X_ and Y_ train Done')
	destination=raw_input('Plz Enter Model Name : ')
	destination = destination + '_jigsaw'
	if os.path.isdir(destination):
		shutil.rmtree(destination, ignore_errors=True)
	os.mkdir(destination)
	checkpoint=ModelCheckpoint(filepath=destination+os.sep+".h5",monitor='val_loss',verbose=0,save_best_only=True,save_weights_only=False,mode='auto',period=1)
	earlystopping=EarlyStopping(monitor='val_loss',patience=5)
	test_path='./test.csv'
	test_df=pd.read_csv(test_path)
	X_test=test_df['comment_text'].fillna('<UNK>').values
	X_test=tokenizer.texts_to_sequences(X_test)
	X_test=pad_sequences(X_test,maxlen=maxlen)
	return embedding_matrix,tokenizer,dataframe,X_train,labels,Y_train,maxlen,word_index,we_dim,NUM_CLASSES,destination,X_test