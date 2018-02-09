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


# Farzi Wala Auto-Encoder
input_layer=Input(shape=(maxlen,),dtype='int32')
embedding_layer=Embedding(len(word_index)+1,we_dim,weights=[embedding_matrix],input_length=maxlen,trainable=False)
emb_seq=embedding_layer(input_layer)

ground_truth=Flatten()(emb_seq)

x=Bidirectional(LSTM(100,return_sequences=True,dropout=0.3,recurrent_dropout=0.2,activation='tanh'))(emb_seq)
#########################################################We can, extract most important summary till now from here###############################################
pooled=GlobalMaxPool1D()(x)
######################################################### :) ------------------------------- :) ###############################################
x=Bidirectional(LSTM(50,activation='tanh'))(x)
################################Now we have summary here, do the rest of the things here. Combine latest summary with the most important learned summary till now.
from keras.layers import concatenate
merged_summary=concatenate([pooled,x],axis=-1)
#Use this merged_summary to reconstruct everything
decoders=Dense(2000,activation='relu')(merged_summary)
decoders=Dense(500,activation='tanh')(decoders)
decoders=Dense(maxlen*we_dim)(decoders)#Predict each embedding val
model=Model(inputs=input_layer,outputs=decoders)


output_model=Model(inputs=input_layer,outputs=ground_truth)

ground_truth_embeddings=output_model.predict(X_train)

from keras.losses import *
def my_custom_loss(y_true,y_pred):
	kld=kullback_leibler_divergence(y_true,y_pred)
	mse=mean_squared_error(y_true,y_pred)
	print('MSE ' + str(mse)+' KLD ' +str(kld))
	return kld +mse 



model.compile(optimizer='sgd',loss='mse',metrics=['acc'])


from keras.callbacks import *
checkpoint=ModelCheckpoint(filepath='./MY_FIRST_AE/'+os.sep+"weights.h5",monitor='val_loss',verbose=0,save_best_only=True,save_weights_only=False,mode='auto',period=1)

model.fit(X_train,ground_truth_embeddings,batch_size=500,verbose=1,validation_split=.10,epochs=51,callbacks=[checkpoint])