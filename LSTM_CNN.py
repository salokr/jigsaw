from keras.layers import *
from keras.models import *
from keras.layers.recurrent import *
from keras.layers.pooling import *
from keras.layers.convolutional import *
from keras.layers.embeddings import *
from keras.callbacks import *
from keras.utils import *


input_layer=Input(shape=(maxlen,),dtype='int32')
embedding_layer=Embedding(len(word_index)+1,emb_dim,weights=[embedding_matrix],input_length=maxlen,trainable=False)
emb_seq=embedding_layer(input_layer)

#######################FIRST_LAYER_LSTM's#################################################
x=Bidirectional(LSTM(51,return_sequences=True,dropout=0.2,recurrent_dropout=0.2))(emb_seq)# # #
x=LeakyReLU()(x)
x=BatchNormalization()(x)
#######################SECOND_LAYERS_CNN##################################################
submodels=[]
filter_size=[3,5,7]

for f in filter_size:
	y=Conv1D(128,f,strides=1)(x)
	y=LeakyReLU()(y)
	y=MaxPooling1D(2)(y)
	y=Dropout(0.1)(y)
	y=Conv1D(64,4)(y)#
	y=LeakyReLU()(y)
	y=GlobalMaxPooling1D()(y)
	y=BatchNormalization()(y)
	y=Dropout(0.1)(y)
	model=Model(inputs=input_layer,outputs=y)
	submodels.append(model)

#######################Combining_All_The_Outputs##########################################
directory_name='LSTM_CNN'
os.mkdir(directory_name)
merged=Sequential()
merged.add(Merge(submodels,mode='concat'))
########################A Classifier#######################################################
merged.add(Dense(64))
merged.add(LeakyReLU())
merged.add(Dense(8))
merged.add(LeakyReLU())
merged.add(Dense(NUM_CLASSES,activation='sigmoid'))
merged.compile(loss='binary_crossentropy',optimizer='nadam',metrics=['acc'])
earlystopping=EarlyStopping(monitor='val_loss',patience=3)
checkpoint=ModelCheckpoint(filepath='./'+directory_name+os.sep+"weights.{epoch:03d}-{loss:.2f}-{val_loss:.2f}.h5",monitor='val_loss',verbose=0,save_best_only=True,save_weights_only=False,mode='auto',period=1)
merged.fit(X_train,Y_train,epochs=51,verbose=1,callbacks=[earlystopping,checkpoint],batch_size=500,validation_split=.10)


plot_model(submodels[0],'./Submodels.png')
plot_model(merged,'./Merged_Submodels.png')