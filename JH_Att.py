#Rank : 838	 Score :.143
import pandas as pd
import numpy as np
from keras.preprocessing.text import Tokenizer

import os

import pickle


max_features=20000
tokenizer=Tokenizer(num_words=max_features)



train_path='./train.csv'

dataframe=pd.read_csv(train_path)

X_train=dataframe['comment_text'].fillna('_na_').values

labels=['toxic','severe_toxic','obscene','threat','insult','identity_hate']

Y_train=dataframe[labels].values

maxlen=100

#Hash... Hash... :D
tokenizer.fit_on_texts(X_train)
X_train=tokenizer.texts_to_sequences(X_train)
###################################################################
word_index=tokenizer.word_index

from keras.preprocessing.sequence import pad_sequences
X_train=pad_sequences(X_train,maxlen=maxlen)
print(X_train.shape)
print(X_train[0])



we_dim=300

embed_size=300


embedding_path='./../glove.840B.300d.txt'

embedding_dict={}

f=open(embedding_path,'r')

for line in f:
	fields=line.strip().split()
	#print(fields)
	word=fields[0]
	w_e=np.asarray(fields[1:],dtype='float32')
	embedding_dict[word]=w_e
	#print("w_e:",w_e)
	#sys.exit(0)

f.close()

we_dim=300


allembs=np.stack(embedding_dict.values())
emb_mean,emb_std=allembs.mean(),allembs.std()
nb_words=min(max_features,len(word_index))

#Correct



embedding_matrix=np.random.normal(emb_mean,emb_std,(nb_words,we_dim))

#Build the weight matrix for embeddings for initialization process :D
for word,index in word_index.items():
	if index>=max_features: continue
	vector=embedding_dict.get(word)
	if vector is not None:
		embedding_matrix[index]=vector
'''
def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')

embeddings_dict = dict(get_coefs(*o.strip().split()) for o in open(embedding_path))

all_embs = np.stack(embeddings_dict.values())


emb_mean,emb_std = all_embs.mean(), all_embs.std()
emb_mean,emb_std

#Correct

word_index = tokenizer.word_index
nb_words = min(max_features, len(word_index))
embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
for word, i in word_index.items():
    if i >= max_features: continue
    embedding_vector = embeddings_dict.get(word)
    if embedding_vector is not None: embedding_matrix[i] = embedding_vector
'''
#Correct
#del(allembs)
#del(embedding_dict)
#del(dataframe)


NUM_CLASSES=Y_train.shape[1]

from keras.layers import *
from keras.models import *
from keras.layers.recurrent import *
from keras.layers.pooling import *
from keras.layers.embeddings import *
from keras.layers.wrappers import Bidirectional

if not os.path.isdir('./Bi_LSTM_BASELINE'):
	os.mkdir('Bi_LSTM_BASELINE')

from keras.callbacks import *
checkpoint=ModelCheckpoint(filepath='./Bi_LSTM_BASELINE'+os.sep+"weights.{epoch:03d}-{loss:.2f}-{val_loss:.2f}.h5",monitor='val_loss',verbose=0,save_best_only=True,save_weights_only=False,mode='auto',period=1)
earlystopping=EarlyStopping(monitor='val_loss',patience=5)

'''
model=Sequential()
model.add(Embedding(max_features,300,weights=[embedding_matrix],input_length=maxlen))
model.add(Bidirectional(LSTM(50,return_sequences=True,dropout=0.1,recurrent_dropout=0.1)))
model.add(GlobalMaxPool1D())
model.add(Dense(50,activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(NUM_CLASSES,activation='sigmoid'))
model.compile(loss='binary_crossentropy',optimizer='nadam',metrics=['acc'])
history=model.fit(X_train,Y_train,epochs=51,verbose=1,callbacks=[earlystopping,checkpoint],batch_size=32,validation_split=.10)
#Load the best model
from keras.models import load_model
#model=load_model('./Bi_LSTM_BASELINE/weights.000-0.16-0.14.h5')
'''

print(X_train[0])
print(X_train.shape)



inp = Input(shape=(maxlen,))
x = Embedding(max_features, 300, weights=[embedding_matrix])(inp)
x = Bidirectional(LSTM(50, return_sequences=True, dropout=0.1, recurrent_dropout=0.1))(x)
x = GlobalMaxPool1D()(x)
x = Dense(50, activation="relu")(x)
x = Dropout(0.1)(x)
x = Dense(6, activation="sigmoid")(x)
model = Model(inputs=inp, outputs=x)
model.compile(loss='binary_crossentropy', optimizer='nadam', metrics=['accuracy'])

model.fit(X_train, Y_train, batch_size=32, epochs=2) # validation_split=0.1);
'''

test_path='./test.csv'
dataframe=pd.read_csv(test_path)
X_test=dataframe['comment_text'].fillna('<UNKNOWN>').values
X_test=tokenizer.texts_to_sequences(X_test)
X_test=pad_sequences(X_test,maxlen=maxlen)
probabilities=model.predict(X_test)

submission_df = pd.DataFrame(columns=['id'] + labels)
submission_df['id'] = dataframe['id'].values 
submission_df[labels] = probabilities
submission_df.to_csv("./lstm_submission.csv", index=False)
'''
"""

#Even Worse, ensemble of lstm and cnn

cnn=pd.read_csv('cnn_multifilter_submission.csv')
lstm=pd.read_csv('lstm_submission.csv')
p_res=lstm.copy()
p_res[labels]=(lstm[labels]+cnn[labels])/2
p_res.to_csv('CNN_LSTM_Ensemble.csv',index=False)
"""