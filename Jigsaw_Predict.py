import pandas as pd
import numpy as np
from keras.preprocessing.sequence import pad_sequences
labels=['toxic','severe_toxic','obscene','threat','insult','identity_hate']


def predict_save_df(test_file_address,model,df_name,tokenizer=None,maxlen=None):
	test_df=pd.read_csv(test_file_address)
	X_test=test_df['comment_text'].fillna('<UNK>')
	X_test=tokenizer.texts_to_sequences(X_test)
	X_test=pad_sequences(X_test,maxlen=maxlen)
	probabilities=model.predict(X_test)
	submission_df = pd.DataFrame(columns=['id'] + labels)
	submission_df['id'] = test_df['id'].values 
	submission_df[labels] = probabilities
	submission_df.to_csv("./" + raw_input('Enter Prediction File Name(Don\'t append .csv at end) : ') + '_jigsaw.csv',index=False)