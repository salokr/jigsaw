#test dataframe is needed, take care while calling from here
from sklearn.linear_model import LogisticRegression

losses=[]
predictions={'id':test_df['id']}

for class_name in labels:
	classifier=LogisticRegression(C=4.0,solver='sag')
	classifier.fit(train_embs,train_dataframe[class_name])
	predictions[class_name]=classifier.predict_proba(test_embs)[:,1]

submission=pd.DataFrame.from_dict(predictions)
submission.to_csv('LSTM_CNN_LR.csv',index=False)