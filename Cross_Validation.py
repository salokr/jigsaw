from keras.models import *
from keras.layers import *
from sklearn.model_selection import KFold
import numpy
from keras.callbacks import *

seed=1331

#dataset is given in numpy format (no dataframe, etc. etc.)
#X_train and Y_train are assumed to be given

#performCrossValidation(X_train,Y_train,epochs=1,batch_size=1000,model_name='./Test')
def performCrossValidation(X,Y,epochs,batch_size,model_name,n_splits=10,shuffle=True,callbacks=None,word_index=None,embedding_matrix=None,maxlen=None,trainable=None,NUM_CLASSES=None,getModel=None):
	kf = KFold(n_splits=n_splits)
	kf.get_n_splits(X)
	cvscores = []
	i=0
	for train_index, val_index in kf.split(X):
		print("TRAIN:", train_index, "TEST:", val_index)
		i=i+1
		X_train, X_val = X[train_index], X[val_index]
		y_train, y_val = Y[train_index], Y[val_index]
		model=getModel(word_index,embedding_matrix,maxlen,trainable,NUM_CLASSES)
		destination=model_name+str(i)
		if os.path.isdir(destination):
			shutil.rmtree(destination, ignore_errors=True)
		os.mkdir(destination)
		checkpoint=ModelCheckpoint(filepath=destination+os.sep+"weights.{epoch:03d}-{loss:.2f}-{val_loss:.2f}.h5",monitor='val_loss',verbose=0,save_best_only=True,save_weights_only=False,mode='auto',period=1)
		earlystopping=EarlyStopping(monitor='val_loss',patience=5)
		model.fit(X_train,y_train,epochs=epochs,batch_size=batch_size,verbose=1,callbacks=callbacks)
		scores=model.evaluate(X_val,y_val,verbose=0)
		print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
		cvscores.append(scores[1] * 100)
		del(model)
	print("%.2f%% (+/- %.2f%%)" % (numpy.mean(cvscores), numpy.std(cvscores)))
#performCrossValidation(X_train,Y_train,epochs=1,batch_size=1000,model_name='./Test')