import os
import re
from keras.utils import np_utils
import numpy as np
from sklearn.preprocessing import LabelEncoder
import string
#stemmer=PorterStemmer()
#ssh fresco_nadia_new_prod_mumbai 
#tkenize = lambda doc: re.split('\W+',doc.lower())
def readFileLineByLine(path):
	x=[]
	with open(path,'r') as file:
		for line in file:
			x=x+[line.lower().strip()]
	return x


def readFileNSeparate(path):
	x=[]
	list=os.listdir(path)
	for f in list:
		filename=os.path.join(path,f)
		if os.path.isdir(filename):
			continue
		with open(filename,'r') as file:
			for line in file:
				x=x+[(line.lower().strip()).split(".")]
	y=[]
	for xx in x:
		for xxx in xx:
			y=y+[xxx]
	return y

#x=readFileNSeparate('/home/ilab/Downloads/Punnet_Sir_Papers')



def readDirectory(path):
	#List of files
	list=os.listdir(path)
	data=[]
	for f in list:
		data=data+(readFileLineByLine(os.path.join(path,f)))
	return data

def readByDirectoryName(path):
	#List of files
	list=os.listdir(path)
	data=[]
	labels=[]
	for f in list:
		filename=os.path.join(path,f)
		if os.path.isdir(filename):
			continue
		contents=readFileLineByLine(filename)
		data=data+contents
		for i in range(len(contents)):
			labels.append(f)
	return [data,labels]

def readLabels(path):
	labels=[]
	with open(path,'r') as file:
		for line in file:
			labels.append(line.strip())
	enocoder=LabelEncoder()
	encoder.fit(labels)
	Y=encoder.transform(labels)
	return np_utils.to_categorical(Y)



def we2table(path):
	embedding_index={}
	f=open(path)
	for line in f:
		values=line.split(' ')
		word=values[0]
		coefs=np.asarray(values[1:],dtype='float32')
		embedding_index[word]=coefs
	f.close()
	return embedding_index
	


def maxLengthInFile(filepath):
	max=0
	f=open(filepath)
	return maxLengthInList(f,regex)
'''
def maxLengthInList(list,regex):
	max=0
	for line in list:
		currlen=getLen(line,regex)
		#print('Currlen is ',currlen)
		if currlen>max:
			max=currlen
	return max
'''
def maxLengthInList(list,regex):
	l=''
	max=0
	for line in list:
		currlen=getLen(line,regex)
		#print('Currlen is ',currlen)
		if currlen>max:
			max=currlen
			l=line
	print(l)
	return max

def getNaN(times):
	blank=''
	for i in range(times):
		blank=blank+' NaN'
	return blank


#########################################!!!!WARNING ZONE!!!!#################################################
def padSentence(sentence,maxlength,regex):
	sentence=removePun(sentence).strip()
	current_len=getLen(sentence,regex)
	if(sentence in ['',' ','  ']):
		current_len=0
	remaining_len=maxlength-current_len
	if(remaining_len<=0):
		#print('Max Sentence ',sentence)
		return [sentence]
	return [(sentence+getNaN(remaining_len)).strip()]

'''
To convert a string in a list
' '.join(list)
'''

def padList(list,regex,max):
	paddedList=[]
	if max is None:
		max=maxLengthInList(list,regex)
	print('Maximum Length is ',max)
	for items in list:
		paddedList=paddedList+padSentence(items,int(max),regex)
	return paddedList

def getLen(sentence,regex):
	return int(len([x for x in (re.split(regex,(sentence))) if x]))



def removePun(sentence):
	punctuation='[!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~]'
	sentence=re.sub(punctuation,' ',sentence)
	sentence=re.sub(r'[^\w\s]',' ',sentence)
	return re.sub('[ ]+',' ',sentence)



#########################################!!!!WARNING ZONE!!!!#################################################
def getVectorForWord(word,model,embedd_dim):
	x=np.zeros(embedd_dim)
	try:
		x=model[word]
	except:
		if(word not in ['NaN']):
			with open('Unknown.txt','a+') as uk:
				uk.write(word+'\n')
	return x

def getConcVectorForSentence(sentence,regex,model,embedd_dim):
	x=np.zeros(embedd_dim*getLen(sentence,regex))
	end=0
	i=-1
	for words in re.split(regex,sentence):
		i=end
		end=i+embedd_dim
		x[i:end]=getVectorForWord(words,model,embedd_dim)
	return x

def getMatrix(list,model,regex,embedd_dim):
	rows=int(len(list))
	collen=getLen(list[0],regex)
	x=np.zeros(shape=(rows,collen*embedd_dim))
	for i in range(rows):
		x[i]=getConcVectorForSentence(list[i],regex,model,embedd_dim)
	return x


#text,'\W+',we2table('/home/ilab/Desktop/Project/glove.6B.100d.txt'),100


#################################################Read Test Files, Split by Full stops###################################
def readFilesByRegex(file,regex):
	ti=''
	x=''
	with open(file,'r') as ff:
		for lines in ff:
			ti=ti+' '+(lines.strip()).lower()
	intro=ti.split(regex)
	try:
		x=intro[1]
	except:
		print('Error File :',file )
	return x

def readTestFiles(dirPath):
	data=''
	list=os.listdir(dirPath)
	for f in list:
		filename=os.path.join(dirPath,f)
		if os.path.isdir(filename):
			continue
		contents=readFilesByRegex(filename,'@introduction')
		data=data+(contents)
	return data.split(".")
	

#cc=readTestFiles(path)
def removeTex(sentence):
	return re.sub('\\w+{.*}','',sentence)

def readMultiLineFile(path):
	texts=[]
	labels=[]
	files=os.listdir(path)
	for f in files:
		n_data=''
		n_path=os.path.join(path,f)
		#print(n_path)
		with open(n_path,'r') as ff:
			for line in ff:
				n_data=n_data+' '+removePun(line.lower().strip())
				if line.strip()=='':
					if n_data.strip()=='':
						continue
					texts=texts+[n_data]
					n_data=''
					labels=labels+[f]		
				#print(n_data)
	return texts,labels
#x,y=readMultiLineFile('/data1/saurabh/Paper_Data')