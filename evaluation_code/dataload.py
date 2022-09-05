import pandas as pd
import numpy as np
from sklearn.datasets import load_boston
from sklearn.datasets import make_classification
import pickle
from tqdm import tqdm
from joblib import load
import os
from termcolor import colored
from modelevaluation import make_prediction
from sklearn.model_selection import RepeatedKFold
import sys

#set seed for randomness
def set_seed(seed):
	np.random.RandomState(seed)
	np.random.seed(seed)
	
seed = 665
set_seed(seed)

#READS/CREATS AND PREPARES DATASETS
def getdata(dataset):
	
	#read an prepare heart failure data set
	if dataset == 'heart':
		
		#read data set from csv
		data = pd.read_csv('dataset/heart_failure_clinical_records_dataset.csv')
		
		#extract feature values and names
		array = data.values
		names = list(data.columns.values)
		
		#remove temporal variable name
		names.remove('time')
		
		#split into prediction features and class (removes temporal variable Follow-up time)
		X = array[:,0:11]
		y = array[:,12]

		return X, y, names
		
	#prepare toy data set (10 bool features, 100 datapoints, if feature 0 == 1 or feature 1 == 1 then class 1
	#but if feature 0==1 and feature 2==1 the class 0, every other combination results in class 0)
	elif dataset == 'toy':
		
		#setup names
		names=['bool0','bool1','bool2','bool3','bool4','bool5','bool6','bool7','bool8','bool9','class']
		
		#setup random state
		np.random.RandomState(seed)
		
		#setup X
		X = np.random.randint(0,2,(500,10))

		#setup y
		y = []
		
		for x in X:
			if x[0] == 1 and x[2] == 1:
				y.append(0)
			elif x[0]==1:
				y.append(1)
			elif x[1]==1:
				y.append(1)
			else:
				y.append(0)

		return X, y, names
	
	#read an prepare diabetes data set
	elif dataset == 'diabetes':
		
		#read data set from csv
		data = pd.read_csv('dataset/diabetes_data_upload.csv')
		
		#extract feature values and names
		array = data.values
		names = list(data.columns.values)
		#names = names[:-1]
			
		#split into prediction features and class
		X = array[:,0:16]
		y = array[:,16]
		
		#replace 'Female'/'Male' and 'Yes'/'No' in X with 1 and 0
		for i in range(len(X)):
			for j in range(len(X[i])):
				if X[i][j] in ['Female', 'Yes']:
					X[i][j] = 1
				elif X[i][j] in ['Male', 'No']:
					X[i][j] = 0
					
		#replace 'Positive' and 'Negative' in y with 1 and 0
		for i in range(len(y)):
			if y[i] == 'Positive':
				y[i] = 1
			elif y[i] == 'Negative':
				y[i] = 0
		
		#setting same type
		X = X.astype('int32')
		y = y.astype('int32')
		
		return X, y, names	
			
	#prepare random data set
	elif dataset == 'random':
		
		names = ""
		
		#prepare X and y
		X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, n_redundant=5, random_state=3)
		
		return X, y, names
	
	#prepare boston data set
	elif dataset == 'boston':
		
		#load boston data set
		boston = load_boston()
		names = np.append(boston.feature_names,['price'])
		
		#prepare X and y
		X = boston.data
		y = boston.target
		
		return X, y, names
		
	elif dataset == 'optilimetest':
		
		names = ""
		
		#prepare X and y
		rng = np.random.RandomState(0)
		x = np.linspace(0,10,100)
		rng.shuffle(x)
		x = np.sort(x[:20])
		X = x[:, np.newaxis]
	
		def f(x):
			return x * np.sin(x) + 10

		y = f(x)
	
		return X, y , names

#load model from .pickle/.joblib file
def loadpickle(path, nametype):
	
	#return correct names
	if nametype == 'calculator':
		names=['0','1','2','3','function==cos','function==sin','function==sqrt','function==tan','8','value==-900','value==0','value','max-char(value)','char-length(value)','value','max-char(value)','char-length(value)','17','18','19','20']
	elif nametype == 'toy':
		names=['0','bool0==0','bool0==1','bool0==exist','bool1==0','bool1==1','bool1==exist','bool2==0','bool2==1','bool2==exist','bool3==0','bool3==1','bool3==exist','bool4==0','bool4==1','bool4==exist','bool5==0','bool5==1','bool5==exist','bool6==0','bool6==1','bool6==exist','bool7==0','bool7==1','bool7==exist','bool8==0','bool8==1','bool8==exist','bool9==0','bool9==1','bool9==exist']
	elif nametype == 'heartfull':
		namesl= []
		df = pd.read_csv('plot/names/heartdepthfull.csv')
		table = df.to_numpy()
		sorttable = table[table[:, 0].argsort()]
		
		#format names
		for i in range(len(sorttable)):
			if sorttable[i][0][:7] == 'max-num':
				namesl.append(sorttable[i][0].partition("@")[0][8:])
			else:
				namesl.append(sorttable[i][3].replace('"', "'"))
		names = np.asarray(namesl)
		
	elif nametype == 'diabetesfull':
		namesl= []
		df = pd.read_csv('plot/names/diabetesdepthfull.csv')
		table = df.to_numpy()
		sorttable = table[table[:, 0].argsort()]
		
		#format names
		for i in range(len(sorttable)):
			if sorttable[i][0][:7] == 'max-num':
				namesl.append(sorttable[i][0].partition("@")[0][8:])
			else:
				namesl.append(sorttable[i][3].replace('"', "'"))
		names = np.asarray(namesl)
				
	elif nametype == 'toysur':
		names=['bool0', 'bool1', 'bool2', 'bool3', 'bool4', 'bool5', 'bool6', 'bool7', 'bool8', 'bool9']
	elif nametype == 'heartsur':
		names=['age', 'anaemia', 'creatinine_phosphokinase', 'diabetes', 'ejection_fraction', 'high_blood_pressure', 'platelets', 'serum_creatinine', 'serum_sodium', 'sex', 'smoking']
	elif nametype == 'diabetessur':
		names=['Age', 'Gender', 'Polyuria', 'Polydipsia', 'sudden_weight_loss', 'weakness', 'Polyphagia', 'Genital_thrush', 'visual_blurring', 'Itching', 'Irritability', 'delayed_healing', 'partial_paresis', 'muscle_stiffness', 'Alopecia', 'Obesity']
	elif nametype == 'none':
		names=[]
		
	#check if .pickle or .joblib
	if path[-6:] == 'pickle':
		#load model form file
		model = pickle.load(open(path, 'rb'))
	elif path[-6:] == 'joblib':
		model = load(path)
	
	return model, names

#generate samples to train and test surrogate models with
def generate_samples(blackbox):
	
	#generate for toydataset
	if blackbox == 'toy':
		
		#load blackbox
		blackbox = load("blackboxes/toy.joblib")
		
		#get paths of folders with generated samples
		foldernames = os.listdir('samples/raw/toydata')
		foldernames.sort()
		
		#check amount of samples
		samplesize = []
		samplesizeall = 0
		for x in foldernames:
			samplesize.append(len(os.listdir('samples/raw/toydata/'+x+'/samples')))
			samplesizeall += len(os.listdir('samples/raw/toydata/'+x+'/samples'))

		#prepare arrays
		samples_x = np.zeros((samplesizeall, 10))
		samples_y = np.zeros((samplesizeall, 1), dtype=int)		
		bugsamples = []
		nobugsamples = []

		#make predictions with the blackbox
		print(colored('Make predictions for samples...', 'green'))	

		for x in range(len(samplesize)):
			
			#get filenames
			filenames = os.listdir('samples/raw/toydata/'+foldernames[x]+'/samples')
			
			for i in tqdm(range(0,samplesize[x]), leave=False):
								
				#load sample from file
				f = open("samples/raw/toydata/"+foldernames[x]+"/samples/"+filenames[i], "r")

				#prepare and split string into array
				currentstring=f.read()[2:-3]					
				split = currentstring.split(",")
				split = [int(x) for x in split]
				samples_x[i] = split			

				#make prediction for the sample
				samples_y[i][0] = int(make_prediction(blackbox, samples_x[i])[0])
				
				if (samples_y[i][0] == 1) and (split not in bugsamples) and (len(bugsamples)<150):
					bugsamples.append(split)
				elif (samples_y[i][0] == 0) and (split not in nobugsamples) and (len(nobugsamples)<150):
					nobugsamples.append(split)
					
		rkf = RepeatedKFold(n_splits=3, n_repeats=4, random_state=seed)
		splittrain = []
		splittest = []
		
		for train_index, test_index in rkf.split(bugsamples):
			splittrain.append(train_index.tolist())
			splittest.append(test_index.tolist())
		
		print(colored('Write sample files...', 'green'))
		for x in tqdm(range(10), leave=False):
				
			#write 200 traning samples to files
			for i in range(100):

				#100 that cause bug	
				p = "samples/toydata/train/train"+str(x+1)+"/toydatatrain."+str(i+1)+".expr"
				f = open(p, "w")
				f.write("[["+str(int(bugsamples[splittrain[x][i]][0]))+","+str(int(bugsamples[splittrain[x][i]][1]))+","+str(int(bugsamples[splittrain[x][i]][2]))+","+str(int(bugsamples[splittrain[x][i]][3]))+","+str(int(bugsamples[splittrain[x][i]][4]))+","+str(int(bugsamples[splittrain[x][i]][5]))+","+str(int(bugsamples[splittrain[x][i]][6]))+","+str(int(bugsamples[splittrain[x][i]][7]))+","+str(int(bugsamples[splittrain[x][i]][8]))+","+str(int(bugsamples[splittrain[x][i]][9]))+"]]\n")
				f.close()
				
				#100 that do not cause bug	
				p = "samples/toydata/train/train"+str(x+1)+"/toydatatrain."+str(i+101)+".expr"
				f = open(p, "w")
				f.write("[["+str(int(nobugsamples[splittrain[x][i]][0]))+","+str(int(nobugsamples[splittrain[x][i]][1]))+","+str(int(nobugsamples[splittrain[x][i]][2]))+","+str(int(nobugsamples[splittrain[x][i]][3]))+","+str(int(nobugsamples[splittrain[x][i]][4]))+","+str(int(nobugsamples[splittrain[x][i]][5]))+","+str(int(nobugsamples[splittrain[x][i]][6]))+","+str(int(nobugsamples[splittrain[x][i]][7]))+","+str(int(nobugsamples[splittrain[x][i]][8]))+","+str(int(nobugsamples[splittrain[x][i]][9]))+"]]\n")
				f.close()

			#write 100 test samples to files			
			for i in range(50):
				
				#50 that cause bug		
				p = "samples/toydata/test/test"+str(x+1)+"/toydatatest."+str(i+1)+".expr"
				f = open(p, "w")
				f.write("[["+str(int(bugsamples[splittest[x][i]][0]))+","+str(int(bugsamples[splittest[x][i]][1]))+","+str(int(bugsamples[splittest[x][i]][2]))+","+str(int(bugsamples[splittest[x][i]][3]))+","+str(int(bugsamples[splittest[x][i]][4]))+","+str(int(bugsamples[splittest[x][i]][5]))+","+str(int(bugsamples[splittest[x][i]][6]))+","+str(int(bugsamples[splittest[x][i]][7]))+","+str(int(bugsamples[splittest[x][i]][8]))+","+str(int(bugsamples[splittest[x][i]][9]))+"]]\n")
				f.close()
				
				#50 that do not cause bug	
				p = "samples/toydata/test/test"+str(x+1)+"/toydatatest."+str(i+51)+".expr"
				f = open(p, "w")
				f.write("[["+str(int(nobugsamples[splittest[x][i]][0]))+","+str(int(nobugsamples[splittest[x][i]][1]))+","+str(int(nobugsamples[splittest[x][i]][2]))+","+str(int(nobugsamples[splittest[x][i]][3]))+","+str(int(nobugsamples[splittest[x][i]][4]))+","+str(int(nobugsamples[splittest[x][i]][5]))+","+str(int(nobugsamples[splittest[x][i]][6]))+","+str(int(nobugsamples[splittest[x][i]][7]))+","+str(int(nobugsamples[splittest[x][i]][8]))+","+str(int(nobugsamples[splittest[x][i]][9]))+"]]\n")
				f.close()

	#generate for heartfailure dataset						
	elif blackbox == 'heart':
		
		#load blackbox
		blackbox = load("blackboxes/heart.joblib")
		
		#get paths of folders with generated samples
		foldernames = os.listdir('samples/raw/heartfailure')
		foldernames.sort()
		
		#check amount of samples
		samplesizeh = []
		samplesizeallh = 0
		for x in foldernames:
			samplesizeh.append(len(os.listdir('samples/raw/heartfailure/'+x+'/samples')))
			samplesizeallh += len(os.listdir('samples/raw/heartfailure/'+x+'/samples'))

		#prepare arrays
		samplesh_x = np.zeros((samplesizeallh, 11))
		samplesh_y = np.zeros((samplesizeallh, 1), dtype=int)		
		bugsamples = []
		nobugsamples = []

		#make predictions with the blackbox
		print(colored('Make predictions for samples...', 'green'))	

		for x in range(2): #len(samplesizeh)
			
			#get filenames
			filenames = os.listdir('samples/raw/heartfailure/'+foldernames[x]+'/samples')
			
			for i in tqdm(range(0,samplesizeh[x]), leave=False):
								
				#load sample from file
				f = open("samples/raw/heartfailure/"+foldernames[x]+"/samples/"+filenames[i], "r")

				#prepare and split string into array
				currentstring=f.read()[2:-3]					
				split = currentstring.split(",")
				split = [float(x) for x in split]
				samplesh_x[i] = split			

				#make prediction for the sample
				samplesh_y[i][0] = int(make_prediction(blackbox, samplesh_x[i])[0])
				
				if (samplesh_y[i][0] == 1) and (split not in bugsamples) and (len(bugsamples)<750):
					bugsamples.append(split)
				elif (samplesh_y[i][0] == 0) and (split not in nobugsamples) and (len(nobugsamples)<750):
					nobugsamples.append(split)
					
		rkf = RepeatedKFold(n_splits=3, n_repeats=4, random_state=seed)
		splittrain = []
		splittest = []
				
		for train_index, test_index in rkf.split(bugsamples):
			splittrain.append(train_index.tolist())
			splittest.append(test_index.tolist())
		
		#print(splittrain)
		#print(splittest)
		
		print(colored('Write sample files...', 'green'))
		for x in tqdm(range(10), leave=False):
			
			#write 1000 traning samples to files
			for i in range(500):
				
				#handel rules of serum_creatinine form
				serumbug = ''
				serumnobug = ''
				
				tempbug=str(bugsamples[splittrain[x][i]][7])
				if len(tempbug) == 4:
					if tempbug[-1] == '0' and tempbug[-2] == '0':
						serumbug = tempbug[0]
					else:
						serumbug = tempbug
				elif len(tempbug) == 3:
					if tempbug[-1] == '0':
						serumbug = tempbug[0]
					else:
						serumbug = tempbug+'0'
				elif len(tempbug) == 1:
					serumbug = tempbug
				
				tempnobug=str(nobugsamples[splittrain[x][i]][7])
				if len(tempnobug) == 4:
					if tempnobug[-1] == '0' and tempnobug[-2] == '0':
						serumnobug = tempnobug[0]
					else:
						serumnobug = tempnobug
				elif len(tempnobug) == 3:
					if tempnobug[-1] == '0':
						serumnobug = tempnobug[0]
					else:
						serumnobug = tempnobug+'0'
				elif len(tempnobug) == 1:
					serumnobug = tempnobug		
				
				#500 that cause bug	
				p = "samples/heartfailure/train/train"+str(x+1)+"/heartfailuretrain."+str(i+1)+".expr"
				f = open(p, "w")
				f.write("[["+str(int(bugsamples[splittrain[x][i]][0]))+","+str(int(bugsamples[splittrain[x][i]][1]))+","+str(int(bugsamples[splittrain[x][i]][2]))+","+str(int(bugsamples[splittrain[x][i]][3]))+","+str(int(bugsamples[splittrain[x][i]][4]))+","+str(int(bugsamples[splittrain[x][i]][5]))+","+str(int(bugsamples[splittrain[x][i]][6]))+","+serumbug+","+str(int(bugsamples[splittrain[x][i]][8]))+","+str(int(bugsamples[splittrain[x][i]][9]))+","+str(int(bugsamples[splittrain[x][i]][10]))+"]]\n")
				f.close()

				#500 that do not cause bug				
				p = "samples/heartfailure/train/train"+str(x+1)+"/heartfailuretrain."+str(i+501)+".expr"
				f = open(p, "w")
				f.write("[["+str(int(nobugsamples[splittrain[x][i]][0]))+","+str(int(nobugsamples[splittrain[x][i]][1]))+","+str(int(nobugsamples[splittrain[x][i]][2]))+","+str(int(nobugsamples[splittrain[x][i]][3]))+","+str(int(nobugsamples[splittrain[x][i]][4]))+","+str(int(nobugsamples[splittrain[x][i]][5]))+","+str(int(nobugsamples[splittrain[x][i]][6]))+","+serumnobug+","+str(int(nobugsamples[splittrain[x][i]][8]))+","+str(int(nobugsamples[splittrain[x][i]][9]))+","+str(int(nobugsamples[splittrain[x][i]][10]))+"]]\n")
				f.close()

			#write 500 test samples to files			
			for i in range(250):
				
				#handel rules of serum_creatinine form
				serumbug = ''
				serumnobug = ''
				
				tempbug=str(bugsamples[splittest[x][i]][7])
				if len(tempbug) == 4:
					if tempbug[-1] == '0' and tempbug[-2] == '0':
						serumbug = tempbug[0]
					else:
						serumbug = tempbug
				elif len(tempbug) == 3:
					if tempbug[-1] == '0':
						serumbug = tempbug[0]
					else:
						serumbug = tempbug+'0'
				elif len(tempbug) == 1:
					serumbug = tempbug
				
				tempnobug=str(nobugsamples[splittest[x][i]][7])
				if len(tempnobug) == 4:
					if tempnobug[-1] == '0' and tempnobug[-2] == '0':
						serumnobug = tempnobug[0]
					else:
						serumnobug = tempnobug
				elif len(tempnobug) == 3:
					if tempnobug[-1] == '0':
						serumnobug = tempnobug[0]
					else:
						serumnobug = tempnobug+'0'
				elif len(tempnobug) == 1:
					serumnobug = tempnobug		

				
				#250 that cause bug					
				p = "samples/heartfailure/test/test"+str(x+1)+"/heartfailuretest."+str(i+1)+".expr"
				f = open(p, "w")
				f.write("[["+str(int(bugsamples[splittest[x][i]][0]))+","+str(int(bugsamples[splittest[x][i]][1]))+","+str(int(bugsamples[splittest[x][i]][2]))+","+str(int(bugsamples[splittest[x][i]][3]))+","+str(int(bugsamples[splittest[x][i]][4]))+","+str(int(bugsamples[splittest[x][i]][5]))+","+str(int(bugsamples[splittest[x][i]][6]))+","+serumbug+","+str(int(bugsamples[splittest[x][i]][8]))+","+str(int(bugsamples[splittest[x][i]][9]))+","+str(int(bugsamples[splittest[x][i]][10]))+"]]\n")
				f.close()

				#250 that do not cause bug			
				p = "samples/heartfailure/test/test"+str(x+1)+"/heartfailuretest."+str(i+251)+".expr"
				f = open(p, "w")
				f.write("[["+str(int(nobugsamples[splittest[x][i]][0]))+","+str(int(nobugsamples[splittest[x][i]][1]))+","+str(int(nobugsamples[splittest[x][i]][2]))+","+str(int(nobugsamples[splittest[x][i]][3]))+","+str(int(nobugsamples[splittest[x][i]][4]))+","+str(int(nobugsamples[splittest[x][i]][5]))+","+str(int(nobugsamples[splittest[x][i]][6]))+","+serumnobug+","+str(int(nobugsamples[splittest[x][i]][8]))+","+str(int(nobugsamples[splittest[x][i]][9]))+","+str(int(nobugsamples[splittest[x][i]][10]))+"]]\n")
				f.close()

	#generate for diabetes dataset						
	elif blackbox == 'diabetes':
		
		#load blackbox
		blackbox = load("blackboxes/diabetes.joblib")
		
		#get paths of folders with generated samples
		foldernames = os.listdir('samples/raw/diabetes')
		foldernames.sort()
		
		#check amount of samples
		samplesizeh = []
		samplesizeallh = 0
		for x in foldernames:
			samplesizeh.append(len(os.listdir('samples/raw/diabetes/'+x+'/samples')))
			samplesizeallh += len(os.listdir('samples/raw/diabetes/'+x+'/samples'))

		#prepare arrays
		samplesh_x = np.zeros((samplesizeallh, 16))
		samplesh_y = np.zeros((samplesizeallh, 1), dtype=int)		
		bugsamples = []
		nobugsamples = []

		#make predictions with the blackbox
		print(colored('Make predictions for samples...', 'green'))	

		for x in range(1): #len(samplesizeh)
			
			#get filenames
			filenames = os.listdir('samples/raw/diabetes/'+foldernames[x]+'/samples')
			
			for i in tqdm(range(0,samplesizeh[x]), leave=False):
								
				#load sample from file
				f = open("samples/raw/diabetes/"+foldernames[x]+"/samples/"+filenames[i], "r")

				#prepare and split string into array
				currentstring=f.read()[2:-3]					
				split = currentstring.split(",")
				split = [int(x) for x in split]
				samplesh_x[i] = split			

				#make prediction for the sample
				samplesh_y[i][0] = int(make_prediction(blackbox, samplesh_x[i])[0])
				
				if (samplesh_y[i][0] == 1) and (split not in bugsamples) and (len(bugsamples)<750):
					bugsamples.append(split)
				elif (samplesh_y[i][0] == 0) and (split not in nobugsamples) and (len(nobugsamples)<750):
					nobugsamples.append(split)
		
		#print(len(bugsamples))
		#print(len(nobugsamples))			
		
		rkf = RepeatedKFold(n_splits=3, n_repeats=4, random_state=seed)
		splittrain = []
		splittest = []
				
		for train_index, test_index in rkf.split(bugsamples):
			splittrain.append(train_index.tolist())
			splittest.append(test_index.tolist())
		
		#print(splittrain)
		#print(splittest)
		
		print(colored('Write sample files...', 'green'))
		for x in tqdm(range(10), leave=False):
			
			#write 1000 traning samples to files
			for i in range(500):
								
				#500 that cause bug	
				p = "samples/diabetes/train/train"+str(x+1)+"/diabetestrain."+str(i+1)+".expr"
				f = open(p, "w")
				f.write("[["+str(int(bugsamples[splittrain[x][i]][0]))+","+str(int(bugsamples[splittrain[x][i]][1]))+","+str(int(bugsamples[splittrain[x][i]][2]))+","+str(int(bugsamples[splittrain[x][i]][3]))+","+str(int(bugsamples[splittrain[x][i]][4]))+","+str(int(bugsamples[splittrain[x][i]][5]))+","+str(int(bugsamples[splittrain[x][i]][6]))+","+str(int(bugsamples[splittrain[x][i]][7]))+","+str(int(bugsamples[splittrain[x][i]][8]))+","+str(int(bugsamples[splittrain[x][i]][9]))+","+str(int(bugsamples[splittrain[x][i]][10]))+","+str(int(bugsamples[splittrain[x][i]][11]))+","+str(int(bugsamples[splittrain[x][i]][12]))+","+str(int(bugsamples[splittrain[x][i]][13]))+","+str(int(bugsamples[splittrain[x][i]][14]))+","+str(int(bugsamples[splittrain[x][i]][15]))+"]]\n")
				f.close()

				#500 that do not cause bug				
				p = "samples/diabetes/train/train"+str(x+1)+"/diabetestrain."+str(i+501)+".expr"
				f = open(p, "w")
				f.write("[["+str(int(nobugsamples[splittrain[x][i]][0]))+","+str(int(nobugsamples[splittrain[x][i]][1]))+","+str(int(nobugsamples[splittrain[x][i]][2]))+","+str(int(nobugsamples[splittrain[x][i]][3]))+","+str(int(nobugsamples[splittrain[x][i]][4]))+","+str(int(nobugsamples[splittrain[x][i]][5]))+","+str(int(nobugsamples[splittrain[x][i]][6]))+","+str(int(nobugsamples[splittrain[x][i]][7]))+","+str(int(nobugsamples[splittrain[x][i]][8]))+","+str(int(nobugsamples[splittrain[x][i]][9]))+","+str(int(nobugsamples[splittrain[x][i]][10]))+","+str(int(nobugsamples[splittrain[x][i]][11]))+","+str(int(nobugsamples[splittrain[x][i]][12]))+","+str(int(nobugsamples[splittrain[x][i]][13]))+","+str(int(nobugsamples[splittrain[x][i]][14]))+","+str(int(nobugsamples[splittrain[x][i]][15]))+"]]\n")
				f.close()

			#write 500 test samples to files			
			for i in range(250):
							
				#250 that cause bug					
				p = "samples/diabetes/test/test"+str(x+1)+"/diabetestest."+str(i+1)+".expr"
				f = open(p, "w")
				f.write("[["+str(int(bugsamples[splittest[x][i]][0]))+","+str(int(bugsamples[splittest[x][i]][1]))+","+str(int(bugsamples[splittest[x][i]][2]))+","+str(int(bugsamples[splittest[x][i]][3]))+","+str(int(bugsamples[splittest[x][i]][4]))+","+str(int(bugsamples[splittest[x][i]][5]))+","+str(int(bugsamples[splittest[x][i]][6]))+","+str(int(bugsamples[splittest[x][i]][7]))+","+str(int(bugsamples[splittest[x][i]][8]))+","+str(int(bugsamples[splittest[x][i]][9]))+","+str(int(bugsamples[splittest[x][i]][10]))+","+str(int(bugsamples[splittest[x][i]][11]))+","+str(int(bugsamples[splittest[x][i]][12]))+","+str(int(bugsamples[splittest[x][i]][13]))+","+str(int(bugsamples[splittest[x][i]][14]))+","+str(int(bugsamples[splittest[x][i]][15]))+"]]\n")
				f.close()

				#250 that do not cause bug			
				p = "samples/diabetes/test/test"+str(x+1)+"/diabetestest."+str(i+251)+".expr"
				f = open(p, "w")
				f.write("[["+str(int(nobugsamples[splittest[x][i]][0]))+","+str(int(nobugsamples[splittest[x][i]][1]))+","+str(int(nobugsamples[splittest[x][i]][2]))+","+str(int(nobugsamples[splittest[x][i]][3]))+","+str(int(nobugsamples[splittest[x][i]][4]))+","+str(int(nobugsamples[splittest[x][i]][5]))+","+str(int(nobugsamples[splittest[x][i]][6]))+","+str(int(nobugsamples[splittest[x][i]][7]))+","+str(int(nobugsamples[splittest[x][i]][8]))+","+str(int(nobugsamples[splittest[x][i]][9]))+","+str(int(nobugsamples[splittest[x][i]][10]))+","+str(int(nobugsamples[splittest[x][i]][11]))+","+str(int(nobugsamples[splittest[x][i]][12]))+","+str(int(nobugsamples[splittest[x][i]][13]))+","+str(int(nobugsamples[splittest[x][i]][14]))+","+str(int(nobugsamples[splittest[x][i]][15]))+"]]\n")
				f.close()
