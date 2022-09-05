from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from modelevaluation import evaluate_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
import numpy as np

#set seed for randomness
def set_seed(seed):
	np.random.RandomState(seed)
	np.random.seed(seed)
	
seed = 665
set_seed(seed)

#train blackbox from given dataset
def trainmodel(data, target, model_arg, datasetname, evaltype):
	if model_arg == 'linear':
		
		#create polynomial model
		model = make_pipeline(PolynomialFeatures(5), Ridge())
		
		#calculate r2 score for regression
		accuracy = evaluate_model(model, data, target, 'reg', datasetname, evaltype)
		
		#train/test split	
		X, X_test, y, y_test = train_test_split(data, target, train_size=0.80, test_size=0.20, random_state=seed)
		
		#data preprocessing
		transform = MinMaxScaler()
		X_preproc = transform.fit_transform(X)
		X_test_preproc = transform.transform(X_test)
		
		#fit data
		model.fit(X, y)
		
		return accuracy, -2, -2, -2, -2, -2, model, X_preproc, X_test_preproc, y_test
		
	elif model_arg == 'treeSurrogatToy':
		
		#create decision trees
		models = []
		for i in range(10):
			models.append(DecisionTreeClassifier(random_state=12345678, max_depth=3))

			#fit data
			models[i].fit(data[i], target[i])
									
		return models
		
	elif model_arg == 'treeSurrogatHeart':
		
		#create decision trees
		models = []
		
		#for max_depth=3
		for i in range(10):
			models.append(DecisionTreeClassifier(random_state=12345678, max_depth=3))

			#fit data
			models[i].fit(data[i], target[i])
		
		#for max_depth=4
		for i in range(10):
			models.append(DecisionTreeClassifier(random_state=12345678, max_depth=4))

			#fit data
			models[i+10].fit(data[i], target[i])
									
		return models
		
	elif model_arg == 'treeSurrogatDiabetes':
		
		#create decision trees
		models = []
		
		#for max_depth=3
		for i in range(10):
			models.append(DecisionTreeClassifier(random_state=12345678, max_depth=3))

			#fit data
			models[i].fit(data[i], target[i])
		
		#for max_depth=4
		for i in range(10):
			models.append(DecisionTreeClassifier(random_state=12345678, max_depth=4))

			#fit data
			models[i+10].fit(data[i], target[i])
									
		return models	
		
	elif model_arg == 'rfr':
		
		#create rf regressor
		model = RandomForestRegressor(random_state=seed)
		
		#calculate r2 score for regression
		accuracy = evaluate_model(model, data, target, 'reg', datasetname, evaltype)

		#train/test split		
		X, X_test, y, y_test = train_test_split(data, target, train_size=0.80, test_size=0.20, random_state=seed)
		
		#data preprocessing
		transform = MinMaxScaler()
		X_preproc = transform.fit_transform(X)
		X_test_preproc = transform.transform(X_test)
		
		#fit data
		model.fit(X, y)
		
		return accuracy, -2, -2, -2, -2, -2, model, X_preproc, X_test_preproc, y_test
		
	elif model_arg == 'rfc':
		
		#create rf classifier
		model = RandomForestClassifier(random_state=seed)
		
		#calculate accuracy, precision, recall, specificity, f1 and mcc
		accuracy, precision, recall, specificity, f1, mcc, params = evaluate_model(model, data, target, 'class', datasetname, evaltype)
		
		#create better rf classifier
		better_model = RandomForestClassifier(**params, random_state=seed)
		
		#train/test split
		X, X_test, y, y_test = train_test_split(data, target, train_size=0.80, test_size=0.20, random_state=seed)
		
		#data preprocessing
		#transform = MinMaxScaler()
		#X_preproc = transform.fit_transform(X)
		#X_test_preproc = transform.transform(X_test)
		
		#fit data
		better_model.fit(X, y)
		
		return accuracy, precision, recall, specificity, f1, mcc, better_model, X, X_test, y_test
