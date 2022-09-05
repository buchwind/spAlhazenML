from sklearn.model_selection import cross_validate, cross_val_score, RepeatedStratifiedKFold, RepeatedKFold, GridSearchCV
from sklearn.metrics import recall_score, make_scorer, matthews_corrcoef
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.pipeline import Pipeline
from termcolor import colored
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import sys

#set seed for randomness
def set_seed(seed):
	np.random.RandomState(seed)
	np.random.seed(seed)
	
seed = 665
set_seed(seed)

#get models to test best amount of samples to draw from X to train each tree
def get_models_samples():
	
	models = dict()
	
	for i in np.arange(0.2, 1.1, 0.1):
			key = '%.1f' % i
			if i >= 1.0:
				key = 'None'
				i = None
			models[key] = RandomForestClassifier(max_samples=i, random_state=seed)
			
	return models
	
#get models to test best number of features to consider for split
def get_models_features():
	
	models = dict()
	
	for i in range(1,12):
			models[str(i)] = RandomForestClassifier(max_features=i, random_state=seed)
			
	return models

#get models to test best number of trees in random forest
def get_models_trees():
	
	models = dict()
	i_trees = [50, 100, 200, 500, 1000]
	
	for i in i_trees:
			models[str(i)] = RandomForestClassifier(n_estimators=i, random_state=seed)
			
	return models

#get models to test best tree depth in random forest
def get_models_depth():
	
	models = dict()
	depths = [i for i in range(2,9)] + [None]
	
	for i in depths:
			models[str(i)] = RandomForestClassifier(max_depth=i, random_state=seed)
			
	return models

#evaluate a model performance for a dataset
def evaluate_performance(model, X, y):
	
	#data preprocessing
	#transform = MinMaxScaler()
	#pipeline = Pipeline(steps=[('t', transform), ('m', model)])
	
	#custom scorer for MCC(Matthews correlation coefficient)
	#matthews = make_scorer(matthews_corrcoef)
	
	cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=seed)
	scores = cross_val_score(model, X, y, scoring='f1', cv=cv, n_jobs=-1)
	
	return scores

#build parameters for next random forest 
def buildparams(i, j, k, l):

	#handle special None case for max_depth and max_samples
	current_depth = ''
	if j == None:
		current_depth = None
	else:
		current_depth = j
		
	current_samples=''
	if l >= 1.0:
		current_samples = None
	else:
		current_samples = round(l, 1)
	
	#construct best parameters to initiate the random forest with 
	params = {"n_estimators": i,
			  "max_depth": current_depth,
			  "max_features": k,
			  "max_samples": current_samples}
		
	return params

def fulloptimize(X, y):
	
	#will contain results for all combinations
	results = {}
	
	#values to iterate over
	i_trees = [50, 100, 200, 500, 1000]  
	depths = [i for i in range(2,9)] + [None] 
	
	#for every combination
	for i in tqdm(i_trees, leave=False):
		for j in tqdm(depths, leave=False):
			for k in tqdm(range(1, 12), leave=False): 
				for l in tqdm(np.arange(0.2, 1.1, 0.1), leave=False): 
					
					current_params = buildparams(i, j, k, l)
					
					#new model with current parameters
					current_model = RandomForestClassifier(**current_params, random_state=seed)
					
					#evaluate model
					score = evaluate_performance(current_model, X, y)
					
					if l >= 1.0:
						score_l = None
					else:
						score_l = round(l, 1)
					
					#add score to results
					results[str(i) + ' ' + str(j) + ' ' + str(k) + ' ' + str(score_l)] = np.mean(score)
	
	print(results)
	
	#find best f1 in all combinations
	best_f1_key = max(results, key=results.get)	
	best_f1 = results.get(best_f1_key)
	print(best_f1_key)
	print(best_f1)
	
	#plit best_f1_key string at whitespaces
	parameters = best_f1_key.split()
	best_estimators = parameters[0]
	best_depth = parameters[1]
	best_features = parameters[2]
	best_samples = parameters[3]
	
	#handle special None case for max_depth and max_samples
	temp_depth = ''
	if best_depth == 'None':
		temp_depth = None
	else:
		temp_depth = int(best_depth)
		
	temp_samples=''
	if best_samples == 'None':
		temp_samples = None
	else:
		temp_samples = float(best_samples)
	
	#construct best parameters to initiate the random forest with 
	params = {"n_estimators": int(best_estimators),
			  "max_depth": temp_depth,
			  "max_features": int(best_features),
			  "max_samples": temp_samples}
	
	return params
#random forest parameter optimization
def fastoptimize(X, y):
	
	#functions to iterate over
	functions = [get_models_trees, get_models_depth, get_models_features, get_models_samples]
		
	#lists to save results in
	estimators = {}
	depths = {}
	features = {}
	samples = {}

	i=1
	#for ever different parameter test case
	for fn in tqdm(functions, leave=False):
		
		#get models with different parameter values
		models = fn()
		
		#evaluate models
		for name, model in tqdm(models.items(), leave=False):
			score = evaluate_performance(model, X, y)
			if i==1:
				estimators[name] = np.mean(score)
			elif i==2:
				depths[name] = np.mean(score)
			elif i==3:
				features[name] = np.mean(score)
			elif i==4:
				samples[name] = np.mean(score)
			#print('>%s %.3f (%.3f)' % (name, np.mean(score), np.std(score)))
			
		i += 1
	
	#handle special None case for max_depth and max_samples
	best_depth = ''
	if max(depths, key=depths.get) == 'None':
		best_depth = None
	else:
		best_depth = int(max(depths, key=depths.get))
		
	best_samples=''
	if max(samples, key=samples.get) == 'None':
		best_samples = None
	else:
		best_samples = float(max(samples, key=samples.get))
	
	#construct best parameters to initiate the random forest with 
	params = {"n_estimators": int(max(estimators, key=estimators.get)),
			  "max_depth": best_depth,
			  "max_features": int(max(features, key=features.get)),
			  "max_samples": best_samples}
			  			  
	return params
	
#find best parameters for surrogate model decision tree
def treeoptimize(X, y):
	
	model = DecisionTreeClassifier(random_state=seed)
	
	#set the different values to check
	max_depth = [3,4,5,6,7]
	parameters = {'max_depth':max_depth}
	
	#use repeated stratified k-fold
	cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=seed)
	
	#find best parameters
	findbest = GridSearchCV(model, parameters, cv=cv, scoring='f1')
	findbest.fit(X, y)
	
	print(colored('Best parameters for decision tree initiation:', 'cyan'))
	print('max_depth:', findbest.best_params_["max_depth"])
	
	params = {
			  "max_depth": findbest.best_params_["max_depth"]
			 }		
		
	return params

#calculate accuracy, precision, recall, specificity, f1, and mcc scores of a given model
def evaluate_model(model, X, y, mode, datasetname, evaltype):
	
	if evaltype == 'None':
		if datasetname == 'toy':
			
			#construct best parameters to initiate the random forest with (standard parameters for random forests yield perfect metrics)
			best_params = {"n_estimators": 100,
						   "max_depth": None,
						   "max_features": "sqrt",
						   "max_samples": None}
						   
			return None, None, None, None, None, None, best_params
			
		elif datasetname == 'heart':
			
			#construct best parameters to initiate the random forest with (found by fulloptimize())
			best_params = {"n_estimators": 500,
						   "max_depth": None,
						   "max_features": 10,
						   "max_samples": 0.6}
		
			return None, None, None, None, None, None, best_params
			
		elif datasetname == 'diabetes':
			
			#construct best parameters to initiate the random forest with (found by fulloptimize())
			best_params = {"n_estimators": 1000,
						   "max_depth": None,
						   "max_features": 1,
						   "max_samples": 0.9}
		
			return None, None, None, None, None, None, best_params
	
	#data preprocessing
	#transform = MinMaxScaler()
	#pipeline = Pipeline(steps=[('t', transform), ('m', model)])
	
	if mode == 'class':
		#custom scorer for TNs(specificity) and MCC(Matthews correlation coefficient)
		specificity = make_scorer(recall_score, pos_label=0)
		matthews = make_scorer(matthews_corrcoef)
		
		#define which metrics to calculate
		scoring = {'acc': 'accuracy',
				   'prec': 'precision',
				   'rec': 'recall',
				   'spec': specificity,
				   'f1': 'f1',
				   'mcc': matthews}
		
		#n_repeats different 80/20 train/test splits
		cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=seed)

	else:
		#calculate r2 for regression
		scoring = {'r2': 'r2'}
		
		#n_repeats different 80/20 train/test splits
		cv = RepeatedKFold(n_splits=5, n_repeats=10, random_state=seed)
	
	#calculate accuracy, precision, recall, specificity, f1, and mcc scores
	scores = cross_validate(model, X, y, scoring=scoring, cv=cv, n_jobs=-1)

	#calculate mean and std
	if mode == 'class':
		acc = [scores['test_acc'].mean(), scores['test_acc'].std()]
		prec = [scores['test_prec'].mean(), scores['test_prec'].std()]
		rec = [scores['test_rec'].mean(), scores['test_rec'].std()]
		spec = [scores['test_spec'].mean(), scores['test_spec'].std()]
		f1 = [scores['test_f1'].mean(), scores['test_f1'].std()]
		mcc = [scores['test_mcc'].mean(), scores['test_mcc'].std()]
		print(colored('Performance metrics before parameter optimization:', 'cyan'))
		print('Accuracy mean:   ', format(acc[0], '.4f'), 'Accuracy std:   ', format(acc[1], '.4f'),
			  '\nPrecision mean:  ', format(prec[0], '.4f'), 'Precision std:  ', format(prec[1], '.4f'),
			  '\nRecall mean:     ', format(rec[0], '.4f'), 'Recall std:     ', format(rec[1], '.4f'),
			  '\nSpecificity mean:', format(spec[0], '.4f'), 'Specificity std:', format(spec[1], '.4f'),
			  '\nF1 mean:         ', format(f1[0], '.4f'), 'F1 std:         ', format(f1[1], '.4f'),
			  '\nMCC mean:        ', format(mcc[0], '.4f'), 'MCC std:        ', format(mcc[1], '.4f'))
			  
		
		if evaltype == 'fast':
			
			#find better but not best parameters in a short time
			best_params = fastoptimize(X, y)
			
		elif evaltype == 'full':
			
			#find best parameters trying all combinations (takes long!)
			best_params = fulloptimize(X, y)
			
		elif evaltype == 'usebest':
			if datasetname == 'toy':
				
				#construct best parameters to initiate the random forest with (standard parameters for random forests yield perfect metrics)
				best_params = {"n_estimators": 100,
							   "max_depth": None,
							   "max_features": "sqrt",
							   "max_samples": None}
				
			elif datasetname == 'heart':
				
				#construct best parameters to initiate the random forest with (found by fulloptimize())
				best_params = {"n_estimators": 500,
							   "max_depth": None,
							   "max_features": 10,
							   "max_samples": 0.6}
							   
			elif datasetname == 'diabetes':
				
				#construct best parameters to initiate the random forest with (found by fulloptimize())
				best_params = {"n_estimators": 1000,
							   "max_depth": None,
							   "max_features": 1,
							   "max_samples": 0.9}
							   
		elif evaltype == 'tree':
			
			#find best parameters for surrogate model decision tree
			best_params = treeoptimize(X, y)
			
		if evaltype != 'tree':			  
			print(colored('Best parameters for random forest initiation:', 'cyan'))
			print('n_estimators:', best_params.get('n_estimators'),
				  '\nmax_depth:   ', best_params.get('max_depth'),
				  '\nmax_features:', best_params.get('max_features'),
				  '\nmax_samples: ', best_params.get('max_samples'))
			
			#evaluate model with optimized parameters
			better_model = RandomForestClassifier(**best_params, random_state=seed)
			
		else:
			
			#evaluate model with optimized parameters
			better_model = DecisionTreeClassifier(**best_params, random_state=seed)
			
		#data preprocessing
		#transform2 = MinMaxScaler()
		#pipeline2 = Pipeline(steps=[('t', transform2), ('m', better_model)])
		
		#calculate accuracy, precision, recall, specificity, f1, and mcc scores
		better_scores = cross_validate(better_model, X, y, scoring=scoring, cv=cv, n_jobs=-1)
		
		#calculate mean and std for better_scores
		better_acc = [better_scores['test_acc'].mean(), better_scores['test_acc'].std()]
		better_prec = [better_scores['test_prec'].mean(), better_scores['test_prec'].std()]
		better_rec = [better_scores['test_rec'].mean(), better_scores['test_rec'].std()]
		better_spec = [better_scores['test_spec'].mean(), better_scores['test_spec'].std()]
		better_f1 = [better_scores['test_f1'].mean(), better_scores['test_f1'].std()]
		better_mcc = [better_scores['test_mcc'].mean(), better_scores['test_mcc'].std()]
		print(colored('Performance metrics after parameter optimization:', 'cyan'))
		print('Accuracy mean:   ', format(better_acc[0], '.4f'), 'Accuracy std:   ', format(better_acc[1], '.4f'),
			  '\nPrecision mean:  ', format(better_prec[0], '.4f'), 'Precision std:  ', format(better_prec[1], '.4f'),
			  '\nRecall mean:     ', format(better_rec[0], '.4f'), 'Recall std:     ', format(better_rec[1], '.4f'),
			  '\nSpecificity mean:', format(better_spec[0], '.4f'), 'Specificity std:', format(better_spec[1], '.4f'),
			  '\nF1 mean:         ', format(better_f1[0], '.4f'), 'F1 std:         ', format(better_f1[1], '.4f'),
			  '\nMCC mean:        ', format(better_mcc[0], '.4f'), 'MCC std:        ', format(better_mcc[1], '.4f'))

		return better_acc, better_prec, better_rec, better_spec, better_f1, better_mcc, best_params

	else:
		r2 = [scores['test_r2'].mean(), scores['test_r2'].std()]
		print(colored('Performance metrics before parameter optimization:', 'cyan'))
		print('r2: ', r2[0], r2[1])
		
		return r2
		
#make predictions with model
def make_prediction(model, datapoint):
	
	prediction=model.predict([datapoint])

	return prediction

