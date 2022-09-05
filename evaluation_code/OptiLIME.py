from stability_utils import Sklearn_Lime
from utils import bayesian_optimisation
import numpy as np
from tqdm import tqdm
import pandas as pd
from normalLIME import singleLIME
from termcolor import colored
import sys
from stability import calc_vsi_lime, calc_fsi_lime

#set seed for randomness
def set_seed(seed):
	np.random.RandomState(seed)
	np.random.seed(seed)
	
seed = 665
set_seed(seed)

#global variables for OptiLIME_loss optimization function
cat_feat = ""
feat_names = ""
cl_names = ""
maxR = ""
mod = ""
ute = ""
X_test = ""
X_train = ""
num_feat = ""


#OptiLIME optimization function for regression problems
def OptiLIME_loss_reg(kernel_width):
	single_lime = Sklearn_Lime(mode="regression",
							   verbose=False,
							   discretize_continuous=False,
							   kernel_width=kernel_width,
							   sample_around_instance=True,
							   penalty=0,
							   epsilon=None,
							   num_samples=5000,
							   maxRsquared=maxR,
							   random_state=seed,
							   feature_names=feat_names,
							   class_names=cl_names,
							   categorical_features=cat_feat,
							   num_features=num_feat
							  )
	single_lime.fit(X_train)
	return single_lime.score(ute, mod.predict)

#OptiLIME optimization function for classification problems	
def OptiLIME_loss_class(kernel_width):	
	single_lime = Sklearn_Lime(mode="classification",
							   verbose=False,
							   discretize_continuous=False,
							   kernel_width=kernel_width,
							   sample_around_instance=True,
							   penalty=0,
							   epsilon=None,
							   num_samples=5000,
							   maxRsquared=maxR,
							   random_state=seed,
							   feature_names=feat_names,
							   class_names=cl_names,
							   categorical_features=cat_feat,
							   num_features=num_feat
							  )
	single_lime.fit(X_train)
	return single_lime.score(ute, mod.predict_proba)

#get most stable lime explanation for chosen adherence
def OptiLIME(train, test, model, toexplain, names, dataset, maxRsquared, num_features):
	
	#find categorical features
	categorical_features = np.argwhere(np.array([len(set(train[:,x])) for x in range(train.shape[1])]) <= 10).flatten()
	
	#split feature and class names
	if names != '':
		feature_names = names[:-1]
		class_names = np.array([names[-1]])

	#set global variables
	global cat_feat
	cat_feat = categorical_features
	global feat_names
	feat_names = feature_names
	global cl_names
	cl_names = class_names
	global maxR
	maxR = maxRsquared
	global mod
	mod = model
	global ute
	ute = toexplain
	global X_test
	X_test = test
	global X_train
	X_train = train
	global num_feat
	num_feat = num_features
	
	#set range of kernel width to optimize over
	bounds = np.array([0.3, 5]).reshape(1, -1)

	#use bayesian optimisation to find optimal kernel width
	if dataset in ('boston', 'optilimetest'):
		kw, R_squared = bayesian_optimisation(n_iters=40,
											  sample_loss=OptiLIME_loss_reg,
											  bounds=bounds,
											  n_pre_samples=20,
											  random_search=100,									
											 )	
	elif dataset in ('random', 'toy', 'heart', 'diabetes'):
		kw, R_squared = bayesian_optimisation(n_iters=40,
											  sample_loss=OptiLIME_loss_class,
											  bounds=bounds,
											  n_pre_samples=20,
											  random_search=100
											 )
											 
	#prints for debugging	
	#print('kw: ', kw)
	#print('R_squared: ', R_squared)	

	#search for best kernel width and the corresponding R_squared
	best_width = float(kw[R_squared.argmax()])
	best_R_squared = R_squared.max()
	
	#print out best kw for the chosen adherence
	print(colored('Best kernel width to get closest to the demanded adherence factor of', 'cyan'), colored(maxRsquared, 'red'), colored('for the LIME model:', 'cyan'))
	print('Best kernel width:', round(best_width, 6))
	print('Reached R_squared:', round(best_R_squared, 6))	

	#collect results into one datastructure						  
	results = pd.concat([
						pd.DataFrame(kw, columns=["kernel_width"]),
						pd.DataFrame(R_squared, columns=["R_squared"])
						], axis=1)


	#set parameters for lime explanation
	params = {"data_row": ute,
			  "predict_fn": model.predict_proba,
			  "num_samples": 5000,
			  "num_features": num_feat,
			  "distance_metric": "euclidean"}
			  	
	#get explanation and stability for best kernel_width
	single_lime = Sklearn_Lime(mode="classification",
							   verbose=False,
							   discretize_continuous=False,
							   kernel_width=best_width,
							   sample_around_instance=True,
							   penalty=0,
							   epsilon=None,
							   num_samples=5000,
							   maxRsquared=maxRsquared,
							   random_state=seed,
							   feature_names=feat_names,
							   class_names=cl_names,
							   categorical_features=cat_feat,
							   num_features=num_feat
							   )
	
	#fit training data to lime instance
	single_lime.fit(X_train)
	
	#get lime explanation
	exp = single_lime.my_lime.explain_instance(**params)
	
	#get more lime explanations to calculate vsi/fsi
	exps = []
	
	#generate lime models to calculate fsi/vsi
	temp_single_lime = Sklearn_Lime(mode="classification",
								    verbose=False,
								    discretize_continuous=False,
								    kernel_width=best_width, 
								    sample_around_instance=True,
								    penalty=0,
								    epsilon=None,
								    num_samples=5000,
								    maxRsquared=maxRsquared,
								    random_state=seed,
								    feature_names=feat_names,
								    class_names=cl_names,
								    categorical_features=cat_feat,
								    num_features=num_feat
								    )
	#fit training data to lime instance
	temp_single_lime.fit(X_train)						
	
	#generate more lime explanations
	for i in range(10):
			
		#get lime explanation
		temp_exp = temp_single_lime.my_lime.explain_instance(**params)

		#gather explanations
		exps.append(temp_exp)
		
	#calculate fsi and csi 
	single_fsi, feature_stability = calc_fsi_lime(exps)
	single_vsi = calc_vsi_lime(exps)
				
	return exp, best_width, best_R_squared, single_fsi, single_vsi, feature_stability, results, exp.predict_proba
	
#calculate csi/vsi scores for all kws used in the optimization (takes time!)
def getOptiLIMEplotData(results, dataset):
		
	#arrays to store the csi/vsi values in
	csi_vals = []
	vsi_vals = []
	
	#calculate csi/vsi values for every kernel width checked by the bayesian optimisation
	if dataset in ('boston', 'optilimetest'):
		for kernel_width in tqdm(results.kernel_width, leave=False, file=sys.stdout):
			
			#create one lime instance for every kernel width
			single_lime = Sklearn_Lime(mode="regression",
									   verbose=False,
									   discretize_continuous=False,
									   kernel_width=kernel_width,
									   sample_around_instance=True,
									   penalty=0,
									   epsilon=None,
									   num_samples=5000,
									   maxRsquared=maxR,
									   random_state=seed,
									   feature_names=feat_names,
									   class_names=cl_names,
									   categorical_features=cat_feat,
									   num_features=num_feat
									   )
			
			#fit the training data
			single_lime.fit(X_train)
			
			#check stability over 10 calls
			csi, vsi, _, exps = single_lime.my_lime.check_stability(ute, mod.predict, num_features=num_feat)
			
			#gather results
			vsi_vals.append(vsi)
			csi_vals.append(csi)
	
	#same for random, toy and heart failure datasets
	elif dataset in ('random', 'toy', 'heart', 'diabetes'):
		for kernel_width in tqdm(results.kernel_width, leave=False, file=sys.stdout):
			single_lime = Sklearn_Lime(mode="classification",
									   verbose=False,
									   discretize_continuous=False,
									   kernel_width=kernel_width,
									   sample_around_instance=True,
									   penalty=0,
									   epsilon=None,
									   num_samples=5000,
									   maxRsquared=maxR,
									   random_state=seed,
									   feature_names=feat_names,
									   class_names=cl_names,
									   categorical_features=cat_feat,
									   num_features=num_feat
									   )
									   
			single_lime.fit(X_train)
			
			csi, vsi, _, exps = single_lime.my_lime.check_stability(ute, mod.predict_proba, num_features=num_feat)
			
			vsi_vals.append(vsi)
			csi_vals.append(csi)
	
	#bring down csi/vsi values into 0..1 range
	vsi_vals = [vsi/100 for vsi in vsi_vals]
	csi_vals = [csi/100 for csi in csi_vals]
	
	#add csi/vsi values to results
	results = pd.concat([
						results,
						pd.DataFrame(csi_vals,columns=["csi"]),
						pd.DataFrame(vsi_vals,columns=["vsi"])
						], axis=1)
	
	#sort results for kernel width
	results.sort_values("kernel_width", inplace=True)
	
	return results
	
