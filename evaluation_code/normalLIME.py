#from stability_utils import LimeTabularExplainerOvr
from stability import calc_vsi_lime, calc_fsi_lime
import numpy as np
import lime
import lime.lime_tabular

#set seed for randomness
def set_seed(seed):
	np.random.RandomState(seed)
	np.random.seed(seed)
	
seed = 665
set_seed(seed)

#get a normal lime explanation and calculate fsi/vsi
def singleLIME(train, test, model, toexplain, names, dataset, num_features):
	
	#find categorical features
	categorical_features = np.argwhere(np.array([len(set(train[:,x])) for x in range(train.shape[1])]) <= 10).flatten()
	
	#split feature and class names
	if names != '':
		feature_names = names[:-1]
		class_names = np.array([names[-1]])
	
	#define lime explainer
	explainer = lime.lime_tabular.LimeTabularExplainer(train, 
													   feature_names=feature_names,
													   class_names=class_names,
													   categorical_features=categorical_features,
													   discretize_continuous=False
													   )
	
	#explain the given instance
	exp = explainer.explain_instance(toexplain, model.predict_proba, num_features=num_features)
	
	#i more explanations to evaluate the stability
	exps = [exp]
	for i in range(9):
		#exp_temp = explainer.explain_instance(**params)
		exp_temp = explainer.explain_instance(toexplain, model.predict_proba, num_features=num_features)
		exps.append(exp_temp)
	
	#get vsi score
	vsi = calc_vsi_lime(exps)
	
	#get fsi score
	fsi, feature_stability = calc_fsi_lime(exps)
	
	return exp, fsi, vsi, exp.predict_proba, feature_stability
