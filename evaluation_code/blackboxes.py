import sys
import os
import getopt
from dataload import getdata, loadpickle, generate_samples
from trainmodel import trainmodel
from normalLIME import singleLIME
from OptiLIME import OptiLIME, getOptiLIMEplotData
from modelevaluation import make_prediction
from ploter import plot_importance, plot_tree, plotstability
from termcolor import colored
from joblib import dump, load
from pathlib import Path
import numpy as np
from tqdm import tqdm
import pickle
from sklearn import metrics
from stability import tree_stability
import pandas as pd

#handle inputs
def main(argv):
	arg_help = ("\n{0} -d <dataset> -m <model> -t <type> -u <ute> -a <adherence> \n"
				"dataset: toy/heart/diabetes/random/boston/optilimetest model: rfr/rfc/linear \n"
				"type: fast/full/usebest/None (type of evaluation) ute: toy/heart/diabetes (unit to explain with LIME) \n"
				"adherence: float 0 < a <= 1 (adherence of the LIME model) \n\n"
				"Or {0} -p <type> -f <filename> to plot tree (type: calc=calculator example, toy=toy dataset, heart=heart failure dataset, diabetes=diabetes data set, toysur=toysurrogate, heartsur=heart surrogate, all=plot all files in /plot folder / name: file name in plot folder or 'all' when type is 'all') \n\n"
				"Or {0} -b <set> train blackbox and save as .joblib file (set: toy=toy data set, heart=heart failure data set, diabetes=diabetes data set) \n\n"
				"Or {0} -g <set> generate sample files to train and test surrogate models with (set: toy=toy data set, heart=heart failure data set, diabetes=diabetes data set) \n\n"
				"Or {0} -s <model> -n <name> Generate a global surrogate model for a black box model (model: toy=toy data set, heart=heart failure data set diabetes=diabetes data set/ name: name of samples folder) \n\n"
				"Or {0} -e <surrogate> -o <name> evaluate performance of a surrogate model (surrogate: name file/folder of the surrogate model/models / name: name of the test sample set to use or 'csv' to use .csv files in surrogate/<surrogate>/performance folder) \n\n"
				"Or {0} -c <surrogates> evaluate stability of over multiple surrogate models (surrogates: folder name of the surrogate models in /surrogate)\n".format(argv[0])
				)
	arg_data = ""
	arg_model = ""
	arg_plot = ""
	arg_plotpath = ""
	arg_unit = ""
	arg_adherence = ""
	arg_blackbox = ""
	arg_evaltype = ""
	arg_generate = ""
	arg_surrogate = ""
	arg_name = ""
	arg_evaluate = ""
	arg_open = ""
	arg_compare = ""
	
	try:
		opts, args = getopt.getopt(argv[1:], "hp:f:b:d:m:t:u:a:g:s:n:e:o:c:", ["help", "plot=", "file=", "blackbox=", "data=", "model=", "evaltype=", "unittoexplain=", "adherence=", "generate=", "surrogate=", "name=", "evaluate=", "open=", "compare="])
	except:
		print(arg_help)
		sys.exit(2)
	
	for opt, arg in opts:
		if opt in ("-h", "--help"):
			print(arg_help)
			sys.exit(2)
		elif opt in ("-d", "--dataset"):
			if arg in ("toy", "heart", "diabetes", "random", "boston", "optilimetest"):
				arg_data = arg
			else:
				print('Dataset has to be in: "toy", "heart", "diabetes", "random", "boston", "optilimetest"')
				sys.exit(2)
		elif opt in ("-m", "--model"):
			if arg in ("linear", "rfr", "rfc"):
				arg_model = arg
			else:
				print('Model has to be in: "linear", "rfr", "rfc"')
				sys.exit(2)
		elif opt in ("-t", "--type"):
			if arg in ("fast", "full", "usebest", "None"):
				arg_evaltype = arg
			else:
				print('Type has to be in: "fast", "full", "usebest", "None"')
				sys.exit(2)
		elif opt in ("-u", "--unittoexplain"):
			if arg in ["toy", "heart", "diabetes"]:
				arg_unit = arg
			else:
				print('Unit to explain has to be in: "toy", "heart", "diabetes"')
				sys.exit(2)
		elif opt in ("-a", "--adherence"):
			if float(arg) > 0 and float(arg) <= 1:
				arg_adherence = float(arg)
			else:
				print('Adherence has to be float: 0 < a <= 1.0')
				sys.exit(2)
		elif opt in("-p", "--plot"):
			if arg in ("calc", "toy" , "heart", "diabetes", "toysur", "heartsur", "all"):
				arg_plot = arg
			else:
				print('Type has to be in: "calc", "toy", "heart", "diabetes", "toysur", "heartsur", "all"')
				sys.exit(2)
		elif opt in("-f", "--filename"):
			if os.path.exists("plot/"+arg) == True:
				arg_plotpath = "plot/"+arg
				break
			elif arg == 'all':
				arg_plotpath = "all"
				break
			else:
				print('File'+str(arg)+'not found in /plot folder!')
				sys.exit(2)			
		elif opt in ("-b", "--blackbox"):
			if arg == 'toy' or arg == 'heart' or arg == 'diabetes':
				arg_blackbox = arg
				break
			else:
				print('Blackbox to train and save as .joblib has to be \'toy\' or \'heart\' or \'diabetes\' (toy: toy data set, heart: heart failure data set, diabetes: diabetes data set)')
				sys.exit(2)
		elif opt in ("-g", "--generate"):
			if arg == 'toy' or arg == 'heart' or arg == 'diabetes':
				arg_generate = arg
				break
			else:
				print('Set has to be in: "toy", "heart", "diabetes"')
				sys.exit(2)
		elif opt in ("-s", "--surrogate"):
			if arg == 'toy' or arg == 'heart' or arg == 'diabetes':
				arg_surrogate = arg
			else:
				print('Model has to be in: "toy", "heart", "diabetes"')
				sys.exit(2)
		elif opt in ("-n", "--name"):
			if os.path.isdir('samples/'+arg):
				arg_name = arg
				break
			else:
				print('No folder "'+arg+'" in /plot!')
				sys.exit(2)
		elif opt in ("-e", "--evaluate"):
			if (os.path.isfile('surrogate/'+arg) == True) or (os.path.isdir('surrogate/'+arg) == True):
				arg_evaluate = arg
			else:
				print('No model file/folder in /surrogate with the name "'+arg+'"!')
				sys.exit(2)
		elif opt in ("-o", "--open"):
			if (os.path.isdir('samples/'+arg) == True) or (arg == 'csv' and os.path.isdir('surrogate/'+arg_evaluate+'/performance') == True):
				arg_open = arg
				break
			else:
				print('No folder in /samples with name "'+arg+'"!')
				sys.exit(2)
		elif opt in ("-c", "--compare"):
			if os.path.isdir('surrogate/'+arg) == True:
				arg_compare = arg
				break
			else:
				print('No folder in /surrogate with name "'+arg+'"!')
				sys.exit(2)
								
	if ((arg_plot != '' and arg_plotpath != '' and arg_blackbox == '' and arg_generate == '' and arg_surrogate == '' and arg_name == '' and arg_evaluate == '' and arg_open == '' and arg_compare == '' and (arg_data == '' and arg_model == '' and arg_evaltype == '' and arg_unit == '' and arg_adherence == '')) or
		(arg_plot == '' and arg_plotpath == '' and arg_blackbox == '' and arg_generate == '' and arg_surrogate == '' and arg_name == '' and arg_evaluate == '' and arg_open == '' and arg_compare == '' and (arg_data != '' and arg_model != '' and arg_evaltype != '' and arg_unit != '' and arg_adherence != '')) or
		(arg_plot == '' and arg_plotpath == '' and arg_blackbox != '' and arg_generate == '' and arg_surrogate == '' and arg_name == '' and arg_evaluate == '' and arg_open == '' and arg_compare == '' and (arg_data == '' and arg_model == '' and arg_evaltype == '' and arg_unit == '' and arg_adherence == '')) or
		(arg_plot == '' and arg_plotpath == '' and arg_blackbox == '' and arg_generate != '' and arg_surrogate == '' and arg_name == '' and arg_evaluate == '' and arg_open == '' and arg_compare == '' and (arg_data == '' and arg_model == '' and arg_evaltype == '' and arg_unit == '' and arg_adherence == '')) or
		(arg_plot == '' and arg_plotpath == '' and arg_blackbox == '' and arg_generate == '' and arg_surrogate != '' and arg_name != '' and arg_evaluate == '' and arg_open == '' and arg_compare == '' and (arg_data == '' and arg_model == '' and arg_evaltype == '' and arg_unit == '' and arg_adherence == '')) or
		(arg_plot == '' and arg_plotpath == '' and arg_blackbox == '' and arg_generate == '' and arg_surrogate == '' and arg_name == '' and arg_evaluate == '' and arg_open == '' and arg_compare != '' and (arg_data == '' and arg_model == '' and arg_evaltype == '' and arg_unit == '' and arg_adherence == '')) or	
		(arg_plot == '' and arg_plotpath == '' and arg_blackbox == '' and arg_generate == '' and arg_surrogate == '' and arg_name == '' and arg_evaluate != '' and arg_open != '' and arg_compare == '' and (arg_data == '' and arg_model == '' and arg_evaltype == '' and arg_unit == '' and arg_adherence == ''))):		
		return arg_plot, arg_plotpath, arg_blackbox, arg_data, arg_model, arg_evaltype, arg_unit, arg_adherence, arg_generate, arg_surrogate, arg_name, arg_evaluate, arg_open, arg_compare
	else:
		print(arg_help)
		sys.exit(2)
	
if __name__ == "__main__":
	arg_plot, arg_plotpath, arg_blackbox, data_arg, model_arg, arg_evaltype, arg_unit, arg_adherence, arg_generate, arg_surrogate, arg_name, arg_evaluate, arg_open, arg_compare = main(sys.argv)
	
if arg_plot == "" and arg_blackbox == "" and arg_generate == "" and arg_surrogate == "" and arg_evaluate == "" and arg_compare == "":
	#get database for the blackbox (data_arg: 'toy'(toy dataset) / 'heart'(heart failure dataset) / 'diabetes'(diabetes dataset) / 'random'(random data set) / 'boston'(boston data set) / 'optilimetest'(test data for optilime))
	print(colored('Fetching and preparing database...', 'green'))
	X, y, names = getdata(data_arg)

	#train on best model parameters and evaluate (model_arg: 'rfr'(random forest regressor) / 'rfr'(random forest classifier) / 'linear'(linear model for random dataset), 
	#data_arg: 'toy'(toy dataset) / 'heart'(heart failure dataset) / 'diabetes'(diabetes dataset) / 'random'(random data set) / 'boston'(boston data set) / 'optilimetest'(test data for optilime)
	#arg_evaltype: 'fast', 'full' (takes time!), 'usebest' (use previous best found with 'full' and randomstate 665))
	print(colored('Optimizing model parameters and training random forest...', 'green'))
	accuracy, precision, recall, spec, f1, mcc, model, train, test, y_test = trainmodel(X, y, model_arg, data_arg, arg_evaltype)
	
	#which input to explain and how many features to consider
	if arg_unit == "toy":
		toexplain = np.array([1,0,0,0,0,0,0,0,0,0])
	elif arg_unit == "heart":
		toexplain = np.array([72,1,110,0,25,0,274000,1,140,1,1])
	elif arg_unit == "diabetes":
		toexplain = np.array([70,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1])
	num_features = 5
	
	#get a normal lime explanation and calculate csi/vsi
	print(colored('Get a single LIME explanation with default settings and calculate stability indices...', 'green'))
	exp, fsi, vsi, predicted, feature_stability = singleLIME(train, test, model, toexplain, names, data_arg, num_features)
	
	#print LIME explanation
	print(colored('Standard LIME explanation for feature set:\n', 'cyan'), colored(toexplain, 'red'), colored('->', 'red'), colored(predicted, 'red') )
	#print(colored(test[toexplain], 'red'), colored('->', 'red'), colored(predicted, 'red'))
	print(f"{'Feature Names':<30}{'Importance':<30}{'Stability':<30}{'in Iteration'}")
	for i in exp.as_list():
		for x in feature_stability:
			if i[0] in x:
				print(f"{'%s:' % i[0]:<30}{round(i[1], 4) :<30}{round(x[3]*100, 2):<30}{x[4]}")
				break
				
	#print LIME explanation stability
	print(colored('Standard LIME explanation stability:', 'cyan'))
	print("FSI: ", fsi, "\nVSI: ", vsi)
	
	#the adherence target for OptiLIME
	maxRsquared = arg_adherence
	
	#get an explanation with OptiLIME
	print(colored('Use OptiLIME to optimize the kernel width to target an adherence factor of', 'green'), colored(maxRsquared, 'red'), colored('for the LIME model...', 'green'))
	optiexp, best_width, best_R_squared, best_fsi, best_vsi, opti_feature_stability, results, optipredict = OptiLIME(train, test, model, toexplain, names, data_arg, maxRsquared, num_features)
	
	#print LIME explanation for targeted adherence (best kernel width calculated by OptiLIME)
	print(colored('LIME explanation (for targeted adherence) for feature set:\n', 'cyan'), colored(toexplain, 'red'), colored('->', 'red'), colored(optipredict, 'red'))
	print(f"{'Feature Names':<30}{'Importance':<30}{'Stability':<30}{'in Iteration'}")
	#for i in optiexp.as_list():
		#print(f"{'%s:' % i[0]:<30}{round(i[1], 4) }")
	for i in optiexp.as_list():
		for x in opti_feature_stability:
			if i[0] in x:
				print(f"{'%s:' % i[0]:<30}{round(i[1], 4) :<30}{round(x[3]*100, 2):<30}{x[4]}")
				break

	print(colored('FSI and VSI values for chosen kernel width', 'cyan'), colored(round(best_width, 4), 'red'), colored('to get closest to the demanded adherence:', 'cyan'))
	print("FSI: ", best_fsi, "\nVSI: ", best_vsi)

	#calculate csi/vsi scores for all kws used in the optimization (takes time!)
	print(colored('Calculating CSI and VSI scores for all kernel widths used in the bayesian optimisation...', 'green'))
	allresults = getOptiLIMEplotData(results, data_arg)
	
	#plot stability with different kernel widths (only if possible if getOptiLIMEplotData is called before)
	print(colored('Plot stability and adherence of LIME for different kernel widths...', 'green'))
	plotstability(best_width, best_R_squared, best_fsi, best_vsi, allresults, maxRsquared, data_arg, arg_adherence)
	
	#make a prediction with the model
	#make_prediction(model, test[6])
	
	#plot feature importance of the model
	#plot_importance(model, names)
	
#plot tree
elif arg_plot != "":
	if arg_plot == 'all':
		
		#find relevant files in /plot
		files = os.listdir("plot/")
		paths = []
		types = []
		for x in files:
			if (x[-6:] in ['joblib', 'pickle']):
				
				#check what dataset tree is based on (calc, toy, heart, diabetes, toy+surrogate or heart+surrogate, diabetes+surrogate should be contained in name)				
				if 'calc' in x:
					paths.append("plot/"+x)
					types.append('calc')
				elif ('toy' in x) and ('surrogate' in x):
					paths.append("plot/"+x)
					types.append('toysur')
				elif ('heart' in x) and ('surrogate' in x):
					paths.append("plot/"+x)
					types.append('heartsur')
				elif ('diabetes' in x) and ('surrogate' in x):
					paths.append("plot/"+x)
					types.append('diabetessur')
				elif 'toy' in x:
					paths.append("plot/"+x)
					types.append('toy')
				elif 'heart' in x:
					paths.append("plot/"+x)
					types.append('heartfull')
				elif 'diabetes' in x:
					paths.append("plot/"+x)
					types.append('diabetesfull')
		
		models = ['' for _ in range(len(paths))] 
		names = ['' for _ in range(len(paths))]
		

		print(colored('Load decision trees from all files and save plots as pngs...', 'green'))		
		for i in range(len(paths)):
			#load decision tree from .pickle/.joblib files
			models[i], names[i] = loadpickle(paths[i], types[i])	
		
			#save decision tree plot to png
			plot_tree(models[i], paths[i], names[i])		
			
		#delete .pickel/.joblib files
		for x in paths:
			os.remove(x)
			
	else:
		
		#load decision tree from .pickle/.joblib file
		print(colored('Load decision tree from .pickle/.joblib file...', 'green'))
		model, names = loadpickle(arg_plotpath, arg_plot)
		
		#save decision tree plot to png
		print(colored('Save decision tree plot to png...', 'green'))
		plot_tree(model, arg_plotpath, names)
		
		#delete .pickel/.joblib file		
		os.remove(x)
				
#train and save blackbox to file
elif arg_blackbox != "":
	
	#train random forest and create .joblib file from toy dataset
	if arg_blackbox == 'toy':
		
		#get database for the blackbox
		print(colored('Fetching and preparing database...', 'green'))
		X, y, names = getdata('toy')
				
		#train on best model parameters
		print(colored('Training random forest with optimal model parameters...', 'green'))
		_, _, _, _, _, _, model, _, test, _ = trainmodel(X, y, 'rfc', 'toy', 'None')
		
		#test code
		#make a prediction with the model
		#print(colored('Make prediction with the model...', 'green'))
		#for i in range (0,10):
			#print('Predicted Class:', make_prediction(model, test[i])[0])
		
		#save model as .joblib file
		print(colored('Saving random forest model as .joblib file...', 'green'))
		dump(model, Path(__file__).parent / "blackboxes" / "toy.joblib")

		#test code
		#make a prediction with the model after loading from .joblib
		#print(colored('Make prediction with the model after loading from .joblib...', 'green'))
		#saved_model = load(Path(__file__).parent / "blackboxes" / "toy.joblib")
		#for i in range (0,10):
			#print('Predicted Class:', make_prediction(saved_model, test[i])[0])
	
	#train random forest and create .joblib file from heart failure dataset
	elif arg_blackbox == 'heart':
			
		#get database for the blackbox
		print(colored('Fetching and preparing database...', 'green'))
		X, y, names = getdata('heart')
				
		#train on best model parameters
		print(colored('Training random forest with optimal model parameters...', 'green'))
		_, _, _, _, _, _, model, _, test, _ = trainmodel(X, y, 'rfc', 'heart', 'None')
				
		#save model as .joblib file
		print(colored('Saving random forest model as .joblib file...', 'green'))
		dump(model, Path(__file__).parent / "blackboxes" / "heart.joblib")
		
	#train random forest and create .joblib file from diabetes dataset
	elif arg_blackbox == 'diabetes':
		
		#get database for the blackbox
		print(colored('Fetching and preparing database...', 'green'))
		X, y, names = getdata('diabetes')
				
		#train on best model parameters
		print(colored('Training random forest with optimal model parameters...', 'green'))
		_, _, _, _, _, _, model, _, test, _ = trainmodel(X, y, 'rfc', 'diabetes', 'None')
				
		#save model as .joblib file
		print(colored('Saving random forest model as .joblib file...', 'green'))
		dump(model, Path(__file__).parent / "blackboxes" / "diabetes.joblib")
		
		#test code
		#make a prediction with the model after loading from .joblib
		#print(colored('Make prediction with the model after loading from .joblib...', 'green'))
		#saved_model = load(Path(__file__).parent / "blackboxes" / "diabetes.joblib")
		#diabetes = np.array([70,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1])
		#nodiabetes = np.array([40,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
		#print('Predicted Class (diabetes):', make_prediction(saved_model, diabetes)[0])
		#print('Predicted Class (nodiabetes):', make_prediction(saved_model, nodiabetes)[0])

#sort generate sample files and create balanced dataset for training and testing surrogate models
elif arg_generate != '':
	
	print(colored('Generate balanced dataset for surrogate model...', 'green'))
	#generate the samples for balanced dataset
	generate_samples(arg_generate)

#train global surrogate model (decision tree) from sample inputs
elif arg_surrogate != '' and arg_name != '':
	
	#train global surrogate for toy data blackbox
	if arg_surrogate == 'toy':
		
		#load blackbox
		blackbox = load("blackboxes/toy.joblib")
		
		#check amount of samples
		samplesize = len(os.listdir('samples/'+arg_name+'/train/train1'))
		samplesizetest = len(os.listdir('samples/'+arg_name+'/test/test1'))
		
		#prepare arrays
		samples_x = np.array([np.zeros((samplesize, 10), dtype=int), np.zeros((samplesize, 10), dtype=int), np.zeros((samplesize, 10), dtype=int), np.zeros((samplesize, 10), dtype=int), np.zeros((samplesize, 10), dtype=int), np.zeros((samplesize, 10), dtype=int), np.zeros((samplesize, 10), dtype=int), np.zeros((samplesize, 10), dtype=int), np.zeros((samplesize, 10), dtype=int), np.zeros((samplesize, 10), dtype=int)])
		samples_y = np.array([np.zeros((samplesize, 1), dtype=int), np.zeros((samplesize, 1), dtype=int), np.zeros((samplesize, 1), dtype=int), np.zeros((samplesize, 1), dtype=int), np.zeros((samplesize, 1), dtype=int), np.zeros((samplesize, 1), dtype=int), np.zeros((samplesize, 1), dtype=int), np.zeros((samplesize, 1), dtype=int), np.zeros((samplesize, 1), dtype=int), np.zeros((samplesize, 1), dtype=int)])
		samplestest_x = np.array([np.zeros((samplesizetest, 10), dtype=int), np.zeros((samplesizetest, 10), dtype=int), np.zeros((samplesizetest, 10), dtype=int), np.zeros((samplesizetest, 10), dtype=int), np.zeros((samplesizetest, 10), dtype=int), np.zeros((samplesizetest, 10), dtype=int), np.zeros((samplesizetest, 10), dtype=int), np.zeros((samplesizetest, 10), dtype=int), np.zeros((samplesizetest, 10), dtype=int), np.zeros((samplesizetest, 10), dtype=int)])
		samplestest_y = np.array([np.zeros((samplesizetest, 1), dtype=int), np.zeros((samplesizetest, 1), dtype=int), np.zeros((samplesizetest, 1), dtype=int), np.zeros((samplesizetest, 1), dtype=int), np.zeros((samplesizetest, 1), dtype=int), np.zeros((samplesizetest, 1), dtype=int), np.zeros((samplesizetest, 1), dtype=int), np.zeros((samplesizetest, 1), dtype=int), np.zeros((samplesizetest, 1), dtype=int), np.zeros((samplesizetest, 1), dtype=int)])
		
		print(colored('Make predictions for training and test samples...', 'green'))
		for z in tqdm(range(10), leave=False):
			
			#make predictions with the blackbox for the training samples
			for i in tqdm(range(0,len(samples_x[z])), leave=False):
				
				#load sample from file
				f = open("samples/"+arg_name+"/train/train"+str(z+1)+"/"+arg_name+"train."+str(i+1)+".expr", "r")
				
				#prepare and split string into array
				currentstring=f.read()[2:-3]
				split = currentstring.split(",")
				split = [int(x) for x in split]
				samples_x[z][i] = split
				
				#make prediction for the sample
				samples_y[z][i][0] = int(make_prediction(blackbox, samples_x[z][i])[0])

			#make predictions with the blackbox for the test samples			
			for i in tqdm(range(0,len(samplestest_x[z])), leave=False):
				
				#load sample from file
				f = open("samples/"+arg_name+"/test/test"+str(z+1)+"/"+arg_name+"test."+str(i+1)+".expr", "r")
				
				#prepare and split string into array
				currentstring=f.read()[2:-3]
				split = currentstring.split(",")
				split = [int(x) for x in split]
				samplestest_x[z][i] = split
				
				#make prediction for the sample
				samplestest_y[z][i][0] = int(make_prediction(blackbox, samplestest_x[z][i])[0])
		
			#save predictions to text files
			np.savetxt('samples/'+arg_name+'/class/trainY'+str(z+1)+'.txt', samples_y[z], fmt='%d')
			np.savetxt('samples/'+arg_name+'/class/testY'+str(z+1)+'.txt', samplestest_y[z], fmt='%d')
		
		#train surrogate models with samples + predictions of blackbox		
		print(colored('Creat global surrogates for toy dataset blackbox...', 'green'))
		surrogates = trainmodel(samples_x, samples_y, 'treeSurrogatToy', 'selftoy', 'tree')	
		
		#save surrogate models to .joblib files in /surrogate folder		
		print(colored('Save global surrogates as .joblib files in /surrogate folder...', 'green'))
		for i in range(len(surrogates)):
			dump(surrogates[i], "surrogate/toydata/"+arg_name+"surrogate12345678samples"+str(i+1)+".joblib")
	
	#train global surrogate for heartfailure blackbox					
	elif arg_surrogate == 'heart':
		
		#load blackbox
		blackbox = load("blackboxes/heart.joblib")
		
		#check amount of samples	
		samplesizeh = len(os.listdir('samples/'+arg_name+'/train/train1'))
		samplesizehtest = len(os.listdir('samples/'+arg_name+'/test/test1'))
		
		#prepare arrays
		samplesh_x = np.array([np.zeros((samplesizeh, 11)), np.zeros((samplesizeh, 11)), np.zeros((samplesizeh, 11)), np.zeros((samplesizeh, 11)), np.zeros((samplesizeh, 11)), np.zeros((samplesizeh, 11)), np.zeros((samplesizeh, 11)), np.zeros((samplesizeh, 11)), np.zeros((samplesizeh, 11)), np.zeros((samplesizeh, 11))])
		samplesh_y = np.array([np.zeros((samplesizeh, 1), dtype=int), np.zeros((samplesizeh, 1), dtype=int), np.zeros((samplesizeh, 1), dtype=int), np.zeros((samplesizeh, 1), dtype=int), np.zeros((samplesizeh, 1), dtype=int), np.zeros((samplesizeh, 1), dtype=int), np.zeros((samplesizeh, 1), dtype=int), np.zeros((samplesizeh, 1), dtype=int), np.zeros((samplesizeh, 1), dtype=int), np.zeros((samplesizeh, 1), dtype=int)]) 
		sampleshtest_x = np.array([np.zeros((samplesizehtest, 11), dtype=int), np.zeros((samplesizehtest, 11), dtype=int), np.zeros((samplesizehtest, 11), dtype=int), np.zeros((samplesizehtest, 11), dtype=int), np.zeros((samplesizehtest, 11), dtype=int), np.zeros((samplesizehtest, 11), dtype=int), np.zeros((samplesizehtest, 11), dtype=int), np.zeros((samplesizehtest, 11), dtype=int), np.zeros((samplesizehtest, 11), dtype=int), np.zeros((samplesizehtest, 11), dtype=int)])
		sampleshtest_y = np.array([np.zeros((samplesizehtest, 1), dtype=int), np.zeros((samplesizehtest, 1), dtype=int), np.zeros((samplesizehtest, 1), dtype=int), np.zeros((samplesizehtest, 1), dtype=int), np.zeros((samplesizehtest, 1), dtype=int), np.zeros((samplesizehtest, 1), dtype=int), np.zeros((samplesizehtest, 1), dtype=int), np.zeros((samplesizehtest, 1), dtype=int), np.zeros((samplesizehtest, 1), dtype=int), np.zeros((samplesizehtest, 1), dtype=int)])
	

		print(colored('Make predictions for training and test samples...', 'green'))
		for z in tqdm(range(10), leave=False):	
		
			#make predictions with the blackbox for the training samples
			for i in tqdm(range(0,len(samplesh_x[z])), leave=False):
				
				#load sample from file
				f = open("samples/"+arg_name+"/train/train"+str(z+1)+"/"+arg_name+"train."+str(i+1)+".expr", "r")
				
				#prepare and split string into array
				currentstring=f.read()[2:-3]
				split = currentstring.split(",")
				split = [float(x) for x in split]
				samplesh_x[z][i] = split
				
				#make prediction for the sample
				samplesh_y[z][i][0] = int(make_prediction(blackbox, samplesh_x[z][i])[0])

			#make predictions with the blackbox for the test samples
			for i in tqdm(range(0,len(sampleshtest_x[z])), leave=False):
				
				#load sample from file
				f = open("samples/"+arg_name+"/test/test"+str(z+1)+"/"+arg_name+"test."+str(i+1)+".expr", "r")
				
				#prepare and split string into array
				currentstring=f.read()[2:-3]
				split = currentstring.split(",")
				split = [float(x) for x in split]
				sampleshtest_x[z][i] = split
				
				#make prediction for the sample
				sampleshtest_y[z][i][0] = int(make_prediction(blackbox, sampleshtest_x[z][i])[0])
		
			#save predictions to text files
			np.savetxt('samples/'+arg_name+'/class/trainY'+str(z+1)+'.txt', samplesh_y[z], fmt='%d')
			np.savetxt('samples/'+arg_name+'/class/testY'+str(z+1)+'.txt', sampleshtest_y[z], fmt='%d')

		#train surrogate models with samples + predictions of blackbox
		print(colored('Create global surrogates for heartfailure dataset blackbox...', 'green'))		
		surrogates = trainmodel(samplesh_x, samplesh_y, 'treeSurrogatHeart', 'selfheart', 'tree')	
		
		#save surrogate models to .joblib files in /surrogate folder	
		print(colored('Save global surrogates as .joblib files in /surrogate folder...', 'green'))

		for i in range(len(surrogates)):
			if i < 10:
				dump(surrogates[i], "surrogate/heartfailureDepth3/"+arg_name+"surrogate12345678depth3samples"+str(i+1)+".joblib")
			elif i < 20:
				dump(surrogates[i], "surrogate/heartfailureDepth4/"+arg_name+"surrogate12345678depth4samples"+str(i+1-10)+".joblib")

	#train global surrogate for diabetes blackbox					
	elif arg_surrogate == 'diabetes':
		
		#load blackbox
		blackbox = load("blackboxes/diabetes.joblib")
		
		#check amount of samples	
		samplesizeh = len(os.listdir('samples/'+arg_name+'/train/train1'))
		samplesizehtest = len(os.listdir('samples/'+arg_name+'/test/test1'))
		
		#prepare arrays
		samplesh_x = np.array([np.zeros((samplesizeh, 16), dtype=int), np.zeros((samplesizeh, 16), dtype=int), np.zeros((samplesizeh, 16), dtype=int), np.zeros((samplesizeh, 16), dtype=int), np.zeros((samplesizeh, 16), dtype=int), np.zeros((samplesizeh, 16), dtype=int), np.zeros((samplesizeh, 16), dtype=int), np.zeros((samplesizeh, 16), dtype=int), np.zeros((samplesizeh, 16), dtype=int), np.zeros((samplesizeh, 16), dtype=int)])
		samplesh_y = np.array([np.zeros((samplesizeh, 1), dtype=int), np.zeros((samplesizeh, 1), dtype=int), np.zeros((samplesizeh, 1), dtype=int), np.zeros((samplesizeh, 1), dtype=int), np.zeros((samplesizeh, 1), dtype=int), np.zeros((samplesizeh, 1), dtype=int), np.zeros((samplesizeh, 1), dtype=int), np.zeros((samplesizeh, 1), dtype=int), np.zeros((samplesizeh, 1), dtype=int), np.zeros((samplesizeh, 1), dtype=int)]) 
		sampleshtest_x = np.array([np.zeros((samplesizehtest, 16), dtype=int), np.zeros((samplesizehtest, 16), dtype=int), np.zeros((samplesizehtest, 16), dtype=int), np.zeros((samplesizehtest, 16), dtype=int), np.zeros((samplesizehtest, 16), dtype=int), np.zeros((samplesizehtest, 16), dtype=int), np.zeros((samplesizehtest, 16), dtype=int), np.zeros((samplesizehtest, 16), dtype=int), np.zeros((samplesizehtest, 16), dtype=int), np.zeros((samplesizehtest, 16), dtype=int)])
		sampleshtest_y = np.array([np.zeros((samplesizehtest, 1), dtype=int), np.zeros((samplesizehtest, 1), dtype=int), np.zeros((samplesizehtest, 1), dtype=int), np.zeros((samplesizehtest, 1), dtype=int), np.zeros((samplesizehtest, 1), dtype=int), np.zeros((samplesizehtest, 1), dtype=int), np.zeros((samplesizehtest, 1), dtype=int), np.zeros((samplesizehtest, 1), dtype=int), np.zeros((samplesizehtest, 1), dtype=int), np.zeros((samplesizehtest, 1), dtype=int)])
	
		print(colored('Make predictions for training and test samples...', 'green'))
		for z in tqdm(range(10), leave=False):	
		
			#make predictions with the blackbox for the training samples
			for i in tqdm(range(0,len(samplesh_x[z])), leave=False):
				
				#load sample from file
				f = open("samples/"+arg_name+"/train/train"+str(z+1)+"/"+arg_name+"train."+str(i+1)+".expr", "r")
				
				#prepare and split string into array
				currentstring=f.read()[2:-3]
				split = currentstring.split(",")
				split = [int(x) for x in split]
				samplesh_x[z][i] = split
				
				#make prediction for the sample
				samplesh_y[z][i][0] = int(make_prediction(blackbox, samplesh_x[z][i])[0])

			#make predictions with the blackbox for the test samples
			for i in tqdm(range(0,len(sampleshtest_x[z])), leave=False):
				
				#load sample from file
				f = open("samples/"+arg_name+"/test/test"+str(z+1)+"/"+arg_name+"test."+str(i+1)+".expr", "r")
				
				#prepare and split string into array
				currentstring=f.read()[2:-3]
				split = currentstring.split(",")
				split = [int(x) for x in split]
				sampleshtest_x[z][i] = split
				
				#make prediction for the sample
				sampleshtest_y[z][i][0] = int(make_prediction(blackbox, sampleshtest_x[z][i])[0])
		
			#save predictions to text files
			np.savetxt('samples/'+arg_name+'/class/trainY'+str(z+1)+'.txt', samplesh_y[z], fmt='%d')
			np.savetxt('samples/'+arg_name+'/class/testY'+str(z+1)+'.txt', sampleshtest_y[z], fmt='%d')

		#train surrogate models with samples + predictions of blackbox
		print(colored('Create global surrogates for diabetes dataset blackbox...', 'green'))		
		surrogates = trainmodel(samplesh_x, samplesh_y, 'treeSurrogatDiabetes', 'selfdiabetes', 'tree')	
		
		#save surrogate models to .joblib files in /surrogate folder	
		print(colored('Save global surrogates as .joblib files in /surrogate folder...', 'green'))

		for i in range(len(surrogates)):
			if i < 10:
				dump(surrogates[i], "surrogate/diabetesDepth3/"+arg_name+"surrogate12345678depth3samples"+str(i+1)+".joblib")
			elif i < 20:
				dump(surrogates[i], "surrogate/diabetesDepth4/"+arg_name+"surrogate12345678depth4samples"+str(i+1-10)+".joblib")
								
elif arg_evaluate != '' and arg_open != '':
	
	#if created global surrogate models
	if arg_evaluate in ['toydata', 'heartfailureDepth3', 'heartfailureDepth4', 'diabetesDepth3', 'diabetesDepth4']:
		path = "surrogate/"+arg_evaluate
		surrogates = []
		files = []

		#load surrogate models	
		if os.path.isfile(path):
			
			surrogate, _ = loadpickle(path, 'none')
			surrogates.append(surrogate)
			
		elif os.path.isdir(path):
			
			#construct paths of files 
			for i in range(10):
				if arg_evaluate == 'toydata':
					files.append('toydatasurrogate12345678samples'+str(i+1)+'.joblib')
				elif arg_evaluate == 'heartfailureDepth3':
					files.append('heartfailuresurrogate12345678depth3samples'+str(i+1)+'.joblib')
				elif arg_evaluate == 'heartfailureDepth4':
					files.append('heartfailuresurrogate12345678depth4samples'+str(i+1)+'.joblib')
				elif arg_evaluate == 'diabetesDepth3':
					files.append('diabetessurrogate12345678depth3samples'+str(i+1)+'.joblib')
				elif arg_evaluate == 'diabetesDepth4':
					files.append('diabetessurrogate12345678depth4samples'+str(i+1)+'.joblib')

			for x in files:
				if (x[-6:] in ['joblib', 'pickle']):		
					surrogate, _ = loadpickle(path+"/"+x, 'none')
					surrogates.append(surrogate)
			
		#load test set black box predictions
		blackbox_y = np.array([np.loadtxt('samples/'+arg_open+'/class/testY1.txt', dtype=int), np.loadtxt('samples/'+arg_open+'/class/testY2.txt', dtype=int), np.loadtxt('samples/'+arg_open+'/class/testY3.txt', dtype=int), np.loadtxt('samples/'+arg_open+'/class/testY4.txt', dtype=int), np.loadtxt('samples/'+arg_open+'/class/testY5.txt', dtype=int), np.loadtxt('samples/'+arg_open+'/class/testY6.txt', dtype=int), np.loadtxt('samples/'+arg_open+'/class/testY7.txt', dtype=int), np.loadtxt('samples/'+arg_open+'/class/testY8.txt', dtype=int), np.loadtxt('samples/'+arg_open+'/class/testY9.txt', dtype=int), np.loadtxt('samples/'+arg_open+'/class/testY10.txt', dtype=int)])
		
		#check number of test samples and load first test sample to check needed array size
		samplesize = len(os.listdir('samples/'+arg_open+'/test/test1'))
		f = open("samples/"+arg_open+"/test/test1/"+arg_open+"test."+str(1)+".expr", "r")
		currentstring=f.read()[2:-3]
		split = currentstring.split(",")
		split = [float(x) for x in split]
		featurecount = len(split)

		#create arrays
		samples_x = np.array([np.zeros((samplesize, featurecount)), np.zeros((samplesize, featurecount)), np.zeros((samplesize, featurecount)), np.zeros((samplesize, featurecount)), np.zeros((samplesize, featurecount)), np.zeros((samplesize, featurecount)), np.zeros((samplesize, featurecount)), np.zeros((samplesize, featurecount)), np.zeros((samplesize, featurecount)), np.zeros((samplesize, featurecount))])
		samples_y = [[] for _ in range(len(surrogates))] 
		
		#make predictions with the surrogate models for the test samples
		print(colored('Make predictions for test samples...', 'green'))
		for z in tqdm(range(10), leave=False):
		
			for i in tqdm(range(0,len(samples_x[z])), leave=False):
				
				#load sample from file
				f = open("samples/"+arg_open+"/test/test"+str(z+1)+"/"+arg_open+"test."+str(i+1)+".expr", "r")
				
				#prepare and split string into array
				currentstring=f.read()[2:-3]
				split = currentstring.split(",")
				split = [float(x) for x in split]
				samples_x[z][i] = split
				
				#make prediction for the sample
				samples_y[z].append(int(make_prediction(surrogates[z], samples_x[z][i])[0])) 
		
	#if read from csv files			
	elif arg_open == 'csv':
		path = "surrogate/"+arg_evaluate+"/performance"
		
		#find paths of relevant files 
		files = os.listdir(path)
		files.sort()
		
		samples_y = [[] for _ in range(len(files))] 
		blackbox_y = [[] for _ in range(len(files))] 
		
		#read surrogate results and swap 'BUG', 'NOBUG'
		for x in range(len(files)):
			results = pd.read_csv(path+'/'+files[x])
			swaper = results.prediction.to_numpy()

			for i in range(len(swaper)):
				if swaper[i] == 'BUG':
					samples_y[x].append(1)		
				else:
					samples_y[x].append(0)		

		#read black box results and swap 'BUG', 'NOBUG'
		for x in range(len(files)):
			results = pd.read_csv(path+'/'+files[x])
			swaper = results.oracle.to_numpy()

			for i in range(len(swaper)):
				if swaper[i] == 'BUG':
					blackbox_y[x].append(1)		
				else:
					blackbox_y[x].append(0)	
		

	#calculate accuracy, precision, recall, specificity, f1 and mcc	
	accuracy = []
	precision = []
	recall = []
	specificity = []
	f1 = []
	mcc = []

	for i in range(len(samples_y)):
		accuracy.append(metrics.accuracy_score(blackbox_y[i], samples_y[i]))
		precision.append(metrics.precision_score(blackbox_y[i], samples_y[i]))
		recall.append(metrics.recall_score(blackbox_y[i], samples_y[i]))
		specificity.append(metrics.recall_score(blackbox_y[i], samples_y[i], pos_label=0))
		f1.append(metrics.f1_score(blackbox_y[i], samples_y[i]))
		mcc.append(metrics.matthews_corrcoef(blackbox_y[i], samples_y[i]))
		
		#if arg_evaluate not in ['toydata', 'heartfailureDepth3', 'diabetesDepth3', 'heartfailureDepth4', 'diabetesDepth4']: 
		#print performance metrics for every model
		print(colored('Performance metrics of ', 'cyan')+colored(files[i], 'red')+colored(':', 'cyan'))
		print('Accuracy:   ', format(accuracy[i], '.4f'),
			  '\nPrecision:  ', format(precision[i], '.4f'),
			  '\nRecall:     ', format(recall[i], '.4f'),
			  '\nSpecificity:', format(specificity[i], '.4f'),
			  '\nF1:         ', format(f1[i], '.4f'),
			  '\nMCC:        ', format(mcc[i], '.4f'))	

	#print performance metrics
	print(colored('Performance metrics of ', 'cyan')+colored(arg_evaluate, 'red')+colored(':', 'cyan'))
	print('Accuracy mean:   ', format(np.mean(accuracy), '.4f'), 'Accuracy std:   ', format(np.std(accuracy), '.4f'),
		  '\nPrecision mean:  ', format(np.mean(precision), '.4f'), 'Precision std:  ', format(np.std(precision), '.4f'),
		  '\nRecall mean:     ', format(np.mean(recall), '.4f'), 'Recall std:     ', format(np.std(recall), '.4f'),
		  '\nSpecificity mean:', format(np.mean(specificity), '.4f'), 'Specificity std:', format(np.std(specificity), '.4f'),
		  '\nF1 mean:         ', format(np.mean(f1), '.4f'), 'F1 std:         ', format(np.std(f1), '.4f'),
		  '\nMCC mean:        ', format(np.mean(mcc), '.4f'), 'MCC std:        ', format(np.std(mcc), '.4f'))	
	
elif arg_compare != '':
		
	folderpath = "surrogate/"+arg_compare
	surrogates = []
	
	#set feature names
	if arg_compare == 'toydata':
		names=['bool0', 'bool1', 'bool2', 'bool3', 'bool4', 'bool5', 'bool6', 'bool7', 'bool8', 'bool9']
	elif 'heartfailure' in arg_compare:
		names=['age', 'anaemia', 'creatinine_phosphokinase', 'diabetes', 'ejection_fraction', 'high_blood_pressure', 'platelets', 'serum_creatinine', 'serum_sodium', 'sex', 'smoking']
	elif 'alhazenheart' in arg_compare:
		namesl= []
		df = pd.read_csv(folderpath+'/depth.csv')
		table = df.to_numpy()
		sorttable = table[table[:, 0].argsort()]
		
		#format names
		for i in range(len(sorttable)):
			if sorttable[i][0][:7] == 'max-num':
				namesl.append(sorttable[i][0].partition("@")[0][8:])
			else:
				namesl.append(sorttable[i][3].replace('"', "'"))
		names = np.asarray(namesl)
		
	elif 'alhazentoy' in arg_compare:
		names=['0','bool0==0','bool0==1','bool0==exist','bool1==0','bool1==1','bool1==exist','bool2==0','bool2==1','bool2==exist','bool3==0','bool3==1','bool3==exist','bool4==0','bool4==1','bool4==exist','bool5==0','bool5==1','bool5==exist','bool6==0','bool6==1','bool6==exist','bool7==0','bool7==1','bool7==exist','bool8==0','bool8==1','bool8==exist','bool9==0','bool9==1','bool9==exist']

	elif 'alhazendiabetes' in arg_compare:
		namesl= []
		df = pd.read_csv(folderpath+'/depth.csv')
		table = df.to_numpy()
		sorttable = table[table[:, 0].argsort()]
		
		#format names
		for i in range(len(sorttable)):
			if sorttable[i][0][:7] == 'max-num':
				namesl.append(sorttable[i][0].partition("@")[0][8:])
			else:
				namesl.append(sorttable[i][3].replace('"', "'"))
		names = np.asarray(namesl)
	elif 'diabetes' in arg_compare:
		names=['Age', 'Gender', 'Polyuria', 'Polydipsia', 'sudden_weight_loss', 'weakness', 'Polyphagia', 'Genital_thrush', 'visual_blurring', 'Itching', 'Irritability', 'delayed_healing', 'partial_paresis', 'muscle_stiffness', 'Alopecia', 'Obesity']		
	
	#check if at least two models in given folder to compare
	if len(os.listdir(folderpath)) < 2:
		print("Need at least two models models to compare in surrogate/"+arg_compare+"!")
		sys.exit(2)
	
	#find paths of relevant files and load
	files = os.listdir(folderpath)
	paths = []

	for x in files:
		if (x[-6:] in ['joblib', 'pickle']):		
			surrogate, _ = loadpickle(folderpath+"/"+x, 'none')
			surrogates.append(surrogate)
	
	#evaluate stability of explanation
	vsi, fsi, feature_stability = tree_stability(surrogates, names)

	#print decision tree stability
	print(colored('Feature stability of surrogate models in', 'cyan'), colored(folderpath, 'red')+colored('...', 'cyan'))
	print(f"{'Feature Names':<40}{'Stability':<30}{'in Iteration'}")
	for i in feature_stability:

		print(f"{'%s' % (i[4]):<40}{round(i[2]*100, 2):<30}{i[3]}")
		
	#print stability indices
	print(colored('Stability indices for surrogate models in', 'cyan'), colored(folderpath, 'red')+colored('...', 'cyan'))
	print("FSI: ", fsi, "\nVSI: ", vsi)
	
	"""
	#print decision tree stability
	print(colored('Feature stability of surrogate models in', 'cyan'), colored(folderpath, 'red')+colored('...', 'cyan'))
	print(f"{'Feature Names':<40}{'Tree Path':<30}{'Stability':<30}{'in Iteration'}")
	for i in feature_stability:

		print(f"{'%s(%s)' % (names[i[0]], i[0]):<40}{'%s' % i[1]:<30}{round(i[4]*100, 2):<30}{i[5]}")
		
	#print stability indices
	print(colored('Stability indices for surrogate models in', 'cyan'), colored(folderpath, 'red')+colored('...', 'cyan'))
	print("FSI: ", fsi, "\nVSI: ", vsi)
	"""
