import numpy as np
import sys

#set seed for randomness
def set_seed(seed):
	np.random.RandomState(seed)
	np.random.seed(seed)
	
seed = 665
set_seed(seed)

#calculate vsi(variable stability index) over a number of lime explanations
def calc_vsi_lime(exps):
	
	#calculate vsi
	vsi_parts=[]
	#for every combination of explanation feature sets
	for i in range (len(exps)):
		for j in range (len(exps)):
			if j > i:
				a=[]
				b=[]
				#extract feature names
				for k in range(len(exps[i].as_list())):
					a.append(exps[i].as_list()[k][0]) 
					b.append(exps[j].as_list()[k][0])
				#calculate amount of same features
				intersect = [x for x in a if x in b]
				percent_same = len(intersect)/len(exps[i].as_list())
				vsi_parts.append(percent_same)
	#mean over all combinations and set to between 0 and 100
	vsi = round(np.mean(vsi_parts)*100, 2)
	
	return vsi
	
#calculate fsi(feature stability index) over a number of lime explanations
def calc_fsi_lime(exps):
	
	#collect importance scores for the same feature into lists
	collect_same = []
	feature_mean = []
	for i in range (len(exps)):
		for j in exps[i].as_list():
			found=False
			for x in collect_same:
				if x[0] == j[0]:
					x.append(j[1])
					found=True
			if found==False:
				collect_same.append([j[0], j[1]])


	#calculate mean for importance of same feature
	for i in collect_same:
		feature_mean.append([i[0], np.mean(i[1:])])
	
	#calculate feature stability over multiple explanations
	feature_stability = []
	for i in collect_same:
		in_interval=0
		number_checks=0
		
		#set allowed fluctuation as percentage of the mean of the feature importance
		for x in feature_mean:
			if i[0] in x:
				#set percentage here!
				interval = abs((x[1]/100) * 10)
				break
		
		#check every combination if fluctuation in allowed range
		for j in range (1,len(i)):
			for k in range (1,len(i)):
				if len(i) > 2 and k > j:
					
					if (i[j] - interval) <= i[k] <= (i[j] + interval):
						in_interval += 1
						number_checks += 1
					else:
						number_checks += 1
			
			#add the results to new list
			if j == len(i)-1:
				if len(i) == 2:
					feature_stability.append([i[0], 0, 0, 1.0])
				else:
					feature_stability.append([i[0], in_interval, number_checks, in_interval/number_checks])
	
	#add to feature_stability in how many iterations a feature was used to explain
	for x in feature_stability:
		for y in collect_same:
			if x[0] in y:
				x.append(len(y)-1)
	
	#calculate feature stability index (FSI) over all features
	feature_stability_index = 0
	count = 0
	for x in feature_stability:
		if x[4] > 1:
			feature_stability_index += x[3]
			count += 1
	feature_stability_index = round((feature_stability_index / count) * 100, 2)
	
	return feature_stability_index, feature_stability

def tree_stability(surrogates, names):
	importance = []
	nonzero = []
		
	#extract feature importance and find index of nonzero features
	for j in range(len(surrogates)):
		importance.append(surrogates[j].feature_importances_)
		nonzero.append([i for i, e in enumerate(importance[j]) if e != 0])

	#calculate vsi
	vsi_parts=[]
	#for every combination of surrogate model feature sets
	for i in range (len(nonzero)):
		for j in range (len(nonzero)):
			if j > i:
				
				#calculate amount of same features
				intersect = [x for x in nonzero[i] if x in nonzero[j]]

				if len(nonzero[i]) >= len(nonzero[j]):
					percent_same = len(intersect)/len(nonzero[i])
				else:
					percent_same = len(intersect)/len(nonzero[j])
		
				vsi_parts.append(percent_same)

	#mean over all combinations and set to between 0 and 100
	vsi = round(np.mean(vsi_parts)*100, 2)
	
	#calculate fsi	
	#collect feature importance scores for the same feature into lists
	collect_same = [[] for _ in range(len(importance[0]))]
	feature_mean = []

	for i in range(len(nonzero)):
		for j in nonzero[i]:
			collect_same[j].append(importance[i][j])
	
	#calculate mean for importance scores of same feature
	for i in collect_same:
		if len(i) > 0:
			feature_mean.append(np.mean(i))
		else: 
			feature_mean.append(0)
	
	
	#calculate importance score stability over multiple explanations
	feature_stability = []
	for i in range(len(collect_same)):
		in_interval=0
		number_checks=0
		
		#set allowed fluctuation as percentage of the mean of the feature importance scores
		#set percentage here!
		interval = abs((feature_mean[i]/100) * 10)

		#check every combination if fluctuation in allowed range
		for j in range(len(collect_same[i])):
			for k in range(len(collect_same[i])):
				if k > j:
					if (collect_same[i][j] - interval) <= collect_same[i][k] <= (collect_same[i][j] + interval):
						in_interval += 1
						number_checks += 1
					else:
						number_checks += 1
			
			#add the results to new list
			if j == len(collect_same[i])-1:
				if len(collect_same[i]) == 1:
					feature_stability.append([0, 0, 1.0, 1, names[i]])
				else:
					feature_stability.append([in_interval, number_checks, in_interval/number_checks, len(collect_same[i]), names[i]])

	#calculate feature importance stability index (FSI) over all features
	feature_stability_index = 0
	count = 0
	for x in feature_stability:
		if x[3] > 1:
			feature_stability_index += x[2]
			count += 1
	fsi = round((feature_stability_index / count) * 100, 2)

	"""
	split_features = [[] for _ in range(len(surrogates))]
	
	#check every tree node for feature and threshold	
	for i in range(len(surrogates)):
		
		children_left = surrogates[i].tree_.children_left
		children_right = surrogates[i].tree_.children_right
		feature = surrogates[i].tree_.feature
		threshold = surrogates[i].tree_.threshold
		stack = [(0, 0, ".")]
		
		while len(stack) > 0:
			
			#get next node from stack
			node_id, depth, path = stack.pop()
			
			#if not leave node
			if children_left[node_id] != children_right[node_id]:
				
				#add children to stack
				stack.append((children_left[node_id], depth + 1, path+"->l "+str(feature[node_id])))
				stack.append((children_right[node_id], depth + 1, path+"->r "+str(feature[node_id])))
				
				#get feature name and threshold from current node and save to split_features
				split_features[i].append((feature[node_id], threshold[node_id], path))
					
	#calculate vsi
	vsi_parts=[]
	#for every combination of surrogate model feature sets
	for i in range (len(split_features)):
		for j in range (len(split_features)):
			if j > i:
				
				a=[]
				b=[]
				
				#extract feature names
				for x in split_features[i]:
					a.append(x[0])
				for x in split_features[j]: 
					b.append(x[0])
					
				#calculate amount of same features
				intersect = [x for x in a if x in b]
				if len(a) >= len(b):
					percent_same = len(intersect)/len(a)
				else:
					percent_same = len(intersect)/len(b)			
				vsi_parts.append(percent_same)

	#mean over all combinations and set to between 0 and 100
	vsi = round(np.mean(vsi_parts)*100, 2)			


	#calculate fsi	
	#collect threshold scores for the same feature into lists
	collect_same = []
	feature_mean = []
	for i in range(len(split_features)):
		for j in split_features[i]:
			found=False
			for x in collect_same:
				if x[0] == j[0] and x[1] == j[2]:
					x.append(j[1])
					found=True
			if found==False:
				collect_same.append([j[0], j[2], j[1]])
	
	#calculate mean for thresholds of same feature
	for i in collect_same:
		feature_mean.append([i[0], i[1], np.mean(i[2:])])
		
	#print(collect_same)
	#print(feature_mean)
	
	#calculate threshold stability over multiple explanations
	feature_stability = []
	for i in collect_same:
		in_interval=0
		number_checks=0
		
		#set allowed fluctuation as percentage of the mean of the feature threshold
		for x in feature_mean:
			if (i[0]) in x:
				if (i[1]) in x:
					#set percentage here!
					interval = abs((x[2]/100) * 10)
					break
					
		#check every combination if fluctuation in allowed range
		for j in range (2,len(i)):
			for k in range (2,len(i)):
				if len(i) > 3 and k > j:
					if (i[j] - interval) <= i[k] <= (i[j] + interval):
						in_interval += 1
						number_checks += 1
					else:
						number_checks += 1
			
			#add the results to new list
			if j == len(i)-1:
				if len(i) == 3:
					feature_stability.append([i[0], i[1], 0, 0, 1.0])
				else:
					feature_stability.append([i[0], i[1], in_interval, number_checks, in_interval/number_checks])
		
	#add to feature_stability in how many iterations a feature was used to explain
	for x in feature_stability:
		for y in collect_same:
			if x[0] in y:
				if x[1] in y:
					x.append(len(y)-2)
		
	#calculate feature stability index (FSI) over all features
	feature_stability_index = 0
	count = 0
	for x in feature_stability:
		if x[5] > 1:
			feature_stability_index += x[4]
			count += 1
	fsi = round((feature_stability_index / count) * 100, 2)
	"""
	
	return vsi, fsi, feature_stability
