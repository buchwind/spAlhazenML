from matplotlib import pyplot
from sklearn import tree
import os

#plot decision tree
def plot_tree(model, path ,names):
	
	#change class names to BUG/NO_BUG
	classes = list(map(str, model.classes_))

	for i in range(len(classes)):
		if classes[i] in ['0', '0.0']:
			classes[i] = 'NO_BUG'
		elif classes[i] in ['1', '1.0']:
			classes[i] = 'BUG'
			
	#dot graphiv plot
	tree.export_graphviz(model, out_file=path+".dot", feature_names=names, class_names=classes, filled=False)
	os.system('dot -Tpng -Gdpi=300 '+path+'.dot -o '+path+'.png')
	os.system('rm '+path+'.dot')

#plot feature importance
def plot_importance(model, names):
	
	#extract feature importance
	importance = model.feature_importances_
	
	#print feature importance
	for i,v in enumerate(importance):
		print('Feature: %s, Score: %.5f' % (names[i], v))
	
	#plot feature importance
	pyplot.bar([names[x] for x in range(len(importance))], importance)
	pyplot.tick_params(labelsize=5.0)
	pyplot.show()
	
#plot stability with different kernel widths
def plotstability(best_width, best_R_squared, best_csi, best_vsi, results, maxRsquared, data, adherence):
	
	#define colors for plot
	colors = ['teal', 'yellowgreen', 'gold']
	
	#define scatterplot
	pyplot.scatter(results.kernel_width, results.csi, label=r'$CSI$', color = colors[1])
	pyplot.scatter(results.kernel_width, results.R_squared, label=r'$l(\tilde{R}^2,kw)$', color = colors[0])
	pyplot.scatter(best_width, best_R_squared, c = "red", s=30, label="kw: %.3f \nR_squared: %.3f" % (best_width, best_R_squared))
	pyplot.plot([], [], ' ', label=r'target $\tilde{R}^2 = %.2f $' % maxRsquared)
	pyplot.xlabel("kernel width")
	pyplot.legend(loc='best')	
	
	#show plot
	path = "plot/optilime/"+data+"/"+data+"OptilimeAdherence"+str(adherence)+".png"
	pyplot.savefig(path)
	

