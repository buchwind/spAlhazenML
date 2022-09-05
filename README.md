# spAlhazenML
Study project: Using Alhazen to explain  behaviour of black box machine learning models

To do your own experiments:

1) build Alhazen
2) create subject in alhazen-dbgbench/subjects
3) run alhazen in alhazen-dbgbench/parsers/build/dist to create explanation (./alhazen -h for more info)  
example: ./alhazen ../../../subjects/toydata/toydata.py ./toycomplex12345678 -s 12345678
4) extract the final decision tree .pickle files, rename and place into surrogate/"name" subfolder in evaluation code folder, also copy one depth.csv containing the feature names
5) put generate_performance.py from alhazen performance folder into  alhazen-dbgbench/parsers/build/dist
6) create testsamples and performance folder in /dist
7) generate samples to test performance of explanation (set_generator in /dist to generate samples)  
example: ./set_generator -g "MoreOfTheSame(pseudocount(1):True)" -s 12345678 ../../../subjects/diabetes/diabetes.py samples/diabetessamples12345678
8) for the next step and later evaluations the code needs to be modified (work through the code and modify/add where specific experiments are mentioned)
9) to balance the sample set copy the samples to evaluation code samples/raw/"set" subfolder and run like example  
example: python3 blackboxes.py -g "set"
10) copy balanced test set to /dist/testsamples/"set"/test1
11) modify generate_performance.py for your experiment
12) generate performance .csv's with generate_performance.py  
example: python3 generate_performance.py -t toydata -f toydata (-t: type defined in generate_performance.py to select correct subject -f: foldername of folder containing the Alhazen runs)
13) copy generated .csv's to evaluation code surrogate/"name"/performance subfolder

Evaluation code:

-the data sets can be found in the dataset folder  
-blackboxes contains the trained models in .joblib form  
-plot contains plots from all surrogate models  
-samples contains the raw sample sets created by set_generator in /dist and balanced training and test sets derived from them  
-surrogate contains all the created surrogate models in .pickle or .joblib form and performance .csv's for the surrogates created by Alhazen  

-blackboxes.py: main file to execute  
-dataload.py: load/prepare data sets, load .pickle/.joblib files, generate balanced data sets  
-trainmodel.py: train black boxes and global surrogate models  
-modelevaluation.py: evaluate black box performance and find best training parameters for black boxes  
-normalLIME.py: generate basic LIME explanation  
-OptiLIME: generate OptiLIME explanation  
-stability_utils.py/utils.py: needed for OptiLIME and taken from original implementation at https://github.com/giorgiovisani/LIME_stability  
-stability.py: calculate VSI/FSI for LIME, OptiLIME and decision trees  
-ploter.py: plot decision trees and OptiLIME adherence results  

for info of usage: python3 blackboxes.py -h  
note: to plot decision trees they have to be in the plot/ subfolder  

-the subjects folder contains the used subjects  
-because of size, the complete contents of the Alhazen /dist folder, with the intermediate steps of the explanation generation, can be found at ...