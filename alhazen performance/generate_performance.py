import sys
import os
import getopt
from termcolor import colored
from pathlib import Path
import numpy as np
from tqdm import tqdm
import pandas as pd
import subprocess


#handle inputs
def main(argv):
	arg_help = ("\n{0} -t <type> -f <foldername> \ntype: type of the model (toydata, heartfailuremax3, heartfailureWithSamplesmax3, heartfailuremax4, diabetesmax3, diabetesmax4, heartfailuremax4cycle100) \nfoldername: name of the folder where to find all model folders".format(argv[0]))
	arg_type = ""
	arg_folder = ""
	
	try:
		opts, args = getopt.getopt(argv[1:], "ht:f:", ["help", "type=", "folder="])
	except:
		print(arg_help)
		sys.exit(2)
	
	for opt, arg in opts:
		if opt in ("-h", "--help"):
			print(arg_help)
			sys.exit(2)
		elif opt in ("-t", "--type"):
			if arg in ("toydata", "heartfailuremax3", "heartfailureWithSamplesmax3", "heartfailuremax4", "diabetesmax3", "diabetesmax4", "heartfailuremax4cycle100"):
				arg_type = arg
			else:
				print('Type has to be in: "toydata", "heartfailuremax3", "heartfailureWithSamplesmax3", "heartfailuremax4", "diabetesmax3", "diabetesmax4", "heartfailuremax4cycle100"')
				sys.exit(2)
		elif opt in ("-f", "--foldername"):
			if os.path.isdir(arg):
				arg_folder = arg
			else:
				print('No folder "'+arg+'"!')
				sys.exit(2)		
								
	return arg_type, arg_folder
	
if __name__ == "__main__":
	arg_type, arg_folder = main(sys.argv)

processes = []
subjectfolder = ""
subjectname = []

#find paths of relevant folders
folders = os.listdir(arg_folder)

if arg_type in ["heartfailuremax3", "heartfailuremax4", "heartfailuremax4cycle100"]:
	subjectfolder = "heartfailure"
	subjectname.append("heartfailure.py")
elif arg_type in ["diabetesmax3", "diabetesmax4"]:
	subjectfolder = "diabetes"
	subjectname.append("diabetes.py")
elif arg_type == "toydata":
	subjectfolder = "toydata"
	subjectname.append("toydata.py")
elif arg_type == "heartfailureWithSamplesmax3":
	subjectfolder = "heartfailureWithSamples"
	testsamplenumber = []
	for i in range(len(folders)):
		if "Samples1_" in folders[i]:
			subjectname.append("heartfailure1.py")
			testsamplenumber.append("1")
		elif "Samples2" in folders[i]:
			subjectname.append("heartfailure2.py")
			testsamplenumber.append("2")
		elif "Samples3" in folders[i]:
			subjectname.append("heartfailure3.py")
			testsamplenumber.append("3")
		elif "Samples4" in folders[i]:
			subjectname.append("heartfailure4.py")
			testsamplenumber.append("4")
		elif "Samples5" in folders[i]:
			subjectname.append("heartfailure5.py")
			testsamplenumber.append("5")
		elif "Samples6" in folders[i]:
			subjectname.append("heartfailure6.py")
			testsamplenumber.append("6")
		elif "Samples7" in folders[i]:
			subjectname.append("heartfailure7.py")
			testsamplenumber.append("7")	
		elif "Samples8" in folders[i]:
			subjectname.append("heartfailure8.py")
			testsamplenumber.append("8")	
		elif "Samples9" in folders[i]:
			subjectname.append("heartfailure9.py")
			testsamplenumber.append("9")	
		elif "Samples10_" in folders[i]:
			subjectname.append("heartfailure10.py")
			testsamplenumber.append("10")	
	
for i in range(len(folders)):

	if arg_type == "heartfailureWithSamplesmax3":
		processes.append(subprocess.Popen(['./alhazen_tree', '--bug-module', '../../../subjects/'+subjectfolder+'/'+subjectname[i], arg_folder+'/'+folders[i]+'/', 'predict', '--samples', 'testsamples/heartfailure/test'+testsamplenumber[i]+'/test'+testsamplenumber[i]+'/', '--csv', 'performance/'+arg_folder+'/'+folders[i]+'.csv', '--execute'], 
										  stdout=subprocess.DEVNULL,
										  stderr=subprocess.STDOUT,#subprocess.PIPE,
										  universal_newlines=True
										 )
						)
						
	elif arg_type == "toydata":
		processes.append(subprocess.Popen(['./alhazen_tree', '--bug-module', '../../../subjects/'+subjectfolder+'/'+subjectname[0], arg_folder+'/'+folders[i]+'/', 'predict', '--samples', 'testsamples/toydata/test1/', '--csv', 'performance/'+arg_folder+'/'+folders[i]+'.csv', '--execute'], 
										  stdout=subprocess.DEVNULL,
										  stderr=subprocess.STDOUT,#subprocess.PIPE,
										  universal_newlines=True
										 )
						)
						
	elif arg_type in ["heartfailuremax3", "heartfailuremax4", "heartfailuremax4cycle100"]:
		processes.append(subprocess.Popen(['./alhazen_tree', '--bug-module', '../../../subjects/'+subjectfolder+'/'+subjectname[0], arg_folder+'/'+folders[i]+'/', 'predict', '--samples', 'testsamples/heartfailure/testbase/testbase/', '--csv', 'performance/'+arg_folder+'/'+folders[i]+'.csv', '--execute'], 
										  stdout=subprocess.DEVNULL,
										  stderr=subprocess.STDOUT,#subprocess.PIPE,
										  universal_newlines=True
										 )
						)
	elif arg_type in ["diabetesmax3", "diabetesmax4"]:
		processes.append(subprocess.Popen(['./alhazen_tree', '--bug-module', '../../../subjects/'+subjectfolder+'/'+subjectname[0], arg_folder+'/'+folders[i]+'/', 'predict', '--samples', 'testsamples/diabetes/test1/', '--csv', 'performance/'+arg_folder+'/'+folders[i]+'.csv', '--execute'], 
										  stdout=subprocess.DEVNULL,
										  stderr=subprocess.STDOUT,#subprocess.PIPE,
										  universal_newlines=True
										 )
						)
count = 0
while True:
	
	remove = []
		
	for i in range(len(processes)):
		return_code = processes[i].poll()
		if return_code is not None:
			count += 1
			print(str(count)+' evaluations done!')
			remove.append(i)
			break
			
	for x in remove:
		del processes[x]
	
	if not processes:
		break



