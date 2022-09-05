#!/bin/python3

"""
This file contains the driver, that is, the code which allows alhazen to execute the target program.
"""
import logging
from pathlib import Path
from alhazen import bug_class
from alhazen import oracles
import pandas
from joblib import load
from ast import literal_eval

saved_model = load(Path(__file__).parent / "heart.joblib")

class HeartFailureBug(bug_class.Bug):
	"""
	This class represents the bug alhazen is looking for.
	It inherits from bug_class.Bug, which contains some more methods alhazen is going to use.
	"""

	def __init__(self):
		super().__init__()
		self.saved_model = saved_model

	def subject(self):
		"""returns a name for the program under test. """
		return "heartfailure"

	def grammar_file(self):
		"""returns the path to the grammar file."""
		return Path(__file__).parent / "grammar.scala"

	def execute_sample_list(self, execdir, samples):
		"""
		This method executes all samples in the given list.
		The return value is a pandas DataFrame. Please have a look at lines 44ff
		to see which fields are required.

		:param execdir: The directory to use as the working dir for all executions.
		:param samples: The list of samples to be executed.
		:return: A pandas DataFrame which contains information about each subject.
		"""
		data = []
		for sample in samples:
			res = self.__execute_sample(sample)
			data.append({
				"file": sample.name,
				"line": sample,
				"subject": self.subject(),
				"oracle": res
			})
		return pandas.DataFrame.from_records(data)

	def __execute_sample(self, file):
		"""
		This is a helper method used within execute_sample_list.
		If you implement execute_sample_list without using execute_sample,
		you do not need it for your own bugs.

		:param file: The input file to be executed.
		:return: the oracle result.
		"""

		with open(file, 'r') as testcode:
			testcode = testcode.read()

		#print(testcode)

		prediction=saved_model.predict(literal_eval(testcode))
		
		#print(prediction)

		if prediction[0] == 0:
			return oracles.OracleResult.NO_BUG
		elif prediction[0] == 1:
			return oracles.OracleResult.BUG
		else:
			return oracles.OracleResult.UNDEF
		
	def sample_files(self):
		"""
		This method is a generator method (using python's yield keyword), which iterates all samples for this bug.
		"""
		for i in range(1000):
			name = "heartfailuretrain."+str(i+1)+".expr"
			yield Path(__file__).parent / "samples3" / name

def create_bug():
    return HeartFailureBug()


if __name__ == "__main__":
    import sys
    bug = create_bug()
    data = bug.execute_samples(Path(sys.argv[1]))
    #data = bug.execute_samples(Path(__file__).parent / "samples")
    print(data["oracle"].head())
