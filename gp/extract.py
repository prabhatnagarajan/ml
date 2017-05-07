import numpy as np
import csv
import sys
import pandas as pd
from pdb import set_trace

def extract_data(filename):
	df = pd.read_csv(filename)
	elapsed_time = df['elapsed_time']
	finger_y = df['finger_y']
	# return (np.array(elapsed_time), np.array(finger_y))
	return prune_negatives(elapsed_time, finger_y)

def prune_negatives(time, data):
	pruned_data = []
	pruned_time = []
	for i in range(len(data)):
		datum = data[i]
		if datum >= 0:
			pruned_data.append(datum)
			pruned_time.append(time[i])
		else:
			print "Negative"
	return (np.array(pruned_time), np.array(pruned_data))