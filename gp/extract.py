import numpy as np
import csv
import sys
import pandas as pd
from pdb import set_trace

def extract_data(filename):
	df = pd.read_csv(filename)
	finger_y = df['finger_y']
	return np.array(finger_y)

def prune_negatives(data):
	pruned_data = []
	for datum in data:
		if datum >= 0:
			pruned_data.append(datum)
	return pruned_data