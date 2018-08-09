import numpy as np
import pandas as pd

def samplePrt(df, row, col):
	indexes = list(range(0, (row + 1)))
	# print(indexes)
	columns = (df.columns.tolist())[0 : (col + 1)]
	# print(columns)

	sample = df[columns].head(row + 1)
	print(sample)

	return;

training = pd.read_csv(	"./Data/train.csv", 
						delimiter = ',',
						header = 0, 
						dtype = int, 
						skipinitialspace = True,
						nrows = 100
						)

samplePrt(training, 10, 10)