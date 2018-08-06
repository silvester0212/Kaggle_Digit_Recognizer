import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys

def display(df, index):

	# Width Size of Image
	row = 28

	# Height Size of Image
	col = 28

	# Columns list
	headers = df.columns.tolist()

	# Columns for Pixels only
	pixels_cols = headers[1:785]

	# Get Image
	image = np.array((df.loc[index, pixels_cols]).tolist())
	image.shape = (row, col)

	# Display Image
	plt.imshow(image, cmap="gray")
	plt.show()

	return;


imageNo = int(sys.argv[1])
training = pd.read_csv(	"./Data/train.csv", 
						delimiter = ',',
						header = 0, 
						dtype = int, 
						skipinitialspace = True,
						nrows = imageNo + 1
						)
display(df = training, index = imageNo)