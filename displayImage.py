import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def display(df, index, pixels_colnames):

	# Width Size of Image
	row = 28

	# Height Size of Image
	col = 28

	# Get Image
	image = np.array((df.loc[index, pixels_colnames]).tolist())
	image.shape = (row, col)

	# Display Image
	plt.imshow(image, cmap="gray")
	plt.show()

	return;


# imageNo = int(sys.argv[1]) - 1
# # imageNo = 0
# training = pd.read_csv(	"./Data/test.csv", 
# 						delimiter = ',',
# 						header = 0, 
# 						dtype = int, 
# 						skipinitialspace = True,
# 						nrows = imageNo + 1
# 						)
# display(df = training, index = imageNo, pixel_col_start = 0, pixel_col_end = 784)