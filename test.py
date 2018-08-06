#	Install package:
#	pandas / numpy / pytz	-	pip3 install pandas
#	matplotlib				-	pip3 install matplotlib

#	import csv

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Image Index to test
index = 35


training = pd.read_csv(	"./Data/train.csv", 
						delimiter = ',',
						header = 0, 
						dtype = int, 
						skipinitialspace = True,
						nrows = 100
						)

headers = training.columns.tolist()
pixels_cols = headers[1:785]
image = np.array((training.loc[index, pixels_cols]).tolist())
image.shape = (row, col)

plt.imshow(image, cmap="gray")
plt.show()

