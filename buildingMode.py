#	Install package:
#	pandas / numpy / pytz	-	pip3 install pandas
#	matplotlib				-	pip3 install matplotlib
#	sklearn					-	pip3 install scikit-learn
#	scipy					-	pip3 install scipy

#	import csv

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn import neighbors

from readDataFromCSV import readDataFromCSV
from displayImage import display
from samplePrt import samplePrt

# Flag to indicate if script is in building mode or in production mode
buildingStage = True

data = data = pd.read_csv(	"./Data/train.csv", 
						delimiter = ',',
						header = 0, 
						dtype = int, 
						skipinitialspace = True,
						# nrows = 10000
						)

nsamples, ncols = data.shape
headers = data.columns.tolist()
pixel_cols = headers[1 : ncols]
label_col  = headers[0]

data_X  = data[pixel_cols]
data_Y  = data[label_col]


# Split Data into Training vs Test at 7 : 3 ratio
train_data, test_data = sklearn.model_selection.train_test_split(data, test_size = 0.3, shuffle = True)

# Obtain X and Ys of training / test data sets
train_X = train_data[pixel_cols]
train_Y = train_data[label_col]
test_X  = test_data[pixel_cols]
test_Y  = test_data[label_col]

KNN_classifier = sklearn.neighbors.KNeighborsClassifier(
						n_neighbors=20, weights='distance', algorithm='brute'
						, leaf_size=30, p=2, metric='minkowski', n_jobs=1 
						)

KNN_classifier.fit(train_X, train_Y)
print(KNN_classifier.score(test_X, test_Y))
