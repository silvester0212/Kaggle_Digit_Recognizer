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
from sklearn import neighbors

# Flag to indicate if script is in building mode or in production mode
buildingStage = False

data = pd.read_csv(	filepath_or_buffer = "./Data/train.csv", 
					delimiter = ',',
					header = 0, 
					dtype = int, 
					skipinitialspace = True
					)

nsamples, ncols = data.shape
headers = data.columns.tolist()
pixel_cols = headers[1 : ncols]
label_col  = headers[0]

# Get Pixels and Labels
data_X  = data[pixel_cols]
data_Y  = data[label_col]


KNN_classifier = sklearn.neighbors.KNeighborsClassifier(
						n_neighbors=20, weights='distance', algorithm='brute'
						, leaf_size=30, p=2, metric='minkowski', n_jobs=1 
						)
KNN_classifier.fit(data_X, data_Y)

# Read CSV from Kaggle test dataset
kaggle_test = pd.read_csv(	filepath_or_buffer = "./Data/test.csv", 
							delimiter = ',',
							header = 0, 
							dtype = int, 
							skipinitialspace = True
						)

# Get Prediction values from prev trained model
knn_predicted_value = KNN_classifier.predict(kaggle_test)


# Test pool size
ntest = kaggle_test.shape[0]
predictionDF = pd.DataFrame({"ImageId" : range(1, ntest + 1), "Label" : knn_predicted_value})

# Write predictions to CSV
predictionDF.to_csv(path_or_buf = "./Data/Prediction.csv", sep = ',', header = True, index = False)
