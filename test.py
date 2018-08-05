print ("Hello World!")

#	Install package:
#	pandas / numpy / pytz	-	pip3 install pandas
#	matplotlib				-	pip3 install matplotlib

#	import csv

import matplotlib.pyplot as plt
import numpy as np
from numpy import genfromtxt

row = 28
col = 28

index = 1

training = genfromtxt('./Data/train_sample.csv', delimiter = ',')


# label = training[index, 0]
# data = training[index, [1,2,3,4,5]]
# print(label)
# print(data)


#X = np.random.random((100, 100)) # sample 2D array
#plt.imshow(X, cmap="gray")
#plt.show()

