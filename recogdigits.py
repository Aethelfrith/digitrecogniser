#Predict digits from the MNIST hand drawn digits dataset

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 

########################### BEGIN LOAD DATA
#Load data
#training_data_filename = './train.csv'
training_data_filename = './trainlight.csv'
training_data = pd.read_csv(training_data_filename,index_col = 0)

#test_data_filename = './test.csv'
#test_data = pd.read_csv(test_data_filename)

##Remove some columns and write to file to enable a lightweight input file
#n_digits_full = training_data.shape[0]
#perc_digits_keep = 0.1
#n_digits_keep = np.ceil(perc_digits_keep*n_digits_full).astype(int)
#training_data_lightweight = training_data.iloc[0:(n_digits_keep-1),:]
#lightweight_training_data_filename = './trainlight.csv'
#training_data_lightweight.to_csv(lightweight_training_data_filename)

#Get the new shape of the data file. Inspect the first few elements.
print('Shape of training data: ',training_data.shape)
print(training_data.head(3))

#Separate target and predictor columns
X_train = training_data.copy()
y_train = training_data['label'].ravel()
X_train.drop('label',1,inplace=True)


############################ END LOAD DATA

################################ BEGIN PLOTTING
#Plot a subset of the data

#Set aesthetic theme
sns.set_style('dark')

#Create 10 subplots
n_sp_row = 2
n_sp_col = 5
fig,axes = plt.subplots(n_sp_row,n_sp_col,figsize = (n_sp_col,n_sp_row))

n_digits_show = n_sp_row*n_sp_col
for i in range(n_sp_row):
	for j in range(n_sp_col):
		axes[i][j].imshow(X_train.values[(i*n_sp_row + j),:].reshape(28,28),
		cmap='gray_r',interpolation='nearest')
		axes[i][j].set_yticks([])
		axes[i][j].set_xticks([])
plt.show()

################################ END PLOTTING
