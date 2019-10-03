#Predict digits from the MNIST hand drawn digits dataset

import pandas as pd
import numpy as np

#Load data
#training_data_filename = './train.csv'
training_data_filename = './trainlight.csv'
training_data = pd.read_csv(training_data_filename)

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
print(training_data.head())
