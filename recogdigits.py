#Predict digits from the MNIST hand drawn digits dataset

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

########################### BEGIN FUNCTION DEFINITIONS 

def write_proportion_of_rows_to_file(df,write_filename,perc_rows_keep = 1):
	"Write an approximate percentage of rows from a dataframe to a file"
	n_rows_full = df.shape[0]
	n_rows_keep = np.ceil(perc_rows_keep*n_rows_full).astype(int)
	df_lightweight = df.iloc[0:(n_rows_keep-1),:]
	df_lightweight.to_csv(write_filename)
	#Could refactor to return status code
	return None

########################### END FUNCTION DEFINITIONS

########################### BEGIN LOAD DATA

#Set options for what to do
is_lighten_data = False
is_plot_data = False
is_fit_predict = True

#If should remove digits, do so first
if is_lighten_data:
	training_data_filename = './train.csv'
	training_data = pd.read_csv(training_data_filename)
	
	test_data_filename = './test.csv'
	test_data = pd.read_csv(test_data_filename)
	
	#Remove some columns and write to file to enable a lightweight input file
	lightweight_training_data_filename = './trainlight.csv'
	perc_keep_training = 0.03
	write_proportion_of_rows_to_file(training_data,lightweight_training_data_filename,
	perc_rows_keep = perc_keep_training)
	
	lightweight_test_data_filename = './testlight.csv'
	perc_keep_test = 0.2
	write_proportion_of_rows_to_file(test_data,lightweight_test_data_filename,
	perc_rows_keep = perc_keep_test)
	
else:
	training_data_filename = './trainlight.csv'
	training_data = pd.read_csv(training_data_filename,index_col = 0)
	test_data_filename = './testlight.csv'
	test_data = pd.read_csv(test_data_filename,index_col = 0)

#test_data_filename = './test.csv'
#test_data = pd.read_csv(test_data_filename)

#Get the new shape of the data file. Inspect the first few elements.
print('Shape of training data: ',training_data.shape)
print(training_data.head(3))

#Separate target and predictor columns
X_train = training_data.copy()
y_train = training_data['label'].ravel()
X_train.drop('label',1,inplace=True)

X_test = test_data.copy()

############################ END LOAD DATA

################################ BEGIN PLOTTING
if is_plot_data:
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

	#Check distribution of digits
	sns.countplot(x='label',data=training_data)
	plt.show()

################################ END PLOTTING

################################ BEGIN FITTING AND PREDICTION
if is_fit_predict:
	C = 1.0 #Regularisation parameter
	multi_class = 'ovr'
	penalty = 'l2'
	fit_intercept = True
	tol = 1e-3
	max_iter = 100
	solver = 'liblinear'

	lr_classifier = LogisticRegression(C=C,multi_class=multi_class,penalty = penalty,fit_intercept = fit_intercept, max_iter = max_iter, tol=tol, solver = solver)

	lr_classifier.fit(X_train,y_train)

	#Do predictions
	y_pred = lr_classifier.predict(X_train)

	print(classification_report(y_train,y_pred))

################################ END FITTING AND PREDICTION


