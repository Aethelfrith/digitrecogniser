#Predict digits from the MNIST hand drawn digits dataset

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

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
#classifier_method = None
classifier_method = 'logreg'
#Classifier method either None, logreg, 

#If should remove digits, do so first
if is_lighten_data:
	trainval_data_filename = './train.csv'
	trainval_data = pd.read_csv(trainval_data_filename)
	
	test_data_filename = './test.csv'
	test_data = pd.read_csv(test_data_filename)
	
	#Remove some columns and write to file to enable a lightweight input file
	lightweight_trainval_data_filename = './trainlight.csv'
	perc_keep_trainval = 0.1
	write_proportion_of_rows_to_file(trainval_data,lightweight_trainval_data_filename,
	perc_rows_keep = perc_keep_trainval)
	
	lightweight_test_data_filename = './testlight.csv'
	perc_keep_test = 0.2
	write_proportion_of_rows_to_file(test_data,lightweight_test_data_filename,
	perc_rows_keep = perc_keep_test)
	
else:
	trainval_data_filename = './trainlight.csv'
	trainval_data = pd.read_csv(trainval_data_filename,index_col = 0)
	test_data_filename = './testlight.csv'
	test_data = pd.read_csv(test_data_filename,index_col = 0)


#Get the shape of the data file. Inspect the first few elements.
print('Shape of training data: ',trainval_data.shape)
print(trainval_data.head(3))

#Separate target and predictor columns
X_trainval = trainval_data.copy()
y_trainval = trainval_data['label'].ravel()
X_trainval.drop('label',1,inplace=True)

valid_size = 0.2
tts_random_state = None
X_train, X_valid, y_train, y_valid = train_test_split(X_trainval, y_trainval, test_size = valid_size, random_state = tts_random_state)

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
if classifier_method == 'logreg':
	C = 1 #Regularisation parameter
	multi_class = 'ovr'
	penalty = 'l1'
	fit_intercept = True
	tol = 1e-3
	max_iter = 100
	solver = 'liblinear'

	lr_classifier = LogisticRegression(C=C,multi_class=multi_class,penalty = penalty,fit_intercept = fit_intercept, max_iter = max_iter, tol=tol, solver = solver)

	lr_classifier.fit(X_train,y_train)

	#Do predictions
	y_pred = lr_classifier.predict(X_valid)

	print(classification_report(y_valid,y_pred))
elif classifier_method is None:
	#Donothing
else:
	print("Invalid classifier method string passed") 

################################ END FITTING AND PREDICTION


