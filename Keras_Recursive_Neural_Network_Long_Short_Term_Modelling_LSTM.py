
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
get_ipython().magic('matplotlib inline')


# In[2]:


df = pd.read_csv('international-airline-passengers.csv', usecols=[1], engine='python', skipfooter=3)


# In[3]:


df.head()


# In[4]:


df.plot(figsize=(16, 8));


# In[5]:


df.columns = ['passengers']


# In[6]:


df.head()


# In[7]:


import numpy
import matplotlib.pyplot as plt
import pandas
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error


# RandomState exposes a number of methods for generating random numbers drawn from a variety of probability distributions. 
# In addition to the distribution-specific arguments, each method takes a keyword argument size that defaults to None. 
# If size is None, then a single value is generated and returned. If size is an integer, then a 1-D array filled with generated values is returned. 
# If size is a tuple, then an array with that shape is filled and returned.

# In[8]:


# fix random seed for reproducibility
# This method is called when RandomState is initialized. It can be called again to re-seed the generator. 
numpy.random.seed(7)


# We can also use the code from the previous section to load the dataset as a Pandas dataframe. 
# We can then extract the NumPy array from the dataframe and convert the integer values to floating point values, 
# which are more suitable for modeling with a neural network.

# In[9]:


# load the dataset
dataframe = pandas.read_csv('international-airline-passengers.csv', usecols=[1], engine='python', skipfooter=3)
dataset = dataframe.values
dataset = dataset.astype('float32')


# LSTMs are sensitive to the scale of the input data, specifically when the sigmoid (default) or tanh activation functions are used. It can be a good practice to rescale the data to the range of 0-to-1, also called normalizing. 
# We can easily normalize the dataset using the MinMaxScaler preprocessing class from the scikit-learn library.

# In[10]:


# normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)


# With time series data, the sequence of values is important. 
# A simple method that we can use is to split the ordered dataset into train and test datasets. 
# The code below calculates the index of the split point and separates the data into the training datasets with 67%
# of the observations that we can use to train our model, leaving the remaining 33% for testing the model.

# In[11]:


# split into train and test sets
train_size = int(len(dataset) * 0.67)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
print(len(train), len(test))


# The function takes two arguments: the dataset, which is a NumPy array that we want to convert into a dataset, and the look_back, which is the number of previous time steps to use as input variables to predict the next time period — in this case defaulted to 1.
# 
# This default will create a dataset where X is the number of passengers at a given time (t) and Y is the number of passengers at the next time (t + 1).

# In[12]:


# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back-1):
		a = dataset[i:(i+look_back), 0]
		dataX.append(a)
		dataY.append(dataset[i + look_back, 0])
	return numpy.array(dataX), numpy.array(dataY)


# If you compare these first 5 rows to the original dataset sample listed in the previous section, you can see the X=t and Y=t+1 pattern in the numbers.

# In[13]:


# reshape into X=t and Y=t+1
look_back = 1
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)


# We are now ready to design and fit our LSTM network for this problem.
# 
# The network has a visible layer with 1 input, a hidden layer with 4 LSTM blocks or neurons, and an output layer that makes a single value prediction. The default sigmoid activation function is used for the LSTM blocks. The network is trained for 100 epochs and a batch size of 1 is used.

# The LSTM network expects the input data (X) to be provided with a specific array structure in the form of: [samples, time steps, features].
# 
# Currently, our data is in the form: [samples, features] and we are framing the problem as one time step for each sample. We can transform the prepared train and test input data into the expected structure using numpy.reshape() as follows:

# In[14]:


# reshape input to be [samples, time steps, features]
trainX = numpy.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))


# The network has a visible layer with 1 input, a hidden layer with 4 LSTM blocks or neurons, and an output layer that makes a single value prediction. The default sigmoid activation function is used for the LSTM blocks. The network is trained for 100 epochs and a batch size of 1 is used.
# 
# RMSE
# 
# In statistics, the mean squared error (MSE) or mean squared deviation (MSD) of an estimator (of a procedure for estimating an unobserved quantity) measures the average of the squares of the errors or deviations—that is, the difference between the estimator and what is estimated. MSE is a risk function, corresponding to the expected value of the squared error loss or quadratic loss. The difference occurs because of randomness or because the estimator doesn't account for information that could produce a more accurate estimate.
# 
# The MSE is a measure of the quality of an estimator—it is always non-negative, and values closer to zero are better.
# 
# ADAM OPTIMIZER
# 
# The choice of optimization algorithm for your deep learning model can mean the difference between good results in minutes, hours, and days.
# 
# The Adam optimization algorithm is an extension to stochastic gradient descent that has recently seen broader adoption for deep learning applications in computer vision and natural language processing.
# 
# EPOCHS
# 
# In machine-learning parlance, an epoch is a complete pass through a given dataset. That is, by the end of one epoch, your neural network – be it a restricted Boltzmann machine, convolutional net or deep-belief network – will have been exposed to every record to example within the dataset once.
# 
# BATCH SIZE (BATCH NORMALIZATION)
# 
# Batch Normalization does what is says: it normalizes mini-batches as they’re fed into a neural-net layer. Batch normalization has two potential benefits: it can accelerate learning because it allows you to employ higher learning rates, and also regularizes that learning.
# 
# VERBOSE 
# 
# Verbose is a general programming term for produce lots of logging output. You can think of it as asking the program to "tell me everything about what you are doing all the time". Just set it to true and see what happens.
# 
# 

# In[15]:


# create and fit the LSTM network
model = Sequential()
# add LSTM with 4 blocks
model.add(LSTM(4, input_shape=(1, look_back)))
# output layer that makes a single value prediction
model.add(Dense(1))
# RMSE for estimators performance measure and ADAM for optimizer the update weights in training data.
model.compile(loss='mean_squared_error', optimizer='adam')
# mini Batch Size is default == 1
model.fit(trainX, trainY, epochs=100, batch_size=1, verbose=2)


# Once the model is fit, we can estimate the performance of the model on the train and test datasets. This will give us a point of comparison for new models.
# 
# Note that we invert the predictions before calculating error scores to ensure that performance is reported in the same units as the original data (thousands of passengers per month).

# In[16]:


# make predictions
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)
# invert predictions
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])
# calculate root mean squared error
trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
print('Test Score: %.2f RMSE' % (testScore))


# In[17]:


# shift train predictions for plotting
trainPredictPlot = numpy.empty_like(dataset)
trainPredictPlot[:, :] = numpy.nan
trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict
# shift test predictions for plotting
testPredictPlot = numpy.empty_like(dataset)
testPredictPlot[:, :] = numpy.nan
testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = testPredict
# plot baseline and predictions
plt.plot(scaler.inverse_transform(dataset))
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.show()

