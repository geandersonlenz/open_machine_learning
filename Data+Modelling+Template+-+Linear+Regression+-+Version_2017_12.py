
# coding: utf-8

# In[2]:


print(__doc__)


# Code source: Jaques Grobler
# License: BSD 3 clause


import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
from sklearn.model_selection import train_test_split
# mlxtend - http://rasbt.github.io/mlxtend/
# scikit plot - http://scikit-plot.readthedocs.io/en/stable/index.html
get_ipython().magic('matplotlib inline')


# In[3]:


# create a x variable with 2 columns and size 100
X = np.random.randint(100, size=(100, 3))
# create a x variable with 2 columns and size 100
y = np.random.randint(100, size=(100, 1))


# In[4]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)


# In[5]:


# Create linear regression object
regr = linear_model.LinearRegression()


# In[6]:



# Train the model using the training sets
regr.fit(X_train, y_train)

# Make predictions using the testing set
y_pred = regr.predict(X_test)


# In[7]:


plt.plot(y_pred)
plt.ylabel("Values of Predict")
plt.xlabel("Predicted Values")
plt.title("Model Predicted");


# In[206]:



# The coefficients
print('Coefficients: \n', regr.coef_)
# The mean squared error
print("Mean squared error: %.2f"
      % mean_squared_error(y_test, y_pred))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(y_test, y_pred))
# Show the parameter of model
print('Parâmetros do Modelo:')
print(regr.get_params())


# In[207]:


import matplotlib.pyplot as plt
from mlxtend.plotting import plot_linear_regression, 
import numpy as np

X_ = np.array(X_test[:, 0])
y_ = np.array(y_test[:,0])


intercept, slope, corr_coeff = plot_linear_regression(X_, y_)

plt.show();


# In[10]:


fig, ax = plt.subplots(figsize=(10, 8))
ax.plot(y_pred, 'b--', label='y predicted', lw=5)
ax.plot(y_test, 'r--', label='y test', lw=4)
ax.legend(loc=1);


# In[209]:


from sklearn import linear_model

regr = linear_model.LinearRegression()
ridge = linear_model.Ridge (alpha = .5)
sgd = linear_model.SGDRegressor()
logistic = linear_model.LogisticRegression()


# In[210]:


regression = [regr, ridge, sgd, logistic]


# In[211]:


[x.fit(X,y) for x in regression]


# In[212]:


y_pred = [x.predict(X_test) for x in regression]


# In[213]:


from sklearn.metrics import mean_squared_error, r2_score
for x in y_pred:
    print('R2 Score:%.3f' % r2_score(y_test, x, multioutput='raw_values'))


# In[218]:


from sklearn import linear_model


# prepare configuration for cross validation test harness
seed = 7
# prepare models
models = []
models.append(('LR', linear_model.LogisticRegression()))
models.append(('LMR', linear_model.LinearRegression()))
models.append(('RIDGE', linear_model.Ridge (alpha = .5)))
models.append(('SGD', linear_model.SGDRegressor()))


# In[259]:


from sklearn.model_selection import cross_val_score, KFold

# evaluate each model in turn
results = []
names = []
scoring = 'lasso'
for name, model in models:
	kfold = KFold(n_splits=10, random_state=seed)
	cv_results = model.fit(X_train, y_train.ravel())
	results.append(model.predict(X_test))
	names.append(name)
	msg = "%s: %f (%f)" % (name, cv_results.score(X, y), cv_results.score(X, y).std())
	print(msg)


# In[260]:


# boxplot algorithm comparison
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()

