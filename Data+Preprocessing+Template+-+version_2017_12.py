
# coding: utf-8

# # Data Preprocessing

# Data Preprocessing modelling with all the automatic features for preparing the dataset for running models:
#     - Initialize
#       Loading Files
#     - Data Overview
#         Describe, Head, Shape, Types
#     - Data Cleaning
#         Handling Missing Data
#         Outlier Detection
#         Handling Categorical Data
#         Dummy Variables
#     - Data Analysis    
#         Plotting Distributions
#     - Data Wrangling
#         Slicing, Selecting, Aggregating
#         Concatenate, Merge, Join
#         Label Encoder
#         One Hot Encoder (OHE)
#         Label Binarizer (if necessary)
#         Normalizer/Standarlizer
#     - Featuring Engineering:
#         Feature Extraction
#         Dimensionality Reduction by PCA
#         Feature Selection

# ## Initialize

# In[2]:


# adicionado na versão de 11_2017
# %load_ext autotime
# %lsmagic
import pandas as pd         #pandas data manipulation
import numpy as np          #numpy arrays manipulation 
import missingno as msno    #missingno missing values
import gc                   #ram optimization
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
get_ipython().magic('matplotlib inline')


# In[3]:


# Setting working directory

path = '../MOZAIC/'


# ## Loading Files

# In[4]:


# read the data from csv
df = pd.read_csv('HR_comma_sep.csv')


# # Data Overview

# In[5]:


df.describe()


# In[6]:


# Return a tuple representing the dimensionality of the DataFrame.
print("Número de Linhas e Colunas")
df.shape
# view columns using df.columns
print("Apresenta as colunas do Dataset")
df.columns
# Return the dtypes in this object
print("Mostra os tipos de dados de cada coluna do Dataset")
df.dtypes
# Return the missing values in dataframe
print("Número de missing values no Dataframe")
df.isnull().values.any()


# In[7]:


# Returns first n rows
df.head(5)


# In[8]:


# Return the count distinct values in single columns
for i in zip([df.columns.values]):
    print ('{}'.format(i))


# # Descriptive Statistics

# ## Data Overview

# In[6]:


# Generates descriptive statistics that summarize the central tendency
df.describe()


# In[51]:



plt.figure(1, figsize=(15,15), facecolor='y')
plt.subplot(211)
plt.plot(df['satisfaction_level'])

plt.subplot(212)
plt.plot(df['last_evaluation'])
plt.show();


# ## Mean

# In[9]:


df.mean().plot(kind='bar', figsize=(8,8));


# In[10]:


# Return the count distinct values in single columns
for i in zip([df.mean()]):
    print('{}'.format(i))


# ## Skewness e Kurtosis

# In[11]:


# Return the value of skewness and kurtosis of dataframe
print ("skewness:"), df.skew()
print ("kurtosis:"), df.kurtosis() 


# ## Skewness

# In[12]:


# skewness
# skewness 0(normal), -1(left), +1(right)

for i, z in zip(df.skew(), list(df)):
    if i == 0:
        '{} {} {}'.format(z, i, 'is normal')
    elif i < 0:
        '{} {} {}'.format(z, i, 'is left')
    else:
        '{} {} {}'.format(z, i, 'is right')


# ## Kurtosis

# In[13]:


# kurtosis 
# kurtosis 3(normal), >3(upnormal), <3(subnormal)
for i, z in zip(df.kurtosis(), list(df)):
    if i == 3:
        '{} {} {}'.format(z, i, 'is normal')
    elif i < 3:
        '{} {} {}'.format(z, i, 'is subnormal')
    else:
        '{} {} {}'.format(z, i, 'is upnormal')


# ## ScatterMatrix with Histogram and Density

# In[14]:


# Visualize each feature's skewness as well - use seaborn pairplots
sns.pairplot(df, hue='left')


# In[19]:


from pandas.plotting import scatter_matrix
scatter_matrix(df, alpha=0.5, figsize=(12, 12), diagonal='kde');


# In[20]:


fig = plt.figure(figsize=(12,12))
#fig.suptitle("Offensive stats and its impact on Runs Scored and Wins")
ax1 = fig.add_subplot(3,2,1)
sns.regplot(x="satisfaction_level", y="last_evaluation", data=df, scatter=True, marker="+", ax=ax1)
ax1.set_xlabel("Satisfaction Level")
ax1.set_ylabel("Last Evaluation")
sns.despine()
plt.tight_layout()
plt.show();


# ## Histogram of all Dataframe

# In[24]:


plt.figure(figsize=(12,10));

df.plot.hist(alpha=0.5, bins=15);


# In[15]:


# Histogram of all dataframe
df[df.dtypes[(df.dtypes=="float64")|(df.dtypes=="int64")|(df.dtypes=="object")].index.values].hist(figsize=[11,11]);


# ## Boxplot

# In[16]:


# BoxPlot less feature with great size difference
df.loc[:, df.columns != 'average_montly_hours'].plot.box(figsize=[11,11]);


# In[23]:


obp_mean = df['satisfaction_level'].mean()
slg_mean = df['last_evaluation'].mean()


# In[31]:


fig = plt.figure(figsize=(12,12))
ax = fig.add_subplot()
sns.boxplot(x=df['satisfaction_level'], y=df['average_montly_hours'], ax=ax)
plt.title("Satisfaction Level by Last Evaluation")
plt.axvline(obp_mean)
plt.show();


# ## Correlations

# In[21]:


corr_df = df.corr()
corr_df


# In[20]:


# Seaborn's heatmap version:
fig, ax = plt.subplots(figsize=(10,10)) 
corr = df.corr()
sns.heatmap(corr, 
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values,
            ax=ax);


# ## Kernel Density Estimate

# Fit and plot a univariate or bivariate kernel density estimate.
# 
# 

# In[69]:


sns.kdeplot(df.last_evaluation, df.average_montly_hours, shade=True);


# ## Andrews Curve

# In[36]:


df_andrews = df.select_dtypes(exclude=['object']).astype(int)
df_andrews.head()


# In[37]:



from pandas.plotting import andrews_curves
plt.figure(figsize=(12,12))
andrews_curves(df_andrews, 'left');


# ## Hexbin Plot

# In[71]:


x, y = df.average_montly_hours, df.satisfaction_level
fig, ax = plt.subplots()
hx = ax.hexbin(x, y, cmap='Greens', gridsize=10)
fig.colorbar(hx)
plt.show();


# In[34]:


sns.jointplot(x='average_montly_hours', y='satisfaction_level', data=df, kind='hex')
sns.jointplot(x='last_evaluation', y='satisfaction_level', data=df, kind='hex')
sns.jointplot(x='average_montly_hours', y='last_evaluation', data=df, kind='hex')
plt.show();


# ## FacetGrid

# In[53]:


g = sns.FacetGrid(df, row='salary', hue='left')
g = g.map(plt.scatter, 'average_montly_hours', 'last_evaluation', edgecolor="w").add_legend()


# ## LagPlot (Random Analysis)

# In[9]:


data = df.select_dtypes(exclude=['object']).astype(int)


# In[10]:


from pandas.plotting import lag_plot

plt.figure()
lag_plot(data);


# ## Autocorrelation

# Autocorrelation plots are often used for checking randomness in time series. 
# This is done by computing autocorrelations for data values at varying time lags. 
# If time series is random, such autocorrelations should be near zero for any and all time-lag separations. 
# If time series is non-random then one or more of the autocorrelations will be significantly non-zero. 
# The horizontal lines displayed in the plot correspond to 95% and 99% confidence bands. 
# The dashed line is 99% confidence band.

# In[11]:


from pandas.plotting import autocorrelation_plot

plt.figure()
autocorrelation_plot(data);


# # Inferential Statistics

# # Data Transformation

# ## Group By

# By “group by” we are referring to a process involving one or more of the following steps
# 
#     Splitting the data into groups based on some criteria
#     Applying a function to each group independently
#     Combining the results into a data structure
# 
# Of these, the split step is the most straightforward. In fact, in many situations you may wish to split the data set into groups and do something with those groups yourself. In the apply step, we might wish to one of the following:
# 
#     Aggregation: computing a summary statistic (or statistics) about each group. Some examples:
#         Compute group sums or means
#         Compute group sizes / counts
#     
#     Transformation: perform some group-specific computations and return a like-indexed. Some examples:
#         Standardizing data (zscore) within group
#         Filling NAs within groups with a value derived from each group
#     
#     Filtration: discard some groups, according to a group-wise computation that evaluates True or False. Some examples:
#         Discarding data that belongs to groups with only a few members
#         Filtering out data based on the group sum or mean
# 
# Some combination of the above: GroupBy will examine the results of the apply step and try to return a sensibly combined result 
# if it doesn’t fit into either of the above two categories

# In[94]:


get_ipython().magic('matplotlib inline')


# In[ ]:


df.groupby('sales').mean()


# In[ ]:


df.groupby('sales').mean().plot(kind='bar');


# In[ ]:


# Return the number of distinct values in each column
df.groupby(lambda idx: 'number').agg(['nunique']).transpose()


# In[ ]:


df.groupby(lambda idx: 'number').nunique().plot(kind='bar')


# ## Delete and Move

# ## Discretize

# ## Handling Missing Data

# ## Outlier Detection

# Dixon's Q-Test is used to help determine whether there is evidence for a given point to be an outlier of a 1D dataset. 
# It is assumed that the dataset is normally distributed. Since we have very strong evidence that our dataset above is normal 
# from all our normality tests, we can use the Q-Test here. As with the normality tests, we are assuming a significance level of 0.05 and for simplicity, we are only considering the smallest datum point in the set.

# In[ ]:


def q_test_for_smallest_point(dataset):
    q_ref = 0.29  # the reference Q value for a significance level of 95% and 30 data points
    q_stat = (dataset[1] - dataset[0])/(dataset[-1] - dataset[0])
    
    if q_stat > q_ref:
        print("Since our Q-statistic is %f and %f > %f, we have evidence that our "
              "minimum point IS an outlier to the data.") %(q_stat, q_stat, q_ref)
    else:
        print("Since our Q-statistic is %f and %f < %f, we have evidence that our "
              "minimum point is NOT an outlier to the data.") %(q_stat, q_stat, q_ref)


# In[ ]:


dataset = data[100:130]['10 Min Sampled Avg'].values.tolist()
dataset.sort()
q_test_for_smallest_point(dataset)


# Existem valores nulos no dataframe e qual a soma desses valores nulos

# In[19]:


df.isnull().sum().sum()


# In[21]:


import missingno as msno
get_ipython().magic('matplotlib inline')
msno.matrix(df.sample(250))


# In[22]:


msno.bar(df.sample(500))


# In[ ]:


# retirar os valores nulos do dataframe
df_no_missing = df.dropna()


# # Target & Estimators

# In[5]:


# Set the target by dataframe column name 
y = df['left']


# In[6]:


dummy_df = pd.get_dummies(df, prefix='salary', columns=['salary'])


# In[7]:


dummy_df.drop(['sales'], axis=1, inplace=True)


# In[8]:


# Set the estimators by dataframe columns name exclude the target variable
X = dummy_df.loc[:, dummy_df.columns != 'left']


# In[9]:


X.columns


# # Handling categorical data 

# ## Label Encoder

# In[ ]:


# TODO: create a LabelEncoder object and fit it to each feature in X
# import preprocessing from sklearn
from sklearn import preprocessing

# 1. INSTANTIATE
# encode labels with value between 0 and n_classes-1.
le = preprocessing.LabelEncoder()

# 2/3. FIT AND TRANSFORM
# use df.apply() to apply le.fit_transform to all columns
X = X.apply(le.fit_transform)

# 2/3. FIT AND TRANSFORM
# use df.apply() to apply le.fit_transform to all columns
y = y.apply(le.fit_transform)


# ## One Hot Encoder (OHE)

# In[ ]:


# TODO: create a OneHotEncoder object, and fit it to all of X

# 1. INSTANTIATE
enc = preprocessing.OneHotEncoder()

# 2. FIT
enc.fit(X)

# 3. Transform

# onehotlabels = enc.transform(X).toarray()
onehotlabels = enc.transform(X).toarray()
onehotlabels.shape

# as you can see, you've the same number of rows 891
# but now you've so many more columns due to how we changed all the categorical data into numerical data


# ## Dummy Variables

# # Testing feature importance with Random Forest

# In[109]:


# colcoar -1 em todos os missing values
y = df['left']
X = df.loc[:, df.columns != 'left']


# In[110]:


X = pd.get_dummies(X)


# In[111]:


# training random forest to test feature importance
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=150, max_depth=8, min_samples_leaf=4, max_features=0.2, n_jobs=-1, random_state=0)
rf.fit(X, y)
features = X.columns.values
print("----- Training Done -----")


# In[112]:


def get_feature_importance_df(feature_importances, 
                              column_names, 
                              top_n=25):
    """Get feature importance data frame.
 
    Parameters
    ----------
    feature_importances : numpy ndarray
        Feature importances computed by an ensemble 
            model like random forest or boosting
    column_names : array-like
        Names of the columns in the same order as feature 
            importances
    top_n : integer
        Number of top features
 
    Returns
    -------
    df : a Pandas data frame
 
    """
     
    imp_dict = dict(zip(column_names, 
                        feature_importances))
    top_features = sorted(imp_dict, 
                          key=imp_dict.get, 
                          reverse=True)[0:top_n]
    top_importances = [imp_dict[feature] for feature 
                          in top_features]
    df = pd.DataFrame(data={'feature': top_features, 
                            'importance': top_importances})
    return df


# In[113]:


feature_importance = get_feature_importance_df(rf.feature_importances_, features)


# In[117]:


feature_importance[0:6]


# In[118]:


fig,ax = plt.subplots()
fig.set_size_inches(20,10)
sns.barplot(data=feature_importance[:10],x="feature",y="importance",ax=ax,color='#1f77b4',)
ax.set(xlabel='Variable name', ylabel='Importance',title="Variable importances");


# In[123]:


from sklearn.tree import export_graphviz


# In[124]:


export_graphviz(rf.estimators_[0],
                feature_names=X.columns,
                filled=True,
                rounded=True)


# In[125]:


import os
os.system('dot -Tpng tree.dot -o tree.png')


# In[143]:


from sklearn.tree import export_graphviz
import graphviz

export_graphviz(rf.estimators_[0], out_file="tree.dot")
with open("tree.dot") as f:
    dot_graph = f.read()
graphviz.Source(dot_graph)


# # Feature Selection

# In machine learning and statistics, feature selection, also known as variable selection, attribute selection or 
# variable subset selection, is the process of selecting a subset of relevant features (variables, predictors) for use 
# in model construction. Feature selection techniques are used for four reasons:
# - simplification of models to make them easier to interpret by researchers/users,
# - shorter training times,
# - to avoid the curse of dimensionality,
# - enhanced generalization by reducing overfitting[2] (formally, reduction of variance[1])
# The central premise when using a feature selection technique is that the data contains many features that are either 
# redundant or irrelevant, and can thus be removed without incurring much loss of information.
# Redundant or irrelevant features are two distinct notions, since one relevant feature may be redundant in the presence 
# of another relevant feature with which it is strongly correlated.

# ## Feature Scalling

# In[67]:


from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

skb = SelectKBest(chi2, k=4)
skb.fit(X, y)
X_new = skb.transform(X)

X_new.shape

skb.get_params()






# In[68]:


print(skb.get_params())


# In[69]:


feature_selection = skb.get_support(indices=True)


# In[70]:


feature_selection


# In[2]:


def indices():
    for i in feature_selection:
        yield X.columns[i]


# In[3]:


alfa = indices()


# In[4]:


fs_variables = list(alfa)


# In[74]:


for i, z in fs_variables, df.describe:
    print(i, z)


# In[1]:


for i, z in fs_variables, df.describe:
    print(i, z)


# Feature scaling is the method to limit the range of variables so that they can be compared on common grounds. It is performed on continuous variables. Lets plot the distribution of all the continuous variables in the data set.
# 
# The processo for feature scalling are:
#     - Rescale Data
#     - Standardize Data
#     - Normalize Data

# ## Rescale Data

# In[ ]:


# Rescale data (between 0 and 1)
import pandas
import scipy
import numpy
from sklearn.preprocessing import MinMaxScaler
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.data"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = pandas.read_csv(url, names=names)
array = dataframe.values
# separate array into input and output components
X = array[:,0:8]
Y = array[:,8]
scaler = MinMaxScaler(feature_range=(0, 1))
rescaledX = scaler.fit_transform(X)
# summarize transformed data
numpy.set_printoptions(precision=3)
print(rescaledX[0:5,:])


# ## Standardize Data

# In[ ]:


# Standardize data (0 mean, 1 stdev)
from sklearn.preprocessing import StandardScaler
import pandas
import numpy
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.data"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = pandas.read_csv(url, names=names)
array = dataframe.values
# separate array into input and output components
X = array[:,0:8]
Y = array[:,8]
scaler = StandardScaler().fit(X)
rescaledX = scaler.transform(X)
# summarize transformed data
numpy.set_printoptions(precision=3)
print(rescaledX[0:5,:])


# ## Normalize Data

# In[ ]:


# Normalize data (length of 1)
from sklearn.preprocessing import Normalizer
import pandas
import numpy
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.data"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = pandas.read_csv(url, names=names)
array = dataframe.values
# separate array into input and output components
X = array[:,0:8]
Y = array[:,8]
scaler = Normalizer().fit(X)
normalizedX = scaler.transform(X)
# summarize transformed data
numpy.set_printoptions(precision=3)
print(normalizedX[0:5,:])


# ## MinMax Scaler

# ## Binarize Data (Make Binary)

# Label Binarizer
# Binarize labels in a one-vs-all fashion
# Several regression and binary classification algorithms are available in the scikit. A simple way to extend these algorithms to the multi-class classification case is to use the so-called one-vs-all scheme.
# At learning time, this simply consists in learning one regressor or binary classifier per class. In doing so, one needs to convert multi-class labels to binary labels (belong or does not belong to the class). LabelBinarizer makes this process easy with the transform method.

# In[ ]:


# binarization
from sklearn.preprocessing import Binarizer
import pandas
import numpy
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.data"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = pandas.read_csv(url, names=names)
array = dataframe.values
# separate array into input and output components
X = array[:,0:8]
Y = array[:,8]
binarizer = Binarizer(threshold=0.0).fit(X)
binaryX = binarizer.transform(X)
# summarize transformed data
numpy.set_printoptions(precision=3)
print(binaryX[0:5,:])


# In[ ]:


def featurize(features):
  transformations = [
                            ('Embarked', LabelBinarizer()),
                            ('Fare', None),
                            ('Parch', None),
                            ('Pclass', LabelBinarizer()),
                            ('Sex', LabelBinarizer()),
                            ('SibSp', None),                                       
                            ('Title', LabelBinarizer()),
                            ('FamilySize', None),
                            ('FamilyID', LabelBinarizer()),
                            ('AgeOriginallyNaN', None),
                            ('AgeFilledMedianByTitle', None)]

  return DataFrameMapper(filter(lambda x: x[0] in df.columns, transformations))


# # Pipeline Modelling

# ## Classifiers

# ### Common Classifiers

# Select the Targets

# In[78]:


y = df['left']


# In[79]:


X = df.loc[:, df.columns != 'left']


# Handling Categorical with Dummies

# In[80]:


X = pd.get_dummies(X)


# In[83]:


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import itertools
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from mlxtend.classifier import EnsembleVoteClassifier
from mlxtend.data import iris_data
from mlxtend.plotting import plot_decision_regions

# Initializing Classifiers
clf1 = LogisticRegression(random_state=0)
clf2 = RandomForestClassifier(random_state=0)
clf3 = SVC(random_state=0, probability=True)
eclf = EnsembleVoteClassifier(clfs=[clf1, clf2, clf3],
                              weights=[2, 1, 1], voting='soft')


# In[85]:


clf1.fit(X, y)
clf2.fit(X, y)
clf3.fit(X, y)
eclf.fit(X, y)


# In[86]:


print(clf1.score(X, y))
print(clf2.score(X, y))
print(clf3.score(X, y))
print(eclf.score(X, y))


# Select the Features for Decision Boundaries Plot

# In[87]:


# select only 2 features in dataset for plot boundaries
X_array = X.iloc[:, 0:2]


# In[88]:


y_ = pd.DataFrame.as_matrix(y)
X_ = pd.DataFrame.as_matrix(X_array)
X_.shape
y_.shape


# In[89]:



# Plotting Decision Regions

gs = gridspec.GridSpec(2, 2)
fig = plt.figure(figsize=(10, 8))

labels = ['Logistic Regression',
          'Random Forest',
          'RBF kernel SVM',
          'Ensemble']

for clf, lab, grd in zip([clf1, clf2, clf3, eclf],
                         labels,
                         itertools.product([0, 1],
                         repeat=2)):
    clf.fit(X_, y_)
    ax = plt.subplot(gs[grd[0], grd[1]])
    fig = plot_decision_regions(X=X_, y=y_, 
                                clf=clf,
                               legend=2)
    plt.title(lab)

plt.show();


# ### Support Vector Machines

# In[11]:


from sklearn import svm
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.pipeline import Pipeline


# ANOVA SVM-C
anova_filter = SelectKBest(f_regression, k=5)
clf = svm.SVC(kernel='linear')
anova_svm = Pipeline([('anova', anova_filter), ('svc', clf)])
# You can set the parameters using the names issued
# For instance, fit using a k of 10 in the SelectKBest
# and a parameter 'C' of the svm
anova_svm.set_params(anova__k=10, svc__C=.1).fit(X, y)
                     



prediction = anova_svm.predict(X)
anova_svm.score(X, y)                        

# getting the selected features chosen by anova_filter
anova_svm.named_steps['anova'].get_support()


# In[108]:


from sklearn import svm
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler


# ANOVA SVM-C
anova_filter = SelectKBest(f_regression, k=4)
clf = svm.SVC(kernel='linear')
anova_svm = Pipeline([('anova', anova_filter), ('MinMax', MinMaxScaler()), ('svc', clf)])
# You can set the parameters using the names issued
# For instance, fit using a k of 10 in the SelectKBest
# and a parameter 'C' of the svm
anova_svm.set_params(anova__k=8, svc__C=.1).fit(X, y)
                     



prediction = anova_svm.predict(X)
anova_svm.score(X, y)                        

# getting the selected features chosen by anova_filter
anova_svm.named_steps['anova'].get_support()


# ### Gradient Boosting Modelling

# In[9]:


from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV
from sklearn import svm

parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10]}
svc = svm.SVC()

# ANOVA SVM-C
anova_filter = SelectKBest(f_regression, k=4)
clf = GradientBoostingClassifier(max_features=5)
grid = GridSearchCV(estimator=svc, param_grid=parameters)

anova_gbm = Pipeline([('anova', anova_filter), 
                      ('MinMax', MinMaxScaler()), 
                      ('gbm', clf),
                      ('grid_search', grid)
                     ])

# You can set the parameters using the names issued
# For instance, fit using a k of 10 in the SelectKBest
# and a parameter 'C' of the svm
anova_gbm.set_params(anova__k=8).fit(X, y)

prediction = anova_gbm.predict(X)
anova_gbm.score(X, y)                        

# getting the selected features chosen by anova_filter
anova_gbm.named_steps['anova'].get_support()


# #### Decision Boundary

# In[51]:


print(__doc__)

from itertools import product

import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier

# Loading some example data
# iris = datasets.load_iris()
# X = iris.data[:, [0, 2]]
# y = iris.target

# Training classifiers
clf1 = DecisionTreeClassifier(max_depth=4)
clf2 = KNeighborsClassifier(n_neighbors=7)
clf3 = SVC(kernel='rbf', probability=True)
eclf = VotingClassifier(estimators=[('dt', clf1), ('knn', clf2),
                                    ('svc', clf3)],
                        voting='soft', weights=[2, 1, 2])

clf1.fit(X, y)
clf2.fit(X, y)
clf3.fit(X, y)
eclf.fit(X, y)

# Plotting decision regions
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))

f, axarr = plt.subplots(2, 2, sharex='col', sharey='row', figsize=(10, 8))

for idx, clf, tt in zip(product([0, 1], [0, 1]),
                        [clf1, clf2, clf3, eclf],
                        ['Decision Tree (depth=4)', 'KNN (k=7)',
                         'Kernel SVM', 'Soft Voting']):

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    axarr[idx[0], idx[1]].contourf(xx, yy, Z, alpha=0.4)
    axarr[idx[0], idx[1]].scatter(X[:, 0], X[:, 1], c=y,
                                  s=20, edgecolor='k')
    axarr[idx[0], idx[1]].set_title(tt)

plt.show();


# ## Regression

# ## Clustering

# ## Recommendation

# ## Natural Language Processing (NLP)

# ## Scrapping

# ## Sentimental Analysis

# ## Chatbot

# ## Reinforcement Learning

# ## Deep Learning
