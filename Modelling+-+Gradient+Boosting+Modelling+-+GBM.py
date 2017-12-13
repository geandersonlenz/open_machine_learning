
# coding: utf-8

# # Import Libraries

# In[ ]:


from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV
from sklearn import svm
from sklearn.model_selection import train_test_split


# # Define a function for modeling and cross-validation
# 
# This function will do the following:
# 1. fit the model
# 2. determine training accuracy
# 3. determine training AUC
# 4. determine testing AUC
# 5. perform CV is performCV is True
# 6. plot Feature Importance if printFeatureImportance is True

# In[ ]:


def modelfit(alg, dtrain, dtest, predictors, performCV=True, printFeatureImportance=True, cv_folds=5):
    #Fit the algorithm on the data
    alg.fit(dtrain[predictors], dtrain['Disbursed'])
        
    #Predict training set:
    dtrain_predictions = alg.predict(dtrain[predictors])
    dtrain_predprob = alg.predict_proba(dtrain[predictors])[:,1]
    
    #Perform cross-validation:
    if performCV:
        cv_score = cross_validation.cross_val_score(alg, dtrain[predictors], dtrain['Disbursed'], cv=cv_folds, scoring='roc_auc')
    
    #Print model report:
    print "\nModel Report"
    print "Accuracy : %.4g" % metrics.accuracy_score(dtrain['Disbursed'].values, dtrain_predictions)
    print "AUC Score (Train): %f" % metrics.roc_auc_score(dtrain['Disbursed'], dtrain_predprob)
    
    if performCV:
        print "CV Score : Mean - %.7g | Std - %.7g | Min - %.7g | Max - %.7g" % (np.mean(cv_score),np.std(cv_score),np.min(cv_score),np.max(cv_score))
                
    #Print Feature Importance:
    if printFeatureImportance:
        feat_imp = pd.Series(alg.feature_importances_, predictors).sort_values(ascending=False)
        feat_imp.plot(kind='bar', title='Feature Importances')
        plt.ylabel('Feature Importance Score')


# # GBM Modelling
# 
# There 2 types of parameters here:
# 1. Tree-specific parameters
#   * min_samples_split
#   * min_samples_leaf
#   * max_depth
#   * min_leaf_nodes
#   * max_features
#   * loss function
# 2. Boosting specific paramters
#   * n_estimators
#   * learning_rate
#   * subsample

# ## Approach for tackling the problem
# 
# 1. Decide a relatively higher value for learning rate and tune the number of estimators requried for that.
# 2. Tune the tree specific parameters for that learning rate
# 3. Tune subsample
# 4. Lower learning rate as much as possible computationally and increase the number of estimators accordingly.

# ## Step 1- Find the number of estimators for a high learning rate
# 
# We will use the following benchmarks for parameters:
# 1. min_samples_split = 500 : ~0.5-1% of total values. Since this is imbalanced class problem, we'll take small value
# 2. min_samples_leaf = 50 : Just using for preventing overfitting. will be tuned later.
# 3. max_depth = 8 : since high number of observations and predictors, choose relatively high value
# 4. max_features = 'sqrt' : general thumbrule to start with
# 5. subsample = 0.8 : typically used value (will be tuned later)
# 
# 0.1 is assumed to be a good learning rate to start with. Let's try to find the optimum number of estimators requried for this.

# So we got 60 as the optimal estimators for the 0.1 learning rate. Note that 60 is a reasonable value and can be used as it is. But it might not be the same in all cases. Other situations:
# 1. If the value is around 20, you might want to try lowering the learning rate to 0.05 and re-run grid search
# 2. If the values are too high ~100, tuning the other parameters will take long time and you can try a higher learning rate
# 
# ## Step 2- Tune tree-specific parameters
# Now, lets move onto tuning the tree parameters. We will do this in 2 stages:
# 1. Tune max_depth and num_samples_split
# 2. Tune min_samples_leaf
# 3. Tune max_features

# ## Step3- Tune Subsample and Lower Learning Rate
# 

# In[ ]:


gb_grid_params = {'learning_rate': [0.1, 0.05, 0.02, 0.01],
              'max_depth': [2, 4, 6, 8],
              'min_samples_leaf': [20, 50,100,150],
              #'max_features': [1.0, 0.3, 0.1] 
              }
print(gb_grid_params)

gb_gs = GradientBoostingClassifier(n_estimators = 600)

clf = GridSearchCV(gb_gs,
                   gb_grid_params,
                   cv=3,
                   scoring='roc_auc',
                   verbose = 3, 
                   n_jobs=10)

anova_filter = SelectKBest(f_regression, k=5)

anova_gbm = Pipeline([('anova', anova_filter), 
                      ('MinMax', MinMaxScaler()), 
                      ('gbm', clf)
                     ])

# You can set the parameters using the names issued
# For instance, fit using a k of 10 in the SelectKBest
# and a parameter 'C' of the svm
anova_gbm.fit(X_train, y_train)




# In[ ]:


scores = anova_gbm.score(X, y)
"Accuracy: %0.5f (+/- %0.5f)"%(scores.mean(), scores.std())b


# In[ ]:


prediction = anova_gbm.predict(X_test)

