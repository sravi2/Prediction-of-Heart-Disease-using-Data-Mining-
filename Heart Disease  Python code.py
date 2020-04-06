# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 19:53:19 2019

@author: Administrator
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 11:34:31 2019

@author: Administrator
"""
#import tensorflow as tf
from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(1)
#from tensorflow import set_random_seed
#5set_random_seed(None)

import numpy as np
import pandas as pd
import seaborn as sns
from statistics import mean
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.metrics import confusion_matrix
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score
from mlxtend.feature_selection import SequentialFeatureSelector
from mlxtend.classifier import EnsembleVoteClassifier
from mlxtend.feature_selection import ColumnSelector
import matplotlib.pyplot as plt

#input cleve3 for Cleveland Data Set
#input statlog for Statlog Data set
file = pd.ExcelFile('cleve3.xlsx')
file.sheet_names
df= file.parse('cleve3')
from sklearn.naive_bayes import MultinomialNB
mnb=MultinomialNB()
#Neural
from keras import Sequential
from keras.layers import Dense
#decision
from sklearn.tree import DecisionTreeClassifier

#Logistic Regression
from sklearn.linear_model import LogisticRegression
#The logistic Regression model is with best features
lm=LogisticRegression(C= 0.1, penalty= 'l2', solver= 'liblinear')
#lm =LogisticRegression(multi_class='auto')
#lm=LogisticRegression()
#feature_selector = SequentialFeatureSelector(lm,k_features=13,forward=True,verbose=2,scoring='roc_auc',cv=10)

from sklearn import tree
#dt = tree.DecisionTreeClassifier(criterion= 'entropy', max_depth= 3, min_samples_leaf= 6)
#Decision Tree model with best features
dt = tree.DecisionTreeClassifier(criterion= 'gini', max_depth= 3, min_samples_leaf= 2)

#Random forest
from sklearn.ensemble import RandomForestClassifier
rf=RandomForestClassifier(criterion= 'entropy', max_depth= 3, min_samples_leaf= 4, n_estimators=100)
#rf=RandomForestClassifier()

#SVM
from sklearn.svm import SVC
sm = SVC(C=0.1, gamma=1, kernel='linear')

# independent variables 0, 1, 2, 3, 4, 5, 6 ,7, 8, 9, 10, 11, 12
#Naive Bayes inputs 1, 2, 5, 6, 8, 10, 11, 12 and transformed features 15, 16, 17, 18, 19
# When Naive Bayes results are required the pipeline, stacking and vote needs to be deactivated.

# Note based on the output of recurseive feature selection the best feature are fed into x
# For best logistic input x with 12, 11, 10, 9, 8, 7, 5, 3, 2, 1
# For best svm input x with 1, 12, 11, 2, 10, 9, 8, 7
# For best Random forest input x with 6, 1, 8, 10, 0, 2, 3, 4, 7, 9, 11, 12
# For best Decision Tree input x with 0, 2, 4, 11, 12
#For best Naive Bayes input x with 1, 2, 5, 6, 8, 10, 11, 12, 15, 16, 17, 18, 19

#Currently all features are inputed
x = df.iloc[:,[0,1,2,3,4,5,6,7,8,9,10,11,12]].values

#Input for Recursive Feature Elimination
x1 = df.iloc[:,[0,1,2,3,4,5,6,7,8,9,10,11,12]].values
# dependent variables 
# 0 - Presence of heart disease
# 1 - Absence of heart disease
y = df.iloc[:,14].values # 14 represents Target class
# y is 14 for cleve3 - cleveland data set
# y is 13 for multiclass for logistic Regression with cleve3 - cleveland data set.
# y is 13 for - statlog data set

# Needs to be activated
#Feature Selection Method
from sklearn.feature_selection import RFE
rfe=RFE(lm,10) # 8 represents it gives best 8 features of 13 features and lm represents Logistic Regression
#rfe=RFE(LogisticRegression(),1)
rfe.fit(x1,y)


#pca
# pca is only applied to Neural Networks
from sklearn.decomposition import PCA
pca = PCA().fit(x)
#Plotting the Cumulative Summation of the Explained Variance
plt.figure()
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Number of Components')
plt.ylabel('Variance (%)') #for each component
plt.title('Pulsar Dataset Explained Variance')
plt.show()

#Heatmap
corrmat = df.corr()
top_corr_features = corrmat.index
plt.figure(figsize=(20,20))
#plot heat map
g=sns.heatmap(df[top_corr_features].corr(),annot=True,cmap="RdYlGn")


from sklearn.ensemble import AdaBoostRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import VotingClassifier 
from sklearn.svm import SVC


# Stratified K fold cross Validation
from sklearn.model_selection import StratifiedKFold
kf = StratifiedKFold(n_splits=10,random_state=None)

from sklearn import tree

#Random forest
from sklearn.ensemble import RandomForestClassifier
#rf=RandomForestClassifier(n_estimators=10,criterion='entropy',random_state=None)

#naive bayes
from sklearn.naive_bayes import MultinomialNB
#mnb=MultinomialNB()

#decision tree
from sklearn import tree
#dt = tree.DecisionTreeClassifier(criterion= 'gini', max_depth=3, max_features= 5, min_samples_leaf= 8)

#Neural
from keras import Sequential
from keras import initializers
from keras.layers import Dense

#Bagging#Boosting 
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier


# Accuracy, Precision, f1 values stored here
acc_scores_lm=[]
acc_scores_sm =[]
acc_scores_dt =[]
acc_scores_rf=[]
acc_scores_mnb=[]
acc_scores_nb=[]
acc_scores_VT1=[]
acc_scores_VT2=[]
acc_scores_VT3=[]
acc_scores_VT4=[]
acc_scores_VT5=[]
acc_scores_VT6=[]
acc_scores_VT7=[]
acc_scores_VT8=[]
acc_scores_VT9=[]
acc_scores_VT10=[]
acc_scores_VT11=[]
acc_scores_VT12=[]
acc_scores_VT13=[]
acc_scores_VT14=[]
acc_scores_VT15=[]
acc_scores_VT16=[]
acc_scores_VT17=[]
acc_scores_VT18=[]
acc_scores_VT19=[]

acc_scores_NN=[]
#bagging#boosting
acc_score_bg_sm=[]
acc_score_bg_lm=[]
acc_score_bg_mnb=[]
acc_score_bg_dt=[]
acc_score_bg_rf=[]
acc_score_adb_lm=[]
acc_score_adb_sm=[]
acc_score_adb_dt=[]
acc_score_adb_rf=[]
acc_score_adb_mnb=[]
acc_scores_nb=[]

prec_scores_NN=[]
prec_scores_lm=[]
prec_scores_mnb=[]
prec_scores_dt=[]
prec_scores_rf=[]
prec_scores_sm=[]
prec_scores_VT1=[]
prec_scores_VT2=[]
prec_scores_VT3=[]
prec_scores_VT4=[]
prec_scores_VT5=[]
prec_scores_VT6=[]
f1_scores_NN=[]
f1_scores_lm=[]
f1_scores_mnb=[]
f1_scores_dt=[]
f1_scores_rf=[]
f1_scores_sm=[]
f1_scores_VT1=[]
f1_scores_VT2=[]
f1_scores_VT3=[]
f1_scores_VT4=[]
f1_scores_VT5=[]
f1_scores_VT6=[]

cm_scores_lm=[]
cm_scores_VT1=[]
cm_scores_NN=[]

model_acc_sc=[]
prec_scores_model=[]
f1_scores_model=[]
#Pipeline
acc_scores_lmpip=[]
prec_scores_lmpip=[]
f1_scores_lmpip=[]

# Pipe Line with best parameters for Logistic, Random Forest and Support Vector Machine
# Note Pipeline to be activated only with stacking or voting one at a time.
#pipe1 = make_pipeline(ColumnSelector(cols=(12,11,10,9,8,7,5,3,2,1)),LogisticRegression(C= 0.1, penalty= 'l2', solver= 'liblinear'))
#pipe2 = make_pipeline(ColumnSelector(cols=(6,1,8,10,0,2,3,4,7,9,11,12)),RandomForestClassifier(criterion= 'entropy', max_depth= 3, min_samples_leaf= 4, n_estimators=100 ))
#pipe3 = make_pipeline(ColumnSelector(cols=(1,12,11,2,10,9,8,7)),SVC(C=0.1, gamma=1, kernel='linear'))
'''
#Grid Search for 4 classifier
# Need to be activated for each classifier after feature selection values set to x

#Logistic
grid={"C":np.logspace(-3,3,7), "penalty":["l2", 'l1'],"solver":["liblinear"]}# l1 lasso l2 ridge
logreg=LogisticRegression(multi_class='auto')
logreg_cv=GridSearchCV(logreg,grid,cv=kf)
logreg_cv.fit(x,y)
print("tuned hpyerparameters :(best parameters) ",logreg_cv.best_params_)
print("accuracy :",logreg_cv.best_score_)

#SVM
# defining parameter range 
param_grid = {'C': [0.1, 1, 10, 100, 1000],  'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 'kernel': [ 'linear']}  
grid = GridSearchCV(SVC(), param_grid, refit = True, verbose = 3,cv=2) 
# fitting the model for grid search 
grid.fit(x, y)
print(grid.best_params_)
print(grid.best_estimator_)

#Random forest
param_grid = {"max_depth": [3, None],
              "n_estimators": [100],
              "min_samples_leaf": [1, 2,3,4,5,6,7,8,9],
              "criterion": ["gini", "entropy"]}




grid_search = GridSearchCV(RandomForestClassifier(), param_grid = param_grid, 
                          cv = kf, n_jobs = -1, verbose = 2)
grid_search.fit(x, y)
print(grid_search.best_params_)

#Decision Tree
param_grid = {"max_depth": [3, None],
              
              "min_samples_leaf": [1,2,3,4,5,6,7,8,9],
              "criterion": ["gini", "entropy"]}
grid_search = GridSearchCV(estimator = DecisionTreeClassifier(), param_grid = param_grid, 
                          cv = kf, n_jobs = -1, verbose = 2)
grid_search.fit(x, y)
print(grid_search.best_params_)
'''
# Note Multinominial Naive Bayes should be run seperately from other classifiers.
# Note Neural Network to be run seperately with activating pca. The other classifiers needs to be deactivated.
# Note Sensitivity and specificity needs to be activated for each model. For each model we need to average 10 time cross-validation  results of sensitivity and specificity to get mean of it
# Loop for splitting data into training and testing data


for train_index, test_index in kf.split(x,y):
    #print("Train:", train_index, "Validation:", test_index)
    x_train, x_test = x[train_index], x[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    # Standardization of data
    sc=StandardScaler(0,1)
    X_train = sc.fit_transform(x_train)
    X_test = sc.transform(x_test)
    
    
    # Note needs to be activated only with Neural Network
    #pca = PCA(n_components = 10) 
    #X_train = pca.fit_transform(X_train) 
    #X_test = pca.transform(X_test) 
    #explained_variance = pca.explained_variance_ratio_ 
    '''
    #Vote classifier
    #pipeline
    eclf = EnsembleVoteClassifier(clfs=[pipe3,pipe2])
    eclf.fit(X_train, y_train)
    y_pred_lmpip=eclf.predict(X_test)
    acc_scores_lmpip.append(accuracy_score(y_test, y_pred_lmpip))
    prec_scores_lmpip.append(precision_score(y_test, y_pred_lmpip))
    f1_scores_lmpip.append(f1_score(y_test, y_pred_lmpip))
    
    sensitivity= confusion_matrix(y_test,y_pred_lmpip )[0,0]/(confusion_matrix(y_test,y_pred_lmpip )[0,0]+confusion_matrix(y_test,y_pred_lmpip )[0,1])
    #print("Sensitivity of Vote ",sensitivity)
    

    specificity1 = confusion_matrix(y_test,y_pred_lmpip )[1,1]/(confusion_matrix(y_test,y_pred_lmpip )[1,0]+confusion_matrix(y_test,y_pred_lmpip )[1,1])
    #spec= print("Specificity of Vote",specificity1)
    '''
    
    #logistic
    lm.fit(X_train, y_train)
    y_pred_lm=lm.predict(X_test)
    acc_scores_lm.append(accuracy_score(y_test, y_pred_lm,))
    prec_scores_lm.append(precision_score(y_test, y_pred_lm))
    f1_scores_lm.append(f1_score(y_test, y_pred_lm))
    #cm_scores_lm.append(confusion_matrix(y_test,y_pred_lm ))
    #cm_scores_lm=print(confusion_matrix(y_test,y_pred_lm ))
    #####from confusion matrix calculate accuracy
    sensitivity= confusion_matrix(y_test,y_pred_lm )[0,0]/(confusion_matrix(y_test,y_pred_lm )[0,0]+confusion_matrix(y_test,y_pred_lm )[0,1])
    print("Sensitivity of Logistic Regression",sensitivity)
    specificity1 = confusion_matrix(y_test,y_pred_lm )[1,1]/(confusion_matrix(y_test,y_pred_lm )[1,0]+confusion_matrix(y_test,y_pred_lm )[1,1])
    spec= print( "Specificity of Logistic",specificity1)
    
    
    #svm
    sm.fit(X_train, y_train)
    y_pred_sm=sm.predict(X_test)
    acc_scores_sm.append(accuracy_score(y_test, y_pred_sm))
    prec_scores_sm.append(precision_score(y_test, y_pred_sm))
    f1_scores_sm.append(f1_score(y_test, y_pred_sm))
    #cm_scores_lm=print(confusion_matrix(y_test,y_pred_lm ))
    #####from confusion matrix calculate accuracy
    sensitivity= confusion_matrix(y_test,y_pred_sm )[0,0]/(confusion_matrix(y_test,y_pred_sm )[0,0]+confusion_matrix(y_test,y_pred_sm )[0,1])
    #print("Sensitivity of svm",sensitivity)
    specificity1 = confusion_matrix(y_test,y_pred_sm )[1,1]/(confusion_matrix(y_test,y_pred_sm )[1,0]+confusion_matrix(y_test,y_pred_sm )[1,1])
    #spec= print( "Specificity of svm",specificity1)
    
    #Naive
    #mnb.fit(x_train, y_train)
    #y_pred_mnb=mnb.predict(x_test)
    #acc_scores_mnb.append(accuracy_score(y_test, y_pred_mnb))
    #prec_scores_mnb.append(precision_score(y_test, y_pred_mnb))
    #f1_scores_mnb.append(f1_score(y_test, y_pred_mnb))
    #sensitivity= confusion_matrix(y_test,y_pred_mnb )[0,0]/(confusion_matrix(y_test,y_pred_mnb )[0,0]+confusion_matrix(y_test,y_pred_mnb )[0,1])
    #print("Sensitivity of Naive",sensitivity)
    #specificity1 = confusion_matrix(y_test,y_pred_mnb )[1,1]/(confusion_matrix(y_test,y_pred_mnb )[1,0]+confusion_matrix(y_test,y_pred_mnb )[1,1])
    #spec= print( "Specificity of Naive",specificity1)
   
    
    
    #dt
    dt.fit(X_train, y_train)
    y_pred_dt=dt.predict(X_test)
    acc_scores_dt.append(accuracy_score(y_test, y_pred_dt))
    prec_scores_dt.append(precision_score(y_test, y_pred_dt))
    f1_scores_dt.append(f1_score(y_test, y_pred_dt))
    sensitivity= confusion_matrix(y_test,y_pred_dt )[0,0]/(confusion_matrix(y_test,y_pred_dt )[0,0]+confusion_matrix(y_test,y_pred_dt )[0,1])
    #print("Sensitivity of Decision Tree",sensitivity)

    specificity1 = confusion_matrix(y_test,y_pred_dt )[1,1]/(confusion_matrix(y_test,y_pred_dt )[1,0]+confusion_matrix(y_test,y_pred_dt )[1,1])
    #spec= print("Specificity of Decision Tree", specificity1)
    
    #Random forest
    rf.fit(X_train, y_train)
    y_pred_rf=rf.predict(X_test)
    acc_scores_rf.append(accuracy_score(y_test, y_pred_rf))
    prec_scores_rf.append(precision_score(y_test, y_pred_rf))
    f1_scores_rf.append(f1_score(y_test, y_pred_rf))
    sensitivity= confusion_matrix(y_test,y_pred_rf )[0,0]/(confusion_matrix(y_test,y_pred_rf )[0,0]+confusion_matrix(y_test,y_pred_rf )[0,1])
    #print("Sensitivity of Random forest",sensitivity)
    specificity1 = confusion_matrix(y_test,y_pred_rf )[1,1]/(confusion_matrix(y_test,y_pred_rf )[1,0]+confusion_matrix(y_test,y_pred_rf )[1,1])
    #spec= print( "Specificity of Random forest",specificity1)
    
    #Stacking 
    #pipe 1 = Best Features for Logistic Regression
    #Pipe 2 = Best Features for Random Forest
    #Pipe 3 = Best Features for SVM
    # Note input values of models that is pipe1, pipe2 and pipe 3  needs to be changed every time for results. These are base classifier
    #Note input values of model that is lm, sm, rf need to be applied as it is metaclassifier
    # Note x should have all the input values from 0 to 12.
    '''
    from vecstack import stacking
    models = [pipe1,pipe2,pipe3]
    s_train, s_test= stacking(models,X_train, y_train,X_test, regression=False,metric=accuracy_score, random_state=0, shuffle=False)
    #model = SVC()
    #model = XGBClassifier(random_state=0, n_jobs=-1, learning_rate=1, n_estimators=100, max_depth=3)
    model = lm
    model=model.fit(s_train, y_train)
    y_pred_model=model.predict(s_test)
    model_acc_sc.append(accuracy_score(y_test,y_pred_model))
    prec_scores_model.append(precision_score(y_test, y_pred_model))
    f1_scores_model.append(f1_score(y_test, y_pred_model))
    sensitivity= confusion_matrix(y_test,y_pred_model )[0,0]/(confusion_matrix(y_test,y_pred_model )[0,0]+confusion_matrix(y_test,y_pred_model )[0,1])
    #print("Sensitivity of stacking",sensitivity)
    specificity1 = confusion_matrix(y_test,y_pred_model )[1,1]/(confusion_matrix(y_test,y_pred_model )[1,0]+confusion_matrix(y_test,y_pred_model )[1,1])
    #spec= print("Sepecificity of stacking", specificity1)
    
    #Neural network
    #Note pca to be activated 
    #Note input_dim =10 in the layers based on pca results
    classifier = Sequential()
    #First Hidden layer
    classifier.add(Dense(5, activation='relu', kernel_initializer= initializers.RandomNormal(mean=0,stddev=0.05,seed=None), bias_initializer=initializers.RandomNormal(mean=0,stddev=0.05,seed=None ), input_dim=10))
    #Second hidden layer
    classifier.add(Dense(5,activation='relu', kernel_initializer=initializers.RandomNormal(mean=0,stddev=0.05,seed=None ),bias_initializer=initializers.RandomNormal(mean=0,stddev=0.05,seed=None), input_dim=10))
    #output Layer
    classifier.add(Dense(1, activation='sigmoid', kernel_initializer=initializers.RandomNormal(mean=0,stddev=0.05,seed=None ), bias_initializer=initializers.RandomNormal(mean=0,stddev=0.05,seed=None)))
    #Compiling neural network
    classifier.compile(optimizer= 'adam', loss=['binary_crossentropy'], metrics=['accuracy'])
    classifier.fit(X_train, y_train, batch_size=5, epochs=50, shuffle=False) 
    eval_model=classifier.evaluate(X_train, y_train)
    y_pred_NN=classifier.predict(X_test)
    y_pred_NN=(y_pred_NN>0.5)
    acc_scores_NN.append(accuracy_score(y_test, y_pred_NN))
    prec_scores_NN.append(precision_score(y_test, y_pred_NN))
    f1_scores_NN.append(f1_score(y_test, y_pred_NN))
    sensitivity= confusion_matrix(y_test, y_pred_NN )[0,0]/(confusion_matrix(y_test, y_pred_NN )[0,0]+confusion_matrix(y_test, y_pred_NN )[0,1])
    #print("Sensitivity of Neural Network",sensitivity)

    specificity1 = confusion_matrix(y_test, y_pred_NN )[1,1]/(confusion_matrix(y_test, y_pred_NN )[1,0]+confusion_matrix(y_test, y_pred_NN)[1,1])
    #spec= print( "Specificity of Neural Network",specificity1)
    
    #Bagging
    bg_sm=BaggingClassifier(SVC(C=0.1, gamma=1, kernel='linear'), max_samples =0.85, max_features = 1.0, n_estimators = 20)
    bg_sm.fit(x_train,y_train)
    y_pred_bg_sm=bg_sm.predict(x_test)
    acc_score_bg_sm.append(accuracy_score(y_test, y_pred_bg_sm))
    
    bg_dt=BaggingClassifier(DecisionTreeClassifier(criterion= 'gini', max_depth= 3, min_samples_leaf= 2), max_samples =0.85, max_features = 1.0, n_estimators = 20)
    bg_dt.fit(x_train,y_train)
    y_pred_bg_dt=bg_dt.predict(x_test)
    acc_score_bg_dt.append(accuracy_score(y_test, y_pred_bg_dt))
    
    bg_rf=BaggingClassifier(RandomForestClassifier(criterion= 'entropy', max_depth= 3, min_samples_leaf= 4, n_estimators=100), max_samples =0.85, max_features = 1.0, n_estimators = 20)
    bg_rf.fit(x_train,y_train)
    y_pred_bg_rf=bg_rf.predict(x_test)
    acc_score_bg_rf.append(accuracy_score(y_test, y_pred_bg_rf))
    
    bg_lm=BaggingClassifier(LogisticRegression(C= 0.1, penalty= 'l2', solver= 'liblinear'), max_samples =0.85, max_features = 1.0, n_estimators = 20)
    bg_lm.fit(X_train,y_train)
    y_pred_bg_lm=bg_lm.predict(X_test)
    acc_score_bg_lm.append(accuracy_score(y_test, y_pred_bg_lm))
    
    #bg_mnb=BaggingClassifier(MultinomialNB(), max_samples =0.85, max_features = 1.0, n_estimators = 20)
    #bg_mnb.fit(x_train,y_train)
    #y_pred_bg_mnb=bg_mnb.predict(x_test)
    #acc_score_bg_mnb.append(accuracy_score(y_test, y_pred_bg_mnb))
    
    
    #acc_score_bg_sm.append(accuracy_score(y_test, y_pred_bg_sm))
    #bg_dt=BaggingClassifier(DecisionTreeClassifier(), max_samples =0.85, max_features = 1.0, n_estimators = 20)
    #bg_dt.fit(X_train,y_train)
    #bg_lm=BaggingClassifier(LogisticRegression(), max_samples =0.85, max_features = 1.0, n_estimators = 30)
    #bg_lm.fit(X_train,y_train)
    #y_pred_bg_lm=lm.predict(X_test)
    #acc_score_bg_lm.append(accuracy_score(y_test, y_pred_bg_lm))
    
    #Boosting
   
    
    adb_sm = AdaBoostClassifier(SVC(C=0.1, gamma=1, kernel='linear'), n_estimators = 10, learning_rate = 1,algorithm='SAMME')
    adb_sm.fit(X_train,y_train) 
    y_pred_adb_sm=adb_sm.predict(X_test)
    acc_score_adb_sm.append(accuracy_score(y_test, y_pred_adb_sm))
    
    #y_pred_adb_sm=sm.predict(X_test)
    #acc_score_adb_sm.append(accuracy_score(y_test, y_pred_adb_sm))
    
    adb_dt = AdaBoostClassifier(DecisionTreeClassifier(criterion= 'gini', max_depth= 3, min_samples_leaf= 2), n_estimators = 50, learning_rate = 1)
    adb_dt.fit(X_train,y_train)
    y_pred_adb_dt=adb_dt.predict(X_test)
    acc_score_adb_dt.append(accuracy_score(y_test, y_pred_adb_dt))
    
    adb_rf = AdaBoostClassifier(RandomForestClassifier(criterion= 'entropy', max_depth= 3, min_samples_leaf= 4, n_estimators=100), n_estimators = 50, learning_rate = 1,algorithm='SAMME.R')
    adb_rf.fit(X_train,y_train)
    y_pred_adb_rf=adb_rf.predict(X_test)
    acc_score_adb_rf.append(accuracy_score(y_test, y_pred_adb_rf))
    
    
    adb_lm = AdaBoostClassifier(LogisticRegression(C= 0.1, penalty= 'l2', solver= 'liblinear'), n_estimators = 50, learning_rate = 1)
    adb_lm_model=adb_lm.fit(X_train,y_train)
    y_pred_adb_lm=adb_lm_model.predict(X_test)
    acc_score_adb_lm.append(accuracy_score(y_test, y_pred_adb_lm))
    
    adb_mnb = AdaBoostClassifier(MultinomialNB(), n_estimators = 10, learning_rate = 1)
    adb_mnb_model=adb_mnb.fit(x_train,y_train)
    y_pred_adb_mnb=adb_mnb_model.predict(x_test)
    acc_score_adb_mnb.append(accuracy_score(y_test, y_pred_adb_mnb))
    '''
# Print the results
    
print("logisticRegression accuracy acc",mean(acc_scores_lm))
print("Random forest acc",mean(acc_scores_rf))
print("SVM accuracy",mean(acc_scores_sm))
#print("Naive Bayes:",mean(acc_scores_mnb))
print("DecisionTree acc",mean(acc_scores_dt))
#print("Neural Network accuracy ",mean(acc_scores_NN))

print("logisticRegression precision ",mean(prec_scores_lm))
#print("Naive bayes",mean(prec_scores_mnb))
print("SVM precision",mean(prec_scores_sm))
print("DecisionTree precision",mean(prec_scores_dt))
print("Random forest precision",mean(prec_scores_rf))
#print("Neural Network precision",mean(prec_scores_NN))

print("logisticRegression f1",mean(f1_scores_lm))
#print("Naive bayes",mean(f1_scores_mnb))
print("SVM f1",mean(f1_scores_sm))
print("DecisionTree f1",mean(f1_scores_dt))
print("Random forest f1",mean(f1_scores_rf))
#print("Neural Network f1",mean(f1_scores_NN))

#Bagging#Boosting#Accuracy
'''
#Vote results
#print(mean(acc_score_bg_lm))
print("accuracy logistic Vote",mean(acc_scores_lmpip))
print("prec logistic Vote",mean(prec_scores_lmpip))
print("f1 logistic Vote",mean(f1_scores_lmpip))
#Boosting results
print("Adb accuracy logistic ",mean(acc_score_adb_lm))
print("Adb accuracy SVM ",mean(acc_score_adb_sm))
print("Adb accuracy rf ",mean(acc_score_adb_rf))
print("Adb accuracy dt ",mean(acc_score_adb_dt))
#print("Adb accuracy nb ",mean(acc_score_adb_mnb))
#bagging results
print("b accuracy logistic ",mean(acc_score_bg_lm))
print("b accuracy SVM ",mean(acc_score_bg_sm))
print("b accuracy rf ",mean(acc_score_bg_rf))
print("b accuracy dt ",mean(acc_score_bg_dt))
#print("b accuracy nb ",mean(acc_score_bg_mnb))
#print("b accuracy dt ",mean(cm_scores_lm))
#print(mean(acc_score_bg_sm))
#print(mean(acc_score_adb_sm))

# Stacking results
print('Stack  acc',mean(model_acc_sc))
print('Stack  pre',mean(f1_scores_model))
print('Stack  f1',mean(prec_scores_model))
'''
# Print Feature selection results

print(rfe.support_)
print(rfe.ranking_)
#print(features.k_feature_idx_)
plt.show()
