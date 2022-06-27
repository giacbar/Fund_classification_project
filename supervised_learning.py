# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 09:27:41 2019

@author: python
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 14:24:11 2019

@author: fida-stage1
"""

'''This part requires the dataframe built in the data management script'''

import os
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import LabelEncoder

#os.chdir('\\\\salaanalisisrv\\ufficio_studi\\python\\Cluster\\Giulio\\def') #fonte libreria
import supervised_learning_def as supervised
import unsupervised_learning_def as unsupervised
#os.chdir('\\\\salaanalisisrv\\ufficio_studi\\python\\Cluster\\Giulio') #per caricamento dati


import warnings
warnings.filterwarnings("ignore")

'''0 - Set up the data for classification'''
'''0.1 - Define label (real life class) to use for classification: superclasse,classe or categoria and the size of the training set'''

label='cat_descrizione'
train_size=0.9
df=df_mixed 
variables=variables_mixed #features list
#variables=variables_pol
#variables=variables_mixed

scoring='accuracy'
#scoring='balanced_accuracy'
#scoring='roc_auc_ovr'
#scoring='roc_auc_ovo'
#scoring='roc_auc_ovo_weighted'
#scoring='roc_auc_ovr_weighted'


'''0.2 - Train and test split'''
X_train_return,y_train,X_test_return,y_test=supervised.prepare_inputs(df,anagrafica,variables_return,label,train_size,'stratified',scaling=False,artificial=False)
X_train_pol,y_train,X_test_pol,y_test=supervised.prepare_inputs(df,anagrafica,variables_pol,label,train_size,'stratified',scaling=False,artificial=False)


'''0.3 - Imputation methods comparison'''
classifier= supervised.LogisticRegression(C=1000)
supervised.simple_imputation(X_train_return,y_train,label,variables_return,anagrafica,classifier,'mean',scoring)
supervised.simple_imputation(X_train_return,y_train,label,variables_return,anagrafica,classifier,'median',scoring)
#supervised.imputation_scores(X_train_return,y_train,label,classifier,scoring)

method='kneighbors'

X_train_return=supervised.impute_missing_values(X_train_return,y_train,variables,label,method)  
X_test_return=supervised.impute_missing_values(X_test_return,y_test,variables,label,method)  

X_train_return=pd.DataFrame(X_train_return)
X_test_return=pd.DataFrame(X_test_return)

X_train=X_train_return.join(X_train_pol)
X_test=X_test_return.join(X_test_pol)

X_train=np.array(X_train)
X_test=np.array(X_test)

'''Class pie chart'''

supervised.pie_plot_classi(y_test)




'''0.4 - Verify the number of classes in train and test sets'''

print('numero categorie nel train set:{0}'.format(y_train[label].nunique()))
print('numero categorie nel test set:{0}'.format(y_test[label].nunique()))




'''0.5 - Random guess classifier: assign to each class probability equal to its frequency, for benchmarking purposes'''

#supervised.random_guess_classifier(X_train,y_train[label],X_test,y_test[label],'stratified')
supervised.random_guess_classifier_theoretical(y_train[label],'train') 
supervised.random_guess_classifier_theoretical(y_test[label],'test')

'''0.6 - PCA scatter plot of funds'''
supervised.train_PCA_plot(X_train,y_train[label],variables,'classe')


##############################################################


''' Finding optimal parameters using gridsearch
   1. Select dictionary of the parameters to use in the model
   2. Fit the model for all combinations of parameters, building a grid for categorical parameters, bayesian optimization for numerical parameters
   3. Pick the parameters that returned the best results concerning cross-validation on the training set
   4. Use the parameters for the test set
'''

#############################################################


'''1 - Logistic regression'''

'''1.1 - no penalty'''

'''Define classifier type, its parameters to optimize and the oversampler type (optional)'''

oversampler = supervised.select_oversampler('random',neighbors=2)
classifier= LogisticRegression(random_state=1)

#parameters={'classifier__solver':('newton-cg','lbfgs','sag',),'classifier__C':(10,100,1000,5000),'classifier__penalty':['l2']}
parameters={'classifier__solver':('lbfgs', 'liblinear', 'sag', 'saga'),'classifier__C':(10,100,200,500),'classifier__penalty':['l1','l2']}

                
logireg_test,best_estimator,output=supervised.crossvalidation_oversampling_test(classifier,parameters,X_train,y_train,X_test,y_test,label,False,oversampler,scoring)

'''validation curves'''
classifier= LogisticRegression( class_weight=None, dual=False, fit_intercept=True,
                   intercept_scaling=1, l1_ratio=None, max_iter=100,
                   multi_class='auto', n_jobs=None, penalty='l2',
                   random_state=1, solver='liblinear', tol=0.0001, verbose=0,
                   warm_start=False)
classifier_name='logistic regression'
param_name='C'
param_range =[1,50,100,150,200,500,1000]
supervised.plot_validation_curve(X_train,y_train[label],classifier,classifier_name,param_range,param_name,scoring)

'''###########cat_descrizione, x=5##### 

#############returns

###################pol


###################pol and return

'''############

'''1.3 - misurazione dei risultati'''
#true_label='classe'
#predicted_label='classe_pred'
#logireg_report,logireg_misclassification=supervised.output_report(output,true_label,predicted_label)


##############################################################################



'''3 - Nearest neighbors'''
oversampler = supervised.RandomOverSampler(random_state=1)
classifier =  KNeighborsClassifier()
parameters={'classifier__n_neighbors':(2,3,4,5,6),'classifier__p':(1,2),'classifier__leaf_size':(2,3,4,5),
            'classifier__weights':('uniform','distance'),
            'classifier__algorithm':('ball_tree','kd_tree')}

knn_test,knn_best_estimator,output=supervised.crossvalidation_oversampling_test(classifier,parameters,X_train, y_train, X_test, y_test,label,False,oversampler,scoring)





'''###########cat_descrizione, x=5##### 

#############returns

#############pol
#############returns+pol

'''
#true_label='classe'
#predicted_label='classe_pred'
#knn_report,knn_misclassification=supervised.output_report(output,true_label,predicted_label)

##############################################################################


'''6 - Support Vector Machines'''

'''6.1 - Part 1: linear and polynomial kernel'''

oversampler = supervised.RandomOverSampler(random_state=1)
#oversampler = supervised.select_oversampler('smote',neighbors=5)

classifier =  SVC(random_state=1,probability=True)
parameters={'classifier__kernel':('linear','poly'),
            'classifier__C':(0.1,0.5,1,10,100,500,1000),
            'classifier__gamma':(1,2,3),'classifier__degree':(1,2,3)}


svm_test,svm_best_estimator,svm_output=supervised.crossvalidation_oversampling_test(classifier,parameters,X_train, y_train, X_test, y_test,label,False,oversampler,scoring)
#svm_report,svm_misclassification=supervised.output_report(svm_output)


'''6.2 - Part 2: nonlinear kernel'''

#classifier =  SVC(random_state=1,probability=True)
#parameters={'classifier__kernel':('rbf','sigmoid'),
            #'classifier__C':(1,10,100,1000,5000),
            #'classifier__gamma':(1,2,3,4)}



#svm_test,svm_best_estimator,svm_output=supervised.crossvalidation_oversampling_test(classifier,parameters,X_train, y_train, X_test, y_test,label,False,oversampler,scoring)
#svm_report,svm_misclassification=supervised.output_report(svm_output)

'''validation curves'''
classifier=  SVC()
classifier_name='SVM'
param_name='C'
param_range =(0.5,1,100,1000)
supervised.plot_validation_curve(X_train,y_train[label],classifier,classifier_name,param_range,param_name,scoring)




'''###########cat_descrizione, x=5##### 
#############returns

###################pol
###################pol e return


'''



#############################################################################

'''7 - Ensemble methods'''

'''7.1 - Random forests'''
#oversampler = supervised.select_oversampler('borderlinesmote1',neighbors=2)
oversampler = supervised.RandomOverSampler(random_state=1)
classifier =  RandomForestClassifier(random_state=1)
parameters={'classifier__criterion':('gini','entropy')}

forest_test,forest_best_estimator,forest_output=supervised.crossvalidation_oversampling_test(classifier,parameters,X_train, y_train, X_test, y_test,label,False,oversampler,scoring)

classifier = forest_best_estimator

parameters={'classifier__n_estimators':[300],
            'classifier__min_samples_leaf':(1,2,3,4,5,10),
            'classifier__min_samples_split':(1,2,3,4,5,10),
            'classifier__max_depth':(10,30,40,50,100,None),
            'classifier__bootstrap':(True,False)}


foresttest,bestestimator,output=supervised.crossvalidation_oversampling_test(classifier,parameters,X_train, y_train, X_test, y_test,label,False,oversampler,scoring)
'''###########cat_descrizione, x=5##### 
#############returns

###################pol
###################pol e return
'''
#true_label='classe'
#predicted_label='classe_pred'
#forest_report,forest_misclassification=supervised.output_report(output,true_label,predicted_label)



'''validation curves'''
classifier =  RandomForestClassifier(random_state=1,n_estimators=1000)
classifier_name='Random forest'
param_name='n_estimators'
param_range =[5,10,100,200,500,1000,2000]
supervised.plot_validation_curve(X_train,y_train[label],classifier,classifier_name,param_range,param_name,scoring)



'''7.3 - Gradient boosting'''

oversampler = supervised.RandomOverSampler(random_state=1)
classifier=GradientBoostingClassifier(random_state=1,min_samples_leaf=2, min_samples_split= 2)
parameters={}

parameters={'classifier__learning_rate':[0.015,0.02,0.025],
            'classifier__n_estimators':[100,300,500], 
            'classifier__max_depth':[2,3,4]}

gradient_test,gradient_best_estimator,gradient_output=supervised.crossvalidation_oversampling_test(classifier,parameters,X_train, y_train, X_test, y_test,label,False,oversampler,scoring)
#gradient_report,gradient_misclassification=supervised.output_report(gradient_output)

classifier_name='Gradient boosting'
param_name='n_estimators'
param_range =[5,10,100,200,500,1000,2000]
supervised.plot_validation_curve(X_train,y_train[label],classifier,classifier_name,param_range,param_name,scoring)


'''###########cat_descrizione, x=5##### 
#############returns
###################pol

###################pol e return
'''




##############################


'''OVERSAMPLING'''

neighbors=2
X_train_oversampled,y_train_oversampled=supervised.oversampler(X_train,y_train['classe'],'smote',neighbors)
'''undersampling'''
#X_train_undersampled,y_train_undersampled=supervised.undersampler(X_train,y_train,'random')
'''plot the result of sampling'''
supervised.sampling_train_PCA_plot(X_train_oversampled,y_train_oversampled,variables_mixed,'classe')


'''Hierarchical classification procedure'''

oversampler = supervised.RandomOverSampler(random_state=1)
classifier =  RandomForestClassifier(random_state=1)

parameters={'n_estimators':[300],
            'min_samples_leaf':(1,2,3,4),
            'min_samples_split':(1,2,3,4),
            'max_depth':(40,50,60),'bootstrap':[False]}

hierarchical_predictions=supervised.hierarchical_classification(X_train,y_train,X_test,y_test,variables,anagrafica,classifier,parameters,scoring) #AUC ROC NON FUNZIONA




''' 8 - Intersection method'''

'''Select 2 sets of variables and 2 classifiers, the function applies a classifier for each set and makes the intersection between the results, evaluating the accuracy'''

scoring='accuracy'
classifier1 =  RandomForestClassifier(random_state=1)
classifier2 =  RandomForestClassifier(random_state=1)

parameters1={'n_estimators':[300],
            'min_samples_leaf':(1,2,3,4),
            'min_samples_split':(2,3,4),
            'max_depth':(40,50,60),'bootstrap':[False]}
parameters2={'n_estimators':[300],
            'min_samples_leaf':(1,2,3,4),
            'min_samples_split':(2,3,4),
            'max_depth':(40,50,60),'bootstrap':[False]}

method='kneighbors'
train_intersection,test_intersection=supervised.intersection_classification(classifier1,classifier2,parameters1,parameters2,df,anagrafica,variables_pol,variables_return,label,method,scoring)


###############################################################################
