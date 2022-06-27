# -*- coding: utf-8 -*-

"""
Created on Mon Nov 11 09:24:29 2019

@author: python
"""
import os
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
import matplotlib.cm as cm

#os.chdir('\\\\salaanalisisrv\\ufficio_studi\\python\\Cluster\\Giulio\\def') #fonte libreria
import data_management_def as data
#os.chdir('\\\\salaanalisisrv\\ufficio_studi\\python\\Cluster\\Giulio') #per caricamento dati


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from imblearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV
from sklearn.metrics import accuracy_score,classification_report,roc_auc_score,balanced_accuracy_score,make_scorer
from pandas_ml import ConfusionMatrix
import unsupervised_learning_def as unsupervised
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import validation_curve


from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import BorderlineSMOTE
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import ADASYN
from imblearn.combine import SMOTETomek

from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score
from imblearn.pipeline import make_pipeline
from sklearn.experimental import enable_iterative_imputer #aggiunto
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.neighbors import KNeighborsRegressor


from imblearn.under_sampling import RandomUnderSampler
from imblearn.ensemble import BalanceCascade
from imblearn.ensemble import EasyEnsemble

def data_summary(X_train, y_train, X_test, y_test):
    """Summarize current state of dataset"""
    print('Train images shape:', X_train.shape)
    print('Train labels shape:', y_train.shape)
    print('Test images shape:', X_test.shape)
    print('Test labels shape:', y_test.shape)

def prepare_inputs(df,anagrafica,variables,label,train_size,shuffle_method,scaling=False,artificial=False):
    
    '''Splits dataframe in train and test sets:
        INPUT -dataframe
              -anagraphic variables (label, name, class..)
              -variables for classification
              -training set fraction
              -True: applies imputation using class averages to missing data
              -True: applies feature standardization
        OUTPUT: - feature array and labels for train and test set'''
    
    '''Split dataframe in training (validation - optional) and test using the chosen proportions'''
   
    if(shuffle_method=='stratified'):
        sss = StratifiedShuffleSplit(n_splits=5, test_size=1-train_size, random_state=1)
    
        for train_index, test_index in sss.split(df, df[label]):
            
            X_train, X_test = df.iloc[train_index,],df.iloc[test_index,]
            y_train, y_test = df[anagrafica].iloc[train_index,], df[anagrafica].iloc[test_index,]
        
        
        X_train=X_train[variables]    
        X_test=X_test[variables]  
        

    elif(shuffle_method==None):
        
        X=df[variables]
        y=df[anagrafica]

        X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=1-train_size,random_state=4)

    
    if(artificial!=False):
        X_train,y_train=artificial_funds(X_train,y_train,artificial)

    if(scaling!=False):
        '''feature standardization'''
        scaler=StandardScaler()
        scaler.fit_transform(X_train)
        scaler.transform(X_test)
        
    test=y_test.join(X_test)   
   #test=test.dropna()
    X_test=test[variables]
    y_test=test[anagrafica]
    
    X_train.index=range(len(X_train))
    X_test.index=range(len(X_test))
    y_train.index=range(len(y_train))
    y_test.index=range(len(X_test))

    print('numero categorie nel train set:{0}'.format(y_train[label].nunique()))
    print('numero categorie nel test set:{0}\n'.format(y_test[label].nunique()))
    
    return X_train,y_train,X_test,y_test

def simple_imputation(X_train,y_train,label,variables,anagrafica,classifier,strategy,scoring):
        '''Change NaN values in the train set with class averages or the global average where it's not possible'''
       # df_train=pd.concat([y_train,X_train],axis=1)
        X_train=np.matrix(X_train)
        if(strategy=='mean'):  
            imputer=SimpleImputer(missing_values=np.nan, strategy='mean')
            X_train=imputer.fit_transform(X_train)
        if(strategy=='median'):  
            imputer=SimpleImputer(missing_values=np.nan, strategy='median')
            X_train=imputer.fit_transform(X_train)

        #if(strategy=='mean_by_cate'):  
        #    for i in variables:
        #        df_train.loc[df_train[i].isnull(),i] = df_train.groupby(label)[i].transform(lambda x: x.mean())
        #    df_train[variables]=df_train[variables].fillna(df_train[variables].mean())
        
        #if(strategy=='median_by_cate'):  
        #    for i in variables:
        #       df_train.loc[df_train[i].isnull(),i] = df_train.groupby(label)[i].transform(lambda x: x.median())
        #    df_train[variables]=df_train[variables].fillna(df_train[variables].median())

        #df_train=remove_NaNs(df_train)
       # X_train=df_train[variables]
       # y_train=df_train[anagrafica]
        score_simple_imputer = cross_val_score(classifier,X_train, y_train[label],scoring=scoring,cv=5)
        print(score_simple_imputer.mean())
        

def imputation_scores(X_train,y_train,label,classifier,scoring):
    '''Selection of imputation method and score comparison
    '''
    #score_simple_imputer = pd.DataFrame()

    #for strategy in ('mean', 'median'):
    #
    #   estimator = make_pipeline(
    #            SimpleImputer(missing_values=np.nan, strategy=strategy),
    #            classifier)
    #
    #    score_simple_imputer[strategy] = cross_val_score(estimator, X_train, y_train[label], scoring='accuracy',cv=5)
    #print(score_simple_imputer.mean())

    estimators = [BayesianRidge(),
            DecisionTreeRegressor(max_features='sqrt', random_state=0),
            ExtraTreesRegressor(n_estimators=100, random_state=0),
            KNeighborsRegressor(n_neighbors=5)]
    
    score_iterative_imputer = pd.DataFrame()
    
    for impute_estimator in estimators:
        estimator = make_pipeline(
                IterativeImputer(random_state=0, estimator=impute_estimator),
                classifier)
    
        score_iterative_imputer[impute_estimator.__class__.__name__] = \
            cross_val_score(estimator, X_train, y_train[label], scoring=scoring,cv=5)
    print(score_iterative_imputer.mean())
    
def impute_missing_values(X_train,y_train,variables,label,method):
    ''' Missing values imputation
    INPUT:
        training set
        method: imputation method
    OUTPUT:
        training set with no missing values
    '''
    if(method=='mean_by_cate'):
        df_train=pd.concat([y_train,X_train],axis=1)
        for i in variables:
            df_train.loc[df_train[i].isnull(),i] = df_train.groupby(label)[i].transform(lambda x: x.mean())
        X_train=df_train[variables]
        imputer=SimpleImputer(missing_values=np.nan, strategy='mean')

    if(method=='mean'):
        #df_train=pd.concat([y_train,X_train],axis=1)
        #for i in variables:
        #    df_train.loc[df_train[i].isnull(),i] = df_train.groupby(label)[i].transform(lambda x: x.mean())
        #X_train=df_train[variables]
        imputer=SimpleImputer(missing_values=np.nan, strategy='mean')
    if(method=='median'):
        #df_train=pd.concat([y_train,X_train],axis=1)
        #for i in variables:
        #    df_train.loc[df_train[i].isnull(),i] = df_train.groupby(label)[i].transform(lambda x: x.median())
        #X_train=df_train[variables]
        imputer=SimpleImputer(missing_values=np.nan, strategy='mean')
    if(method=='bayesian_ridge'):
        imputer=IterativeImputer(random_state=0, estimator=BayesianRidge())
    if(method=='kneighbors'):
        imputer=IterativeImputer(random_state=0, estimator= KNeighborsRegressor(n_neighbors=5))
    if(method=='tree'):
        imputer=IterativeImputer(random_state=0, estimator=DecisionTreeRegressor(max_features='sqrt', random_state=0))
    if(method=='extra_trees'):
        imputer=IterativeImputer(random_state=0, estimator=ExtraTreesRegressor(n_estimators=100, random_state=0))
    
    X_train=imputer.fit_transform(X_train)
    
    return X_train

def artificial_funds(X_train,y_train,n_iterations):
    
    '''Add artificially created funds to the training set to simulate managment costs
        xk: list of costs'''
    
    xk=[0.05,0.1]
    pk=[1/len(xk)]*len(xk)
    
    lista_X=[0]*n_iterations
    for i in range(n_iterations):
        lista_X[i]=X_train
        e=np.random.choice(xk,len(lista_X[i]),p=pk)
        e=e/12
        e=e[:,np.newaxis]
        lista_X[i]=np.multiply(lista_X[i],1-e)
        noise=1-np.random.normal(loc=0,scale=0.05,size=len(lista_X[i]))
        noise=noise[:,np.newaxis]
        lista_X[i]=np.multiply(lista_X[i],noise)
        
    lista_y=[0]*n_iterations
    for i in range(n_iterations):
        lista_y[i]=y_train
    
    X_train=pd.concat(lista_X,ignore_index=True)
    y_train=pd.concat(lista_y,ignore_index=True)

    return X_train,y_train

def select_oversampler(method, neighbors):
    
    if(method=='random'):
        ov = RandomOverSampler(random_state=1)
    if(method=='smote'):
        ov = SMOTE(k_neighbors=neighbors,random_state=1)
    if(method=='borderlinesmote1'):
        ov = BorderlineSMOTE(k_neighbors=neighbors,random_state=1,kind='borderline-1')
    if(method=='borderlinesmote2'):
        ov = BorderlineSMOTE(k_neighbors=neighbors,random_state=1,kind='borderline-2')
    if(method=='adasyn'):
        ov = ADASYN(n_neighbors=neighbors,random_state=1)
    if(method=='smotetomek'):
        ov = SMOTETomek(random_state=1,smote=os)
    if(method=='smoteenn'):
        ov = SMOTEENN(random_state=1,smote=os)
    return ov

def oversampler(X_train,y_train,method,neighbors):
    if(method=='random'):
        ov = RandomOverSampler(random_state=1)
        X_train, y_train= ov.fit_resample(X_train, y_train)
    if(method=='smote'):
        ov = SMOTE(k_neighbors=neighbors,random_state=1)
        X_train, y_train=ov.fit_sample(X_train, y_train)
        y_train=y_train[:,np.newaxis]

    if(method=='borderlinesmote1'):
        ov = BorderlineSMOTE(k_neighbors=neighbors,random_state=1,kind='borderline-1')
        X_train, y_train=ov.fit_sample(X_train, y_train)
        y_train=y_train[:,np.newaxis]

    if(method=='borderlinesmote2'):
        ov = BorderlineSMOTE(k_neighbors=neighbors,random_state=1,kind='borderline-2')
        X_train, y_train=ov.fit_sample(X_train, y_train)
        y_train=y_train[:,np.newaxis]

    if(method=='adasyn'):
        ov = ADASYN(n_neighbors=neighbors,random_state=1)
        X_train, y_train=ov.fit_sample(X_train, y_train)
        y_train=y_train[:,np.newaxis]
        
    if(method=='smotetomek'):
        ov = SMOTETomek(random_state=1,smote=os)
        X_train, y_train= ov.fit_resample(X_train, y_train)
        y_train=y_train[:,np.newaxis]
        
    if(method=='smoteenn'):
        ov = SMOTEENN(random_state=1,smote=os)
        X_train, y_train=ov.fit_sample(X_train, y_train)
        y_train=y_train[:,np.newaxis]

    return X_train,y_train

def undersampler(X_train,y_train,undersampler):
    
    if(undersampler=='random'):
        us=RandomUnderSampler(random_state=1)
        X_train, y_train= us.fit_resample(X_train, y_train)
        
    if(undersampler=='easyensemble'):
        us = EasyEnsemble(random_state=0, n_subsets=10)
        X_train,y_train=us.fit_resample(X_train, y_train)

        X_train=X_train[0]
        y_train=y_train[0]
        y_train=y_train[:,np.newaxis]

    if(undersampler=='cascade'):
        
        us = BalanceCascade(random_state=0,
                    estimator=LogisticRegression(random_state=0))
        X_train,y_train=X_train,y_train=us.fit_resample(X_train, y_train)

        X_train=X_train[0]
        y_train=y_train[0]
        y_train=y_train[:,np.newaxis]

    return X_train,y_train

def random_guess_classifier_theoretical(y,name):
    '''theoretical accuracy'''
    classfreq=y.value_counts()/len(y)
    print("the random guess classifier has an accuracy of {0} on the {1} set".format(sum(classfreq*classfreq),name))

def random_guess_classifier(X_train,y_train,X_test,y_test,strategy):
    '''strategy: (stratified, most_frequent, prior, uniform, constant)'''
    dummy=DummyClassifier(strategy=strategy)
    dummy.fit(X_train,y_train)
    print("the random guess classifier has an accuracy of {0:.3f} on the train set".format(dummy.score(X_train, y_train)))
    print("the random guess classifier has an accuracy of {0:.3f} on the test set".format(dummy.score(X_test, y_test)))


def train_PCA_plot(X_train,y_train,variables,label):
    
    X_train_per_plot=pd.DataFrame(X_train, columns=variables)
    X_train_per_plot[label]=y_train
    unsupervised.PCA_plot(X_train_per_plot,variables,3,label)

def my_autopct(pct):
    return ('%.1f' % pct) if pct > 5 else ''

def pie_plot_classi(y_test):
    class_amount=data.category_count(y_test,'classe')
    class_amount['frequency']=class_amount['quantity']/len(y_test)
    class_amount.index=range(len(class_amount))
    
    for i in class_amount.index:
        if (class_amount.iloc[i]['frequency']<0.05):
            class_amount.iloc[i]['classe']=' '
    
    fig1, ax1 = plt.subplots( figsize=(8,8))
    ax1.pie(class_amount['frequency'], labels=class_amount['classe'],   wedgeprops=dict(width=0.3, edgecolor='w'),autopct= my_autopct,
            startangle=90)
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

    plt.show()


def sampling_train_PCA_plot(X_train,y_train,variables,label):

    LE = unsupervised.LabelEncoder()
    y_train2=pd.DataFrame(y_train,columns=[label])
    y_train2['code'] = LE.fit_transform(y_train2[label])
    X_train2=pd.DataFrame(X_train, columns=variables)
    testdf2=y_train2.join(X_train2)
    unsupervised.PCA_plot(testdf2,variables,3,label)
    
def crossvalidation_oversampling_test(classifier,parameters,X_train,y_train,X_test,y_test,label,do_oversampling,oversampler,scoring):
    '''Define oversampling-model pipeline
    INPUT:
        classifier
        parameters
        Training and Test sets
        do_oversampling: True/False
        oversampler
    OUTPUT:
        crossvalidation results
        best estimator parameters
        output for report generation
    '''
    start_time = time.time()

    print('using {0} score'.format(scoring))
    
    if(scoring=='roc_auc_ovr'): #only for auc socre
        indexes_to_remove=y_train[~y_train[label].isin(y_test[label])][label].index #need to remove the classes not included in the test
        X_train=X_train.drop(indexes_to_remove)
        y_train=y_train.drop(indexes_to_remove)
        
    if(do_oversampling!=False):
        print('performing oversampling')
        model = Pipeline([('oversampler', oversampler),('classifier', classifier)])
    if(do_oversampling==False):
        print('not performing oversampling')
        model=classifier
        parameters = {key.replace("classifier__", ""): value for key, value in parameters.items()} #to avoid optimization errors with gridsearch
    grid = GridSearchCV(model,parameters,scoring=scoring,return_train_score=True,cv=3) # optimize parameters using predefined combinations
    grid.fit(X_train, y_train[label])
    clftest=pd.DataFrame(grid.cv_results_)
    
    print('best parameters:{0:s}'.format(str(grid.best_estimator_)))
    print('accuracy score on training set: {0:f}'.format(grid.score(X_train,y_train[label])))
    print('mean accuracy score on validation sets:{0:f}'.format(grid.best_score_))
    if(scoring=='accuracy'):
            print('accuracy score on test set:{0:f}'.format(accuracy_score(y_test[label], grid.best_estimator_.predict(X_test))))
    if(scoring=='balanced_accuracy'):
            print('balanced accuracy score on test set:{0:f}'.format(balanced_accuracy_score(y_test[label], grid.best_estimator_.predict(X_test))))
    if(scoring=='roc_auc_ovr'):
            #y_score = grid.predict_proba(X_test)
            #print('ROC AUC score on test set:{0:f}'.format(roc_auc_score(y_test[label],y_score,multi_class='ovr')))
            print('ROC AUC score on test set:{0:f}'.format(grid.score(X_test,y_test[label])))

    output=pd.DataFrame()
    output['real_values']=y_test[label]
    output['predicted_values']=grid.best_estimator_.predict(X_test)

    if(label=='cat_descrizione'):
            output['classe']=y_test['classe']
            output['super_classe']=y_test['super_classe']
            output['classe_pred']=0

            for i in output.index:
                try:
                    output['classe_pred'].iloc[i]=y_test[y_test['cat_descrizione']==output['predicted_values'].iloc[i]]['classe'].iloc[0]
                except:
                    print('classe assente')
                    output['classe_pred'].iloc[i]=y_test['classe'].iloc[i]

    elif(label=='classe'):
            output['super_classes']=y_test['super_classe']
            output['super_classe_pred']=0
            for i in output.index:
                try:
                    output['super_classe_pred'].iloc[i]=y_test[y_test['classe']==output['predicted_values'].iloc[i]]['super_classe'].iloc[0]
                except:
                    print('super_classe assente')
                    output['super_classe_pred'].iloc[i]=y_test['super_classe'].iloc[i]

    
    elapsed_time = time.time() - start_time
    print('elapsed time:{0:.2f} seconds'.format(elapsed_time))
    return clftest, grid.best_estimator_,output


def randomized_crossvalidation_oversampling_test(classifier,parameters,X_train,y_train,X_test,y_test,label,do_oversampling,oversampler,scoring):
    '''Define oversampling-model pipeline
    INPUT:
        classifier
        parameters
        Training and Test sets
        do_oversampling: True/False
        oversampler
    OUTPUT:
        crossvalidation results
        best estimator parameters
        output for report generation
    '''
    start_time = time.time()
    print('using {0} score'.format(scoring))

    if(do_oversampling!=False):
        print('performing oversampling')
        model = Pipeline([('oversampler', oversampler),('classifier', classifier)])
    if(do_oversampling==False):
        print('not performing oversampling')
        model=classifier
        parameters = {key.replace("classifier__", ""): value for key, value in parameters.items()}
    grid = RandomizedSearchCV(model,parameters,scoring=scoring,cv=3,return_train_score=True,n_iter=10,random_state=1) #Extracts the series of parameters and optimizes
    grid.fit(X_train, y_train[label])
    clftest=pd.DataFrame(grid.cv_results_)
    
    print('best parameters:{0:s}'.format(str(grid.best_estimator_)))
    print('accuracy score on training set: {0:f}'.format(grid.score(X_train,y_train[label])))
    print('mean accuracy score on validation sets:{0:f}'.format(grid.best_score_))
    print('accuracy score on test set:{0:f}'.format(accuracy_score(y_test[label], grid.best_estimator_.predict(X_test))))

    output=pd.DataFrame()
    output['real_values']=y_test[label]
    output['predicted_values']=grid.best_estimator_.predict(X_test)

    if(label=='cat_descrizione'):
            output['real_classes']=y_test['classe']
            output['real_super_classes']=y_test['super_classe']
            
    elif(label=='classe'):
            output['real_super_classes']=y_test['super_classe']

    elapsed_time = time.time() - start_time
    print('elapsed time:{0:.2f} seconds'.format(elapsed_time))

    return clftest,grid.best_estimator_,output


def output_report(output,true_label,predicted_label):
    ''' Generates report with confusion matrix'''
    report=classification_report(output[true_label],output[predicted_label])
    
    confu = ConfusionMatrix(output[true_label], output[predicted_label])
    confu.plot(normalized=True,cmap = 'Greens')
    
    misclassified_entries=output[np.array(output[true_label])!= output[predicted_label]]

    misclassified_frequencies=data.category_count(misclassified_entries,true_label)
    
    #total_amount=pd.DataFrame(output.groupby(['real_classes']).count())
    #comparison=misclassified_frequencies.merge(total_amount,left_on='real_classes',right_on='real_classes')
    #comparison['ratio']=comparison['quantity']/comparison['real_values']
    
    misclassified_frequencies['quantity']=misclassified_frequencies['quantity']/len(misclassified_entries)
    misclassified_frequencies.index=misclassified_frequencies[true_label]
    misclassified_frequencies.plot.pie(y='quantity',figsize=(10,10))
    
    
    #test_confusion = pd.crosstab(output['real_values'], output['predicted_values'])
    return report,misclassified_entries

def plot_validation_curve(X_train,y_train,classifier,classifer_name,param_range,param_name,scoring):
    '''Plots the validation curves referring to chosen hyperparameter for the classifier'''
    print('using {0} score'.format(scoring))

    train_scores, test_scores = validation_curve(classifier, X_train, y_train, 
                                             param_name=param_name, param_range=param_range,
                                             scoring=scoring, n_jobs=1)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.figure(figsize=(7,7))
    plt.title("Validation Curve with {0}".format(classifer_name))
    plt.xlabel(param_name)
    plt.ylabel("{0} score".format(scoring))
    #plt.ylim(0.0, 1.1)
    lw = 2
    plt.semilogx(param_range, train_scores_mean, label="Training score",
             color="grey", lw=lw)
    plt.fill_between(param_range, train_scores_mean - train_scores_std,
                 train_scores_mean + train_scores_std, alpha=0.2,
                 color="grey", lw=lw)
    plt.semilogx(param_range, test_scores_mean, label="Cross-validation score",
             color="red", lw=lw)
    plt.fill_between(param_range, test_scores_mean - test_scores_std,
                 test_scores_mean + test_scores_std, alpha=0.2,
                 color="red", lw=lw)
    plt.legend(loc="best")
    plt.show()




def hierarchical_classification(X_train,y_train,X_test,y_test,variables,anagrafica,classifier,parameters,scoring):
    """INPUT: -dataframe dei fondi
              -lista di variabili da usare per la classificazione
              -lista di variabili da usare come anagrafica
       OUTPUT:
              -dataframe con predizioni """
              
    print('using {0} score'.format(scoring))
        
    if(scoring=='roc_auc_ovr'): #solo per roc auc score
        indexes_to_remove=y_train[~y_train['cat_descrizione'].isin(y_test['cat_descrizione'])]['cat_descrizione'].index #devo togliere dal training le classi non contenute nel test se no non funziona
        X_train=X_train.drop(indexes_to_remove)
        y_train=y_train.drop(indexes_to_remove)
        y_train.index=range(len(y_train))
        X_train.index=range(len(X_train))
        
    variabili_con_indici_train=np.append(X_train,y_train,axis=1)
    variabili_con_indici_test=np.append(X_test,y_test,axis=1)

    variabili_con_indici_train=pd.DataFrame(variabili_con_indici_train,columns=variables+anagrafica)
    variabili_con_indici_test=pd.DataFrame(variabili_con_indici_test,columns=variables+anagrafica)
         
    '''FIRST LEVEL: SUPERCLASSI'''
    grid = GridSearchCV(classifier,parameters,cv=5,scoring=scoring,return_train_score=True) # ottimizza i parametri attraverso combinazioni predefinite da un dizionario
    grid.fit(X_train, y_train['super_classe'])
    
    '''predicted train labels'''
    pred = pd.DataFrame(grid.predict(X_train),index=range(len(X_train)),columns=["super_classe_pred"])
    pred.reset_index()
    y_train.index=range(len(y_train))
    superclassi_train = pd.concat([y_train, pred], axis=1)
    scores1_train=grid.score(X_train,y_train['super_classe']) #percentuale di fondi azzeccati (accuracy score)
    scores1_val=grid.best_score_
    bestestimator=grid.best_estimator_
    bestestimator.fit(X_train,y_train['super_classe'])

    '''predicted test labels'''
    pred = pd.DataFrame(bestestimator.predict(X_test),index=range(len(X_test)),columns=["super_classe_pred"])
    pred.reset_index()
    y_test.index=range(len(y_test))
    superclassi_test = pd.concat([y_test, pred], axis=1)
    #scores1_test=bestestimator.score(X_test,y_test['super_classe']) #percentuale di fondi azzeccati (accuracy score)
    scores1_test=grid.score(X_test,y_test['super_classe'])
    print('training score at superclasse level:{0}'.format(scores1_train))
    print('cross-validation score at superclasse level:{0}'.format(scores1_val))
    print('test score at superclasse level:{0}\n'.format(scores1_test))
    
    '''formatting the results'''
    X_train=pd.DataFrame(X_train,columns=variables)
    X_test=pd.DataFrame(X_test,columns=variables)

    superclassi_train=superclassi_train.join(X_train)
    superclassi_test=superclassi_test.join(X_test)
    
    #superclassi_train.drop( superclassi_train[ superclassi_train['super_classe'] !=  superclassi_train['super_classe_pred'] ].index , inplace=True) #drop the wrong ones in the training set
    #superclassi_train.index=range(len(superclassi_train))
    
    gruppi_superclassi_train = superclassi_train.groupby('super_classe')    
    gruppi_superclassi_train=[gruppi_superclassi_train.get_group(x) for x in gruppi_superclassi_train.groups]
    
    gruppi_superclassi_test=superclassi_test.groupby('super_classe_pred')    
    gruppi_superclassi_test=[gruppi_superclassi_test.get_group(x) for x in gruppi_superclassi_test.groups]
    
    '''SECOND LEVEL: CLASSI'''
    superclassi_pred=[]
    for i in range(len(gruppi_superclassi_test)):
        superclassi_pred.append(str(gruppi_superclassi_test[i]['super_classe_pred'].iloc[0]))
   
    k=[]
    y_train=[]
    X_train=[]
    for i in superclassi_pred:
        k.append(superclassi_train[superclassi_train['super_classe']==i]['classe'].nunique())
        y_train.append(superclassi_train[superclassi_train['super_classe']==i][anagrafica+["super_classe_pred"]])
        X_train.append(superclassi_train[superclassi_train['super_classe']==i][variables])
    for i in range(len(X_train)):
        X_train[i]=pd.DataFrame(X_train[i].values)
        X_train[i]=np.array(X_train[i],dtype=float)

    y_test=[]
    X_test=[]
    for i in superclassi_pred:
        y_test.append(superclassi_test[superclassi_test['super_classe_pred']==i][anagrafica+["super_classe_pred"]])
        X_test.append(superclassi_test[superclassi_test['super_classe_pred']==i][variables])
    for i in range(len(X_test)):
        X_test[i]=pd.DataFrame(X_test[i].values)
        X_test[i]=np.array(X_test[i],dtype=float)

    '''predicted train labels'''
    pred=[0]*len(superclassi_pred)
    pred_train=[0]*len(superclassi_pred)
    scores2_train=[0]*len(superclassi_pred) 
    scores2_val=[0]*len(superclassi_pred)    
    grid=[0]*len(superclassi_pred)
    bestestimator=[0]*len(superclassi_pred)

    for i in range(len(superclassi_pred)):
    
        if(k[i]==1):
            pred[i] = pd.DataFrame(y_train[i]["classe"].iloc[0],index=range(len(X_train[i])),columns=["classe_pred"])
            pred[i].reset_index()
            y_train[i].index=range(len(y_train[i]))
            pred_train[i] = pd.concat([y_train[i], pred[i]], axis=1)
            pred_train[i]= pred_train[i].merge(variabili_con_indici_train,left_on=anagrafica,right_on=anagrafica)
            scores2_train[i]=1 #percentuale di fondi azzeccati (accuracy score)
            scores2_val[i]=1 #percentuale di fondi azzeccati (accuracy score)

        else: 
            grid[i] =  GridSearchCV(classifier,parameters,cv=5,scoring=scoring,return_train_score=True) # ottimizza i parametri attraverso combinazioni predefinite da un dizionario
            grid[i].fit(X_train[i],y_train[i]['classe'])
            pred[i] = pd.DataFrame(grid[i].predict(X_train[i]),index=range(len(X_train[i])),columns=["classe_pred"])
            pred[i].reset_index()
            y_train[i].index=range(len(y_train[i]))
            pred_train[i] = pd.concat([y_train[i], pred[i]], axis=1)
            pred_train[i]= pred_train[i].merge(variabili_con_indici_train,left_on=anagrafica,right_on=anagrafica)
            scores2_train[i]=grid[i].score(X_train[i],y_train[i]['classe']) #percentuale di fondi azzeccati (accuracy score)
            scores2_val[i]=grid[i].best_score_ #percentuale di fondi azzeccati (accuracy score)
            bestestimator[i]=grid[i].best_estimator_

    '''predicted test labels'''
    pred=[0]*len(superclassi_pred)
    pred_test=[0]*len(superclassi_pred)
    scores2_test=[0]*len(superclassi_pred)    
    #grid=[0]*len(X_test)
    for i in range(len(superclassi_pred)):
        
        if(k[i]==1):
            pred[i] = pd.DataFrame(y_test[i]["classe"].iloc[0],index=range(len(X_test[i])),columns=["classe_pred"])
            pred[i].reset_index()
            y_test[i].index=range(len(y_test[i]))
            pred_test[i] = pd.concat([y_test[i], pred[i]], axis=1)
            pred_test[i]= pred_test[i].merge(variabili_con_indici_test,left_on=anagrafica,right_on=anagrafica)
            scores2_test[i]=1 #percentuale di fondi azzeccati (accuracy score)
        else: 
            bestestimator[i].fit(X_train[i],y_train[i]['classe'])
            pred[i] = pd.DataFrame(bestestimator[i].predict(X_test[i]),index=range(len(X_test[i])),columns=["classe_pred"])
            pred[i].reset_index()
            y_test[i].index=range(len(y_test[i]))
            pred_test[i] = pd.concat([y_test[i], pred[i]], axis=1)
            pred_test[i]= pred_test[i].merge(variabili_con_indici_test,left_on=anagrafica,right_on=anagrafica)
            #scores2_test[i]=bestestimator[i].score(X_test[i],y_test[i]['classe']) #percentuale di fondi azzeccati (accuracy score)   
            scores2_test[i]=grid[i].score(X_test[i],y_test[i]['classe'])
    
    train_lengths=[len(x) for x in X_train]
    avg_train_score2=np.average(scores2_train,weights=train_lengths)
    val_lengths=[len(x) for x in X_train]
    avg_val_score2=np.average(scores2_val,weights=val_lengths)
    test_lengths=[len(x) for x in X_test]
    avg_test_score2=np.average(scores2_test,weights=test_lengths)

    print('training score at classe level:{0}'.format(avg_train_score2))
    print('cross-validation score at classe level:{0}'.format(avg_val_score2))
    print('test score at classe level:{0}\n'.format(avg_test_score2))


    '''formatting the results'''
    classi_train=pd.concat(pred_train)
    classi_test=pd.concat(pred_test)
    
    #classi_train.drop( classi_train[ classi_train['classe'] !=  classi_train['classe_pred'] ].index , inplace=True) #drop the wrong ones in the training set
    #classi_train.index=range(len(classi_train))

    gruppi_classi_train = classi_train.groupby('classe')    
    gruppi_classi_train=[gruppi_classi_train.get_group(x) for x in gruppi_classi_train.groups]
    
    gruppi_classi_test=classi_test.groupby('classe_pred')    
    gruppi_classi_test=[gruppi_classi_test.get_group(x) for x in gruppi_classi_test.groups]

    '''THIRD LEVEL: CATEGORIE'''
    
    classi_pred=[]
    for i in range(len(gruppi_classi_test)):
        classi_pred.append(str(gruppi_classi_test[i]['classe_pred'].iloc[0]))

    k=[]
    y_train=[]
    X_train=[]
    for i in classi_pred:
        k.append(classi_train[classi_train['classe']==i]['cat_descrizione'].nunique())
        y_train.append(classi_train[classi_train['classe']==i][anagrafica+["classe_pred"]])
        X_train.append(classi_train[classi_train['classe']==i][variables])
    for i in range(len(X_train)):
        X_train[i]=pd.DataFrame(X_train[i].values)
        X_train[i]=np.array(X_train[i],dtype=float)

    y_test=[]
    X_test=[]
    for i in classi_pred:
        y_test.append(classi_test[classi_test['classe_pred']==i][anagrafica+["classe_pred"]])
        X_test.append(classi_test[classi_test['classe_pred']==i][variables])
    for i in range(len(X_test)):
        X_test[i]=pd.DataFrame(X_test[i].values)
        X_test[i]=np.array(X_test[i],dtype=float)

  
    pred=[0]*len(X_train)
    pred_train=[0]*len(X_train)
    scores3_train=[0]*len(X_train)    
    scores3_val=[0]*len(X_train)    
    grid=[0]*len(X_train)
    bestestimator=[0]*len(X_train)

    for i in range(len(X_train)):
    
        if(k[i]==1):
            pred[i] = pd.DataFrame(y_train[i]["cat_descrizione"].iloc[0],index=range(len(X_train[i])),columns=["cat_descrizione_pred"])
            pred[i].reset_index()
            y_train[i].index=range(len(y_train[i]))
            pred_train[i] = pd.concat([y_train[i], pred[i]], axis=1)
            pred_train[i]= pred_train[i].merge(variabili_con_indici_train,left_on=anagrafica,right_on=anagrafica)
            scores3_train[i]=1 #percentuale di fondi azzeccati (accuracy score)
        else: 
            grid[i] = GridSearchCV(classifier,parameters,cv=3,scoring=scoring,return_train_score=True) # ottimizza i parametri attraverso combinazioni predefinite da un dizionario
            grid[i].fit(X_train[i],y_train[i]['cat_descrizione'])
            pred[i] = pd.DataFrame(grid[i].predict(X_train[i]),index=range(len(X_train[i])),columns=["cat_descrizione_pred"])
            pred[i].reset_index()
            y_train[i].index=range(len(y_train[i]))
            pred_train[i] = pd.concat([y_train[i], pred[i]], axis=1)            
            pred_train[i]= pred_train[i].merge(variabili_con_indici_train,left_on=anagrafica,right_on=anagrafica)
            scores3_train[i]=grid[i].score(X_train[i],y_train[i]['cat_descrizione']) #percentuale di fondi azzeccati (accuracy score)
            scores3_val[i]=grid[i].best_score_  
            bestestimator[i]=grid[i].best_estimator_

    pred=[0]*len(X_test)
    pred_test=[0]*len(X_test)
    scores3_test=[0]*len(X_test)    
    #grid=[0]*len(X_test)
    for i in range(len(X_test)):
        if(k[i]==1):
            pred[i] = pd.DataFrame(y_test[i]["cat_descrizione"].iloc[0],index=range(len(X_test[i])),columns=["cat_descrizione_pred"])
            pred[i].reset_index()
            y_test[i].index=range(len(y_test[i]))
            pred_test[i] = pd.concat([y_test[i], pred[i]], axis=1)
            pred_test[i]= pred_test[i].merge(variabili_con_indici_test,left_on=anagrafica,right_on=anagrafica)
            scores3_test[i]=1 #percentuale di fondi azzeccati (accuracy score)
        else: 
            bestestimator[i].fit(X_train[i],y_train[i]['cat_descrizione'])
            pred[i] = pd.DataFrame(bestestimator[i].predict(X_test[i]),index=range(len(X_test[i])),columns=["cat_descrizione_pred"])
            pred[i].reset_index()
            y_test[i].index=range(len(y_test[i]))
            pred_test[i] = pd.concat([y_test[i], pred[i]], axis=1)
            
            pred_test[i]= pred_test[i].merge(variabili_con_indici_test,left_on=anagrafica,right_on=anagrafica)
            
            #scores3_test[i]=bestestimator[i].score(X_test[i],y_test[i]['cat_descrizione']) #percentuale di fondi azzeccati (accuracy score)   
            scores3_test[i]=grid[i].score(X_test[i],y_test[i]['cat_descrizione'])


    train_lengths=[len(x) for x in X_train]
    avg_train_score3=np.average(scores3_train,weights=train_lengths)
    val_lengths=[len(x) for x in X_train]
    avg_val_score3=np.average(scores3_val,weights=val_lengths)
    test_lengths=[len(x) for x in X_test]
    avg_test_score3=np.average(scores3_test,weights=test_lengths)
    
    print('training score at categoria level:{0}'.format(avg_train_score3))
    print('cross-validation score at categoria level:{0}'.format(avg_val_score3))
    print('test score at categoria level:{0}\n'.format(avg_test_score3))
    
    return pred_test


def multiperiod_classification(lista,anagrafica, classifier):
    
    """INPUT:- list of dataframes split with respect to period 
             - list of anagraphic variables
       OUTPUT: - list of predicted classes and accuracy scores"""
    
    X=[0]*len(lista)
    for i in range(len(lista)):
        X[i]=lista[i][lista[i].columns[~lista[i].columns.isin(anagrafica)]]   
        X[i]=pd.DataFrame(X[i].values)
        X[i]=np.array(X[i],dtype=float)
        
    """label list"""
    y=[0]*len(lista)
    for i in range(len(lista)):
        y[i]=lista[i][lista[i].columns[lista[i].columns.isin(anagrafica)]]   
    
    
    label="cat_descrizione" #su quale livello della gerarchia fare la classificazione
    k=[0]*len(lista)
    for i in range(len(lista)):
        k[i]=y[i][label].nunique()
    
    X_train, X_test, y_train, y_test = [0]*len(lista),[0]*len(lista),[0]*len(lista),[0]*len(lista)
    for i in range(len(lista)):
        X_train[i], X_test[i], y_train[i], y_test[i] = train_test_split(X[i], y[i], test_size=0.2)
    
    scaler = MinMaxScaler() # da importare
    for i in range(len(lista)):
        X_train[i] = scaler.fit_transform(X_train[i])
        X_test[i] = scaler.transform(X_test[i])
        
    del i,anagrafica
        
    """training the model"""
    clf=[0]*len(lista)
    for i in range(len(lista)):
        clf[i]=classifier
        clf[i].fit(X_train[i],y_train[i][label])
    
    """train set evaluation, accuracy"""
    
    pred=[0]*len(lista)
    pred_train=[0]*len(lista)
    train_scores=[0]*len(lista)
    for i in range(len(lista)):
        pred[i] = pd.DataFrame(clf[i].predict(X_train[i]),index=range(len(X_train[i])),columns=["cat_descrizione_pred"])
        y_train[i].index=range(len(y_train[i]))
        pred_train[i] = pd.concat([y_train[i], pred[i]], axis=1)
        train_scores[i]=clf[i].score(X_train[i],y_train[i][label]) #percentuale di fondi azzeccati (accuracy score)    
    
    """test set evaluation, accuracy"""
    
    pred=[0]*len(lista)
    pred_test=[0]*len(lista)
    test_scores=[0]*len(lista)
    for i in range(len(lista)):
        pred[i] = pd.DataFrame(clf[i].predict(X_test[i]),index=range(len(X_test[i])),columns=["cat_descrizione_pred"])
        y_test[i].index=range(len(y_test[i]))
        pred_test[i] = pd.concat([y_test[i], pred[i]], axis=1)
        test_scores[i]=clf[i].score(X_test[i],y_test[i][label]) #percentuale di fondi azzeccati (accuracy score)
    
    return pred_train, train_scores, pred_test, test_scores

def intersection_classification(classifier_algorithm1,classifier_algorithm2,parameters1,parameters2,df,anagrafica,variables1,variables2,label,method,scoring):
    ''' Classification method with intersection
    INPUT:
        classifier_algorithm1/2 : classification models
        df: dataframe with both variable types
        anagrafica: list of anagraphic columns
        variables1: list of features 1st classifier
        variables2: list of features 2nd classifier
        label: classification targer
        scoring: scoring method
    '''
    print('using {0} score'.format(scoring))
    X_train,y_train,X_test,y_test=prepare_inputs(df,anagrafica,variables1+variables2,label,0.9,'stratified',scaling=False,artificial=False) #split all'interno della funzione
    if(scoring=='roc_auc_ovr'): #solo per roc auc score
        indexes_to_remove=y_train[~y_train[label].isin(y_test[label])][label].index #devo togliere dal training le classi non contenute nel test se no non funziona
        X_train=X_train.drop(indexes_to_remove)
        y_train=y_train.drop(indexes_to_remove)

    X_train=impute_missing_values(X_train,y_train,variables1+variables2,label,method)  
    X_test=impute_missing_values(X_test,y_test,variables1+variables2,label,method)  
    X_test=pd.DataFrame(X_test,columns=variables1+variables2)

    classifier1=classifier_algorithm1
    classifier2=classifier_algorithm2
    
    X_train=pd.DataFrame(X_train,columns=variables1+variables2)
    
    '''classification of the first df'''
    grid = GridSearchCV(classifier1,parameters1,scoring=scoring,cv=3,return_train_score=True) # ottimizza i parametri attraverso combinazioni predefinite da un dizionario
    grid.fit(X_train[variables1], y_train[label])
    score1_val=grid.best_score_
    score1_test=grid.score(X_test[variables1],y_test[label])
    score1_train=grid.score(X_train[variables1],y_train[label])
    
    print('training score of the first dataframe:{0}'.format(score1_train))
    print('cross-validaiton score of the first dataframe:{0}'.format(score1_val))
    print('test score of the first dataframe:{0}\n'.format(score1_test))

    pred_train_1 = pd.DataFrame(grid.predict(X_train[variables1]),index=range(len(X_train[variables1])),columns=["predicted_1"])
    y_train.index=range(len(y_train[label]))
    pred_train_1= pd.concat([y_train, pred_train_1], axis=1)

    pred_test_1 = pd.DataFrame(grid.predict(X_test[variables1]),index=range(len(X_test[variables1])),columns=["predicted_1"])
    y_test.index=range(len(y_test[label]))
    pred_test_1= pd.concat([y_test, pred_test_1], axis=1)
    
    '''classification of the second df'''
    grid = GridSearchCV(classifier2,parameters2,scoring=scoring,cv=3,return_train_score=True) # ottimizza i parametri attraverso combinazioni predefinite da un dizionario
    grid.fit(X_train[variables2], y_train[label])
    score2_val=grid.best_score_
    score2_test=grid.score(X_test[variables2],y_test[label])
    score2_train=grid.score(X_train[variables2],y_train[label])
    
    print('training score of the second dataframe:{0}'.format(score2_train))
    print('cross-validaiton score of the second dataframe:{0}'.format(score2_val))
    print('test score of the second dataframe:{0}\n'.format(score2_test))
    
    pred_train_2 = pd.DataFrame(grid.predict(X_train[variables2]),index=range(len(X_train[variables2])),columns=["predicted_2"])
    y_train.index=range(len(y_train[label]))
    pred_train_2= pd.concat([y_train, pred_train_2], axis=1)

    pred_test_2 = pd.DataFrame(grid.predict(X_test[variables2]),index=range(len(X_test[variables2])),columns=["predicted_2"])
    y_test.index=range(len(y_test[label]))
    pred_test_2= pd.concat([y_test, pred_test_2], axis=1)
    
    '''intersection of the two train df'''
    train_intersection_df=pd.merge(pd.merge(pred_train_1,pred_train_2,on='ana_ticker'),df[anagrafica],on='ana_ticker')
    train_intersection_df=train_intersection_df[['ana_name_x','ana_ticker','cat_descrizione_x','classe_x','super_classe_x','predicted_1','predicted_2']]
    train_intersection_df=train_intersection_df.rename(columns={'ana_name_x':'ana_name','cat_descrizione_x':'cat_descrizione','classe_x':'classe','super_classe_x':'super_classe'})
    train_intersection_df['intersection']=np.nan
    n_intersections=0
    
    for i in range(len(train_intersection_df)):
        if(train_intersection_df['predicted_1'][i]==train_intersection_df['predicted_2'][i]):
            train_intersection_df['intersection'][i]=train_intersection_df['predicted_1'][i]
            n_intersections+=1
        else:
            train_intersection_df['intersection'][i]=0
            
    train_intersection_rate=n_intersections/len(train_intersection_df)
    if (scoring=='accuracy'):
        train_accuracy=accuracy_score(train_intersection_df[train_intersection_df['intersection']!=0]['cat_descrizione'], train_intersection_df[train_intersection_df['intersection']!=0]['intersection'])
    elif(scoring=='balanced_accuracy'):
        train_accuracy=balanced_accuracy_score(train_intersection_df[train_intersection_df['intersection']!=0]['cat_descrizione'], train_intersection_df[train_intersection_df['intersection']!=0]['intersection'])

    '''intersection of the two test df'''
    test_intersection_df=pd.merge(pd.merge(pred_test_1,pred_test_2,on='ana_ticker'),df[anagrafica],on='ana_ticker')
    test_intersection_df=test_intersection_df[['ana_name_x','ana_ticker','cat_descrizione_x','classe_x','super_classe_x','predicted_1','predicted_2']]
    test_intersection_df=test_intersection_df.rename(columns={'ana_name_x':'ana_name','cat_descrizione_x':'cat_descrizione','classe_x':'classe','super_classe_x':'super_classe'})
    
    test_intersection_df['intersection']=np.nan
    
    n_intersections=0
    
    for i in range(len(test_intersection_df)):
        if(test_intersection_df['predicted_1'][i]==test_intersection_df['predicted_2'][i]):
            test_intersection_df['intersection'][i]=test_intersection_df['predicted_1'][i]
            n_intersections+=1
        else:
            test_intersection_df['intersection'][i]=0
            
    test_intersection_rate=n_intersections/len(test_intersection_df)
    if(scoring=='accuracy'):
        test_accuracy=accuracy_score(test_intersection_df[test_intersection_df['intersection']!=0]['cat_descrizione'], test_intersection_df[test_intersection_df['intersection']!=0]['intersection'])
    elif(scoring=='balanced_accuracy'):
        test_accuracy=balanced_accuracy_score(test_intersection_df[test_intersection_df['intersection']!=0]['cat_descrizione'], test_intersection_df[test_intersection_df['intersection']!=0]['intersection'])

    print('percentage of train set intersections:{0}'.format(train_intersection_rate))
    print('intersections training score:{0}\n'.format(test_accuracy))

    print('percentage of test set intersections:{0}'.format(test_intersection_rate))
    print('intersections test score:{0}'.format(train_accuracy))

    return train_intersection_df,test_intersection_df

def pie_plot_classi(y_test):
    class_amount=data.category_count(y_test,'classe')
    class_amount['frequency']=class_amount['quantity']/len(y_test)
    class_amount.index=range(len(class_amount))
    
    class_amount_small= class_amount[class_amount['frequency']<0.05]
    class_amount_small['classe']= ' '
    class_amount = pd.concat([class_amount[class_amount['frequency']>=0.05], class_amount_small])

    fig1, ax1 = plt.subplots( figsize=(8,8))
    ax1.pie(class_amount['frequency'], labels=class_amount['classe'],   wedgeprops=dict(width=0.3, edgecolor='w'),autopct= my_autopct,
            startangle=90)
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

    plt.show()



