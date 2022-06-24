# -*- coding: utf-8 -*-
"""
Created on Fri Jan 10 09:48:06 2020

@author: python
"""
#Takes as input df_pol from data_management.py
#os.chdir("\\\\salaanalisisrv\\ufficio_studi\\python\\Cluster\\barilaro\\python") #analisti1

import os 
import pandas as pd
import numpy as np
from gensim.models import Doc2Vec

import supervised_learning_def as supervised
import text_vectorizer_def as text


import warnings
warnings.filterwarnings("ignore")


'''Defines stopwords (words to be excluded from the analysis)'''

#import nltk
#nltk.download('wordnet') #farlo una volta sola che viene scaricato tutto nel sistema (FORSE INUTILE PERCHè POTREBBE ESSERE QULLO IN ITALIANO)
#nltk.download('stopwords') #farlo una volta sola che viene scaricato tutto nel sistema
#nltk.download('punkt')


'''stopwords in english and italian'''
stopwords=text.load_stopwords('ita+eng') 


'''selects which variables to be used'''
df=df_mixed.copy()
variables_pol='testo_intero'
scoring='accuracy'
print('using {0} score'.format(scoring))

df[variables_pol]=df['testo_intero'] #selects the text
label='cat_descrizione' #defines the hierarchical class to be used

'''textual features normalization'''
normal=text.text_normalizer(df[variables_pol],stopwords) 
df[variables_pol]=normal

'''splits train and test set'''
train_size=0.9
X_train,y_train,X_test,y_test=supervised.prepare_inputs(df,anagrafica,variables_pol,label,train_size,'stratified',scaling=False,artificial=False)



'''1 - BAG OF WORDS: comparison between count vectorizer and tfidf vectorizer'''

ngram_range=(1,1) # bigram (1,2) and trigram (1,3) paramater, unigram (11) is the default 

'''1.1 - Count vectorizer'''

vectorizer = text.CountVectorizer(ngram_range=ngram_range,analyzer='word') #defines the vectorized type
classifier = supervised.LogisticRegression() #defines the classifier to be used 
n_features = ['none'] #defines the range of features to be considered ('none' uses all of them)

print('vectorizer: count\n')
feature_result,nfeatures_plot,names=text.vectorizer_test_crossval(vectorizer,classifier,n_features,X_train,y_train[label],scoring=scoring)


'''1.2 - tfidf vectorizer'''

vectorizer = text.TfidfVectorizer(ngram_range=ngram_range,analyzer='word')
classifier = supervised.LogisticRegression()
n_features = ['none']

print('vectorizer: TFIDF\n')
feature_result,nfeatures_plot,names=text.vectorizer_test_crossval(vectorizer,classifier,n_features,X_train,y_train[label],scoring=scoring)

'''1.3 - reducing the number of features for text classification'''

''' Compares data to see if reduction is needed by looking at how discriminant each word is'''

n_features = np.arange(100,1000,100) #intervalli di numero di features (ordinate per chi2 score o frequenza) da utilizzare
vectorizer = text.CountVectorizer(ngram_range=ngram_range,analyzer='word') #selezionando il modello che presenta un accuratezza maggiore

'''limiting the number of features using vectorizer max features (the frequency of each term)'''

feature_result = text.nfeature_accuracy_checker_crossval(vectorizer, classifier,n_features,X_train, y_train[label],scoring=scoring) #ordino per frequenza delle parole
nfeatures_plot = pd.DataFrame(feature_result,columns=['nfeatures','validation_accuracy'])

'''limiting the number of features using chi2'''

#ch2_result,topvalues,chi2_loops=text.chi2_feature_comparison_crossval(X_train,y_train,n_features,vectorizer, classifier) -> OLD: topvalues rimosso da funzione
ch2_result,chi2_loops=text.chi2_feature_comparison_crossval(X_train,y_train[label],n_features, vectorizer, classifier,scoring=scoring) #ordino per chi2 

'''comparison plot of limiting the features in both ways'''
text.plot_comparison(nfeatures_plot.nfeatures,nfeatures_plot.validation_accuracy,ch2_result,n_features,scoring )






'''2 - Doc2vec evaluation'''

vectorization = text.labelize(X_train, 'all')  #vettorizzazione, input necessario per doc2vec


'''2.2 - Distributed Bag of Words, DBOW'''

print('DOC2VEC\n')

'''Define hte classifier and doc2vec parameters'''
classifier= supervised.LogisticRegression()

alpha_drop=0.0005
model=Doc2Vec(dm=0, window=5,vector_size=100,negative=15, min_count=1, alpha=0.05, min_alpha=0.01,seed=1)
print('Distributed Bag of Words')
model=text.doc2vec_test_crossval(model,vectorization,classifier, X_train,y_train[label],alpha_drop,scoring=scoring)

model.save('d2v_model_ug_dbow.doc2vec')
model = Doc2Vec.load('d2v_model_ug_dbow.doc2vec')
model.delete_temporary_training_data(keep_doctags_vectors=True, keep_inference=True)




'''2.3 - Distributed Memory (concatenated) DMC'''

alpha_drop=0.0005
model = Doc2Vec(dm=1, dm_concat=1,window=5, size=100, negative=15, min_count=1,  alpha=0.05, min_alpha=0.01,seed=1)
print('Distributed Memory Concatenated')
model=text.doc2vec_test_crossval(model,vectorization,classifier, X_train,y_train[label],alpha_drop,scoring=scoring)

model.save('d2v_model_ug_dmc.doc2vec')
model = Doc2Vec.load('d2v_model_ug_dmc.doc2vec')
model.delete_temporary_training_data(keep_doctags_vectors=True, keep_inference=True)


'''2.4 - Distributed Memory (mean) DMM'''

alpha_drop=0.0005
model = Doc2Vec(dm=1, dm_mean=1,window=5, size=100, negative=15, min_count=1, alpha=0.05, min_alpha=0.01)
print('Distributed Memory Mean')
model=text.doc2vec_test_crossval(model,vectorization,classifier, X_train,y_train[label],alpha_drop,scoring=scoring)

model.save('d2v_model_ug_dmm.doc2vec')
model = Doc2Vec.load('d2v_model_ug_dmm.doc2vec')
model.delete_temporary_training_data(keep_doctags_vectors=True, keep_inference=True)


'''2.5 - DBOW + DMC'''

model_ug_dbow= Doc2Vec.load('d2v_model_ug_dbow.doc2vec')
model_ug_dmc= Doc2Vec.load('d2v_model_ug_dmc.doc2vec')
print('DBOW + DMC')
text.combining_doc2vec_models_crossval(model_ug_dbow,model_ug_dmc,classifier,X_train,y_train[label],scoring=scoring)


'''2.6 - DBOW + DMM'''
model_ug_dbow= Doc2Vec.load('d2v_model_ug_dbow.doc2vec')
model_ug_dmm= Doc2Vec.load('d2v_model_ug_dmm.doc2vec')
print('DBOW + DMM')
text.combining_doc2vec_models_crossval(model_ug_dbow,model_ug_dmm,classifier,X_train,y_train[label],scoring=scoring)




'''
#3 - valutazione DOC2VEC: bigrams

print('BIGRAMS\n')

vectorization = text.labelize_bigrams(pd.concat([X_train]), 'all')
classifier= supervised.LogisticRegression()

#3.1 - Distributed Bag of Words, DBOW


alpha_drop=0.0005
model=Doc2Vec(dm=0,window=5, vector_size=100,negative=15, min_count=1, alpha=0.05, min_alpha=0.01,seed=1)
print('Distributed Bag of Words')
model=text.doc2vec_test_crossval(model,vectorization,classifier, X_train,y_train[label],alpha_drop,scoring=scoring)

model.save('d2v_model_ug_dbow.doc2vec')
model = Doc2Vec.load('d2v_model_ug_dbow.doc2vec')
model.delete_temporary_training_data(keep_doctags_vectors=True, keep_inference=True)




#3.2 - Distributed Memory (concatenated) DMC

alpha_drop=0.0005
model = Doc2Vec(dm=1,window=5, dm_concat=1, size=100, negative=15, min_count=1,  alpha=0.05, min_alpha=0.01,seed=1)
print('Distributed Memory Concatenated')
model=text.doc2vec_test_crossval(model,vectorization,classifier, X_train,y_train[label],alpha_drop,scoring=scoring)

model.save('d2v_model_ug_dmc.doc2vec')
model = Doc2Vec.load('d2v_model_ug_dmc.doc2vec')
model.delete_temporary_training_data(keep_doctags_vectors=True, keep_inference=True)


#3.3 - Distributed Memory (mean) DMM

alpha_drop=0.0005
model = Doc2Vec(dm=1, dm_mean=1, window=5,size=100, negative=15, min_count=1, alpha=0.05, min_alpha=0.01)
print('Distributed Memory Mean')
model=text.doc2vec_test_crossval(model,vectorization,classifier, X_train,y_train[label],alpha_drop,scoring=scoring)

model.save('d2v_model_ug_dmm.doc2vec')
model = Doc2Vec.load('d2v_model_ug_dmm.doc2vec')
model.delete_temporary_training_data(keep_doctags_vectors=True, keep_inference=True)


#3.4 - DBOW + DMC

model_ug_dbow= Doc2Vec.load('d2v_model_ug_dbow.doc2vec')
model_ug_dmc= Doc2Vec.load('d2v_model_ug_dmc.doc2vec')
print('DBOW + DMC')
text.combining_doc2vec_models_crossval(model_ug_dbow,model_ug_dmc,classifier,X_train,y_train[label],scoring=scoring)


#3.5 - DBOW + DMM
model_ug_dbow= Doc2Vec.load('d2v_model_ug_dbow.doc2vec')
model_ug_dmm= Doc2Vec.load('d2v_model_ug_dmm.doc2vec')
print('DBOW + DMM')
text.combining_doc2vec_models_crossval(model_ug_dbow,model_ug_dmm,classifier,X_train,y_train[label],scoring=scoring)

'''




'''4 - Application of the methodology: selects the best performing vectorization'''
'''4.1 - Picks the name of the dataframe to be generated and the variables on which to make the classification'''


'''Choose which variables to use'''


'''alternative:
vectorizer = text.TfidfVectorizer(ngram_range=ngram_range,analyzer='word')
vectorizer = text.CountVectorizer(ngram_range=ngram_range,analyzer='word') # Selecting the model with greater accuracy
'''

ngram_range=(1,1)
n_features_to_select=100
vectorizer = text.CountVectorizer(ngram_range=ngram_range,analyzer='word') # Selecting the model with greater accuracy

#df_mixed,variables_mixed,variables_pol=text.select_best_chi2_features(df,variables_pol,variables_return,anagrafica,label,vectorizer,'count',n_features_to_select)

df_mixed,variables_mixed,variables_pol=text.select_best_chi2_features(df,variables_pol,variables_return,anagrafica,vectorizer,label,n_features_to_select)


'''5 - Choice between doc2vec'''

alpha_drop=0.0005
model=Doc2Vec(dm=0, vector_size=100,negative=15, min_count=1, alpha=0.05, min_alpha=0.01,seed=1)
#model = Doc2Vec(dm=1, dm_mean=1, size=100, negative=15, min_count=1, alpha=0.05, min_alpha=0.01)

'''5.1 - Vector selection'''

#df_mixed,variables_pol=text.doc2vec_get_vectors(model,df,variables_pol,variables_return,anagrafica,alpha_drop)








