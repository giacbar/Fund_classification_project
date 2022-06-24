# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 13:31:06 2019

@author: python
"""
import pandas as pd
import data_management_def as data
import text_vectorizer_def as text
import supervised_learning_def as supervised
import numpy as np

'''Stopwords definition'''
#nltk.download('wordnet') perform only once
#nltk.download('stopwords') perform only once



'''stopwords in english'''


stopwords=text.load_stopwords('ita+eng')


'''1 - BAG OF WORDS'''

label='cat_descrizione'
normal=text.text_normalizer(df_pol['testo_intero'],stopwords)

X_train, y_train, X_val, y_val, X_test, y_test =  text.prepare_data_for_vectorization(df_pol,normal,'cat_descrizione',0.1,0.5)

'''1.1 - Count vectorizer'''
vectorizer = text.CountVectorizer()
classifier = supervised.LogisticRegression()
n_features = ['none']

feature_result,nfeatures_plot,names=text.vectorizer_test(vectorizer,classifier,n_features,stopwords,X_train,y_train,X_val,y_val)
    
'''1.2 - tfidf vectorizer'''
vectorizer = text.TfidfVectorizer()
classifier = supervised.LogisticRegression()
n_features = ['none']

feature_result,nfeatures_plot,names=text.vectorizer_test(vectorizer,classifier,n_features,stopwords,X_train,y_train,X_val,y_val)



'''1.3 - reducing the number of features for text classification'''

import warnings
warnings.filterwarnings("ignore")

n_features = np.arange(100,2300,100)


'''limiting the number of features using vectorizer max features (the frequency of each term)'''

feature_result = text.nfeature_accuracy_checker(vectorizer, classifier,n_features, stopwords,X_train, y_train, X_val, y_val)
nfeatures_plot = pd.DataFrame(feature_result,columns=['nfeatures','validation_accuracy'])

'''limiting the number of features using chi2'''

ch2_result=text.chi2_feature_comparison(X_train,y_train,X_val,y_val,n_features,vectorizer)

'''comparison plot of limiting the features in both ways'''
text.plot_comparison(nfeatures_plot.nfeatures,nfeatures_plot.validation_accuracy,ch2_result)







'''doc2vec'''

from tqdm import tqdm
from gensim.models import Doc2Vec
from sklearn import utils

label='cat_descrizione'
normal=text.text_normalizer(df_pol['testo_intero'],stopwords)

normal2=pd.DataFrame(normal,index=df_pol.index)
df_pol2=df_pol.merge(normal2, left_on=df_pol.index,right_on=normal2.index)
testdf=df_pol2[['ana_ticker','cat_descrizione',0]]
X=testdf[0]
y=testdf['cat_descrizione']

X_train,X_test_val,y_train,y_test_val = supervised.train_test_split(X, y, test_size=.1,random_state=12)
X_val, X_test, y_val, y_test = supervised.train_test_split(X_test_val, y_test_val, test_size=.5,random_state=12)



all_x_w2v = text.labelize(normal2[0], 'all')
del normal2


'''DBOW'''
model_ug_dbow = Doc2Vec(dm=0, size=100,alpha=0.0545,negative=5,seed=1)

model_ug_dbow = Doc2Vec(dm=0, size=100,negative=5, min_count=2,  alpha=0.065, min_alpha=0.065)

model_ug_dbow.build_vocab(documents=[x for x in tqdm(all_x_w2v)])

for epoch in range(40):
    model_ug_dbow.train(utils.shuffle([x for x in tqdm(all_x_w2v)]), total_examples=len(all_x_w2v), epochs=1)
    model_ug_dbow.alpha -= 0.002
    model_ug_dbow.min_alpha = model_ug_dbow.alpha
    
train_vecs_dbow = text.get_vectors(model_ug_dbow, X_train, 100)
validation_vecs_dbow = text.get_vectors(model_ug_dbow, X_val, 100)
test_vecs_dbow = text.get_vectors(model_ug_dbow, X_test, 100)

clf= supervised.LogisticRegression()

clf.fit(train_vecs_dbow, y_train)
clf.score(validation_vecs_dbow, y_val)
clf.score(test_vecs_dbow, y_test)

model_ug_dbow.save('d2v_model_ug_dbow.doc2vec')
model_ug_dbow = Doc2Vec.load('d2v_model_ug_dbow.doc2vec')
model_ug_dbow.delete_temporary_training_data(keep_doctags_vectors=True, keep_inference=True)



'''Distributed Memory (concatenated)'''

model_ug_dmc = Doc2Vec(dm=1, dm_concat=1, size=100, window=2, negative=5, min_count=2,  alpha=0.065, min_alpha=0.065)
model_ug_dmc.build_vocab([x for x in tqdm(all_x_w2v)])

for epoch in range(30):
    model_ug_dmc.train(utils.shuffle([x for x in tqdm(all_x_w2v)]), total_examples=len(all_x_w2v), epochs=1)
    model_ug_dmc.alpha -= 0.002
    model_ug_dmc.min_alpha = model_ug_dmc.alpha
    
#model_ug_dmc = Doc2Vec.load('d2v_model_ug_dmc.doc2vec')
    
train_vecs_dmc = text.get_vectors(model_ug_dmc, X_train, 100)
validation_vecs_dmc = text.get_vectors(model_ug_dmc, X_val, 100)
test_vecs_dmc = text.get_vectors(model_ug_dmc, X_test, 100)

clf = supervised.LogisticRegression()
clf.fit(train_vecs_dmc, y_train)
clf.score(validation_vecs_dmc, y_val)
clf.score(test_vecs_dmc, y_test)


'''Distributed Memory (mean)'''

model_ug_dmm = Doc2Vec(dm=1, dm_mean=1, size=100, window=4, negative=5, min_count=2, alpha=0.065, min_alpha=0.065)
model_ug_dmm.build_vocab([x for x in tqdm(all_x_w2v)])

for epoch in range(30):
    model_ug_dmm.train(utils.shuffle([x for x in tqdm(all_x_w2v)]), total_examples=len(all_x_w2v), epochs=1)
    model_ug_dmm.alpha -= 0.002
    model_ug_dmm.min_alpha = model_ug_dmm.alpha

train_vecs_dmm = text.get_vectors(model_ug_dmm, X_train, 100)
validation_vecs_dmm = text.get_vectors(model_ug_dmm, X_val, 100)
test_vecs_dmm = text.get_vectors(model_ug_dmm, X_test, 100)

clf = supervised.LogisticRegression()
clf.fit(train_vecs_dmc, y_train)
clf.score(validation_vecs_dmc, y_val)
clf.score(test_vecs_dmc, y_test)




model_ug_dmm.save('d2v_model_ug_dmm.doc2vec')
model_ug_dmm = Doc2Vec.load('d2v_model_ug_dmm.doc2vec')
model_ug_dmm.delete_temporary_training_data(keep_doctags_vectors=True, keep_inference=True)



'''DBOW + DMC'''

train_vecs_dbow_dmc = text.get_concat_vectors(model_ug_dbow,model_ug_dmc, X_train, 200)
validation_vecs_dbow_dmc = text.get_concat_vectors(model_ug_dbow,model_ug_dmc, X_val, 200)
test_vecs_dbow_dmc = text.get_concat_vectors(model_ug_dbow,model_ug_dmc, X_test, 200)

clf = supervised.LogisticRegression()
clf.fit(train_vecs_dbow_dmc, y_train)
clf.score(validation_vecs_dbow_dmc, y_val)
clf.score(test_vecs_dbow_dmc, y_test)

'''DBOW + DMM'''

train_vecs_dbow_dmm = text.get_concat_vectors(model_ug_dbow,model_ug_dmm, X_train, 200)
validation_vecs_dbow_dmm = text.get_concat_vectors(model_ug_dbow,model_ug_dmm, X_val, 200)
test_vecs_dbow_dmm = text.get_concat_vectors(model_ug_dbow,model_ug_dmm, X_test, 200)

clf = supervised.LogisticRegression()
clf.fit(train_vecs_dbow_dmm, y_train)
clf.score(validation_vecs_dbow_dmm, y_val)
clf.score(test_vecs_dbow_dmm, y_test)


'''BIGRAMS'''

from gensim.models.phrases import Phrases
from gensim.models.phrases import Phraser

tokenized_train = [t.split() for t in X_train]

phrases = Phrases(tokenized_train)
bigram = Phraser(phrases)




'''DBOW bigram'''

all_x_w2v_bg = text.labelize_bg(normal[0], 'all')

model_bg_dbow = Doc2Vec(dm=0, size=100, negative=5, min_count=2, alpha=0.065, min_alpha=0.065)
model_bg_dbow.build_vocab([x for x in tqdm(all_x_w2v_bg)])

for epoch in range(30):
    model_bg_dbow.train(utils.shuffle([x for x in tqdm(all_x_w2v_bg)]), total_examples=len(all_x_w2v_bg), epochs=1)
    model_bg_dbow.alpha -= 0.002
    model_bg_dbow.min_alpha = model_bg_dbow.alpha
    
train_vecs_dbow_bg =  text.get_vectors(model_bg_dbow, X_train, 100)
validation_vecs_dbow_bg =  text.get_vectors(model_bg_dbow, X_val, 100)    
    
clf = supervised.LogisticRegression()
clf.fit(train_vecs_dbow_bg, y_train)    
    
clf.score(validation_vecs_dbow_bg, y_val)    
    
'''DMC bigram'''    
    
model_bg_dmc = Doc2Vec(dm=1, dm_concat=1, size=100, window=2, negative=5, min_count=2, alpha=0.065, min_alpha=0.065)
model_bg_dmc.build_vocab([x for x in tqdm(all_x_w2v_bg)])    
    
for epoch in range(30):
    model_bg_dmc.train(utils.shuffle([x for x in tqdm(all_x_w2v_bg)]), total_examples=len(all_x_w2v_bg), epochs=1)
    model_bg_dmc.alpha -= 0.002
    model_bg_dmc.min_alpha = model_bg_dmc.alpha    
    
train_vecs_dmc_bg =  text.get_vectors(model_bg_dmc, X_train, 100)
validation_vecs_dmc_bg =  text.get_vectors(model_bg_dmc, X_val, 100)    
    
clf = supervised.LogisticRegression()
clf.fit(train_vecs_dmc_bg, y_train)    
clf.score(validation_vecs_dmc_bg, y_val)    


model_bg_dmc.save('d2v_model_bg_dmc.doc2vec')
model_bg_dmc = Doc2Vec.load('d2v_model_bg_dmc.doc2vec')
model_bg_dmc.delete_temporary_training_data(keep_doctags_vectors=True, keep_inference=True)


'''DMM bigram'''

model_bg_dmm = Doc2Vec(dm=1, dm_mean=1, size=100, window=4, negative=5, min_count=2, alpha=0.065, min_alpha=0.065)
model_bg_dmm.build_vocab([x for x in tqdm(all_x_w2v_bg)])
for epoch in range(30):
    model_bg_dmm.train(utils.shuffle([x for x in tqdm(all_x_w2v_bg)]), total_examples=len(all_x_w2v_bg), epochs=1)
    model_bg_dmm.alpha -= 0.002
    model_bg_dmm.min_alpha = model_bg_dmm.alpha
    
train_vecs_dmm_bg = text.get_vectors(model_bg_dmm, X_train, 100)
validation_vecs_dmm_bg =  text.get_vectors(model_bg_dmm, X_val, 100)
clf = supervised.LogisticRegression()
clf.fit(train_vecs_dmm_bg, y_train)

clf.score(validation_vecs_dmm_bg, y_val)


model_bg_dmm.save('d2v_model_bg_dmm.doc2vec')
model_bg_dmm = Doc2Vec.load('d2v_model_bg_dmm.doc2vec')
model_bg_dmm.delete_temporary_training_data(keep_doctags_vectors=True, keep_inference=True)









'''applies final text vectorization'''
X,nomi=text.text_vectorizer(normal,method='count',start=0,end=-1)


'''creates dataframe with numerical vectors from the texts'''
X=pd.DataFrame(X,index=dfpol['ana_ticker'],columns=nomi)


