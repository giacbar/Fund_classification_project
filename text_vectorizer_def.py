# -*- coding: utf-8 -*-

"""
Created on Mon Dec  9 09:01:33 2019

@author: python
"""

'''Modules for text'''
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectKBest

from sklearn.model_selection import StratifiedShuffleSplit
import nltk
from bs4 import BeautifulSoup
import re, unicodedata
from sklearn.model_selection import cross_val_score

# Modules for other languages
#from nltk import word_tokenize, sent_tokenize,bigrams
#from nltk.stem.snowball import ItalianStemmer
#from nltk.stem.lancaster import LancasterStemmer
#from nltk.stem.porter import PorterStemmer

from nltk.stem import WordNetLemmatizer
from nltk.stem.snowball import SnowballStemmer
from collections import Counter
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from googletrans import Translator
import matplotlib.pyplot as plt
import supervised_learning_def as supervised
import pandas as pd
import stop_words
from tqdm import tqdm
from sklearn import utils
from gensim.models.phrases import Phrases
from gensim.models.phrases import Phraser
from gensim.models.doc2vec import LabeledSentence
import numpy as np

def load_stopwords(language):
    '''Loads stopwords using the selected language'''
    from nltk.corpus import stopwords

    if(language=='ita'):
        stopwords_ita=open('stopwords.txt','r+',encoding="utf8")
        stopwords_ita=stopwords_ita.read()
        stopwords_ita=stopwords_ita.split()+stop_words.get_stop_words('italian')

    if(language=='eng'):
        stopwords_eng=stopwords.words('english')

    if(language=='ita+eng'):
        stopwords_eng=stopwords.words('english')
        stopwords_ita=open('stopwords.txt','r+',encoding="utf8")
        stopwords_ita=stopwords_ita.read()
        stopwords_ita=stopwords_ita.split()+stop_words.get_stop_words('italian')
        stopwords=stopwords_eng+stopwords_ita
    return stopwords #lista di string: stopwords

def prepare_data_for_vectorization(df,normaldf,label,testvalsize,testsize,shuffle_method):
    
    '''Needs a normalized text df
    split in 3 parts: training, validation and test'''
    
    normaldf2=pd.DataFrame(normaldf,index=df.index)
    df2=df.merge(normaldf2, left_on=df.index,right_on=normaldf2.index)
    df_to_split=df2[['ana_ticker',label,0]]
    
    X=df_to_split[0]
    y=df_to_split[label]
    
    if(shuffle_method=='stratified'): #tries to split the classes equally between the two parts
        
        X_train,X_test_val,y_train,y_test_val = supervised.train_test_split(X, y, test_size=testvalsize,random_state=12)


        sss = StratifiedShuffleSplit(n_splits=5, test_size=0.5, random_state=1)

        for val_index, test_index in sss.split(X_test_val, y_test_val):
            print("TRAIN:", val_index, "TEST:", test_index)
            X_val, X_test = X.iloc[val_index,],X.iloc[test_index,]
            y_val, y_test = y.iloc[val_index,], y.iloc[test_index,]
        

    elif(shuffle_method==None): #splits without considering the classes
        X_train,X_test_val,y_train,y_test_val = supervised.train_test_split(X, y, test_size=testvalsize,random_state=12)
        X_val, X_test, y_val, y_test = supervised.train_test_split(X_test_val, y_test_val, test_size=testsize,random_state=12)

    # print class number
    print('numero categorie nel train set:{0}'.format(y_train.nunique()))
    print('numero categorie nel validation set:{0}'.format(y_val.nunique()))
    print('numero categorie nel test set:{0}'.format(y_test.nunique()))

    
    return  X_train, y_train, X_val, y_val, X_test, y_test

def prepare_data_for_vectorization_crossval(df,normaldf,label,testsize,shuffle_method=None):
    '''needs a normalized text df
    split in 3 parts: training e test, validation is performed on the training set'''
    
    normaldf2=pd.DataFrame(normaldf,index=df.index)
    df2=df.merge(normaldf2, left_on=df.index,right_on=normaldf2.index)
    df_to_split=df2[['ana_ticker',label,0]]
    
    X=df_to_split[0]
    y=df_to_split[label]
    
    if(shuffle_method=='stratified'):#tries to split the classes equally between the two parts
        
        sss = StratifiedShuffleSplit(n_splits=5, test_size=testsize, random_state=1)

        for val_index, test_index in sss.split(X, y):
            
            X_train, X_test = X.iloc[val_index,],X.iloc[test_index,]
            y_train, y_test = y.iloc[val_index,], y.iloc[test_index,]
        

    elif(shuffle_method==None): #splits without considering the classes
        X_train,X_test,y_train,y_test = supervised.train_test_split(X, y, test_size=testsize,random_state=12)

    # print class number
    print('numero categorie nel train set:{0}'.format(y_train.nunique()))
    print('numero categorie nel test set:{0}\n'.format(y_test.nunique()))

    
    return  X_train, y_train, X_test, y_test

def vectorizer_test(vectorizer,classifier,n_features,stopwords,X_train,y_train,X_val,y_val):
    ''' Version without crossvalidation
    INPUT:
        vectorizer:  vectorizations function
        classifier: classification model
        n_features: list with intervals for iteration of number of features and accuracy test, with empty list it takes all without multiple iterations
        stopword= stopwords list
        Training and validation set
    OUTPUT:
        feature_result = accuracy results
        nfeatures_plot = feature_result in df for plot
        names =  features list
        '''
    feature_result = nfeature_accuracy_checker(vectorizer, classifier,n_features, stopwords,X_train, y_train, X_val, y_val) #richiama la funzione successiva
    nfeatures_plot = pd.DataFrame(feature_result,columns=['validation_score'])
    
    print('number of features: {0}\n'.format(len(vectorizer.get_feature_names())))
    
    return feature_result,nfeatures_plot,vectorizer.get_feature_names()

def nfeature_accuracy_checker(vectorizer, classifier,n_features, X_train, y_train, X_val, y_val):
    result = []

    if(len(n_features)<2):
        vectorizer.set_params( max_features=None)
        checker_pipeline = Pipeline([
            ('vectorizer', vectorizer),
            ('classifier', classifier)])
        print ("Validation score for all features")
        nfeature_accuracy = accuracy_summary(checker_pipeline, X_train, y_train, X_val, y_val)
        result.append(nfeature_accuracy)
    else:
        print('Selecting features using feature frequency')
        for n in n_features:
            vectorizer.set_params( max_features=n)
            checker_pipeline = Pipeline([
                    ('vectorizer', vectorizer),
                    ('classifier', classifier)])
            print ("Validation score for {} features".format(n))
            nfeature_accuracy = accuracy_summary(checker_pipeline, X_train, y_train, X_val, y_val)
            result.append((n,nfeature_accuracy))
    return result

def accuracy_summary(pipeline, X_train, y_train, X_test, y_test):
    
    sentiment_fit = pipeline.fit(X_train, y_train)
    y_pred = sentiment_fit.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print ("score: {0:.2f}%".format(accuracy*100))
    return accuracy

def vectorizer_test_crossval(vectorizer,classifier,n_features,X_train,y_train,scoring):
    ''' Versione con crossvalidation
    INPUT:
     	vectorizer:  vectorizations function
        classifier: classification model
        n_features: list with intervals for iteration of number of features and accuracy test, with empty list it takes all without multiple iterations
        stopword= stopwords list
        Training  set
    OUTPUT:
        feature_result = accuracy results
        nfeatures_plot = feature_result in df for plot
        names =  features list
        '''
		
    feature_result = nfeature_accuracy_checker_crossval(vectorizer, classifier,n_features, X_train, y_train,scoring)
    nfeatures_plot = pd.DataFrame(feature_result,columns=['validation_score'])
    vectorizer.fit(X_train,y_train)
    print('number of features: {0}\n'.format(len(vectorizer.get_feature_names())))
    return feature_result,nfeatures_plot,vectorizer.get_feature_names()

def nfeature_accuracy_checker_crossval(vectorizer, classifier,n_features, X_train, y_train,scoring):
    
    result = []

    if(len(n_features)<2):
        vectorizer.set_params( max_features=None)
        checker_pipeline = Pipeline([
            ('vectorizer', vectorizer),
            ('classifier', classifier)])
        print ("Validation score for all features\n")
        nfeature_accuracy = accuracy_summary_crossval(checker_pipeline, X_train, y_train,scoring)[1]
        result.append(nfeature_accuracy)
    else:
        print('Selecting features using feature frequency\n')
        for n in n_features:
            vectorizer.set_params( max_features=n)
            checker_pipeline = Pipeline([
                    ('vectorizer', vectorizer),
                    ('classifier', classifier)])
            print ("Validation for {} features".format(n))
            nfeature_accuracy = accuracy_summary_crossval(checker_pipeline, X_train, y_train,scoring=scoring)[1]
            result.append((n,nfeature_accuracy))
    return result

def accuracy_summary_crossval(pipeline, X_train, y_train,scoring):
    
    #sentiment_fit = pipeline.fit(X_train, y_train)
    accuracy=cross_val_score(pipeline, X_train, y_train, scoring=scoring,cv=5)
    accuracy_mean=accuracy.mean()
    #print('crossvalidation scores: {0}'.format(accuracy))
    print ("Mean cross-valiadtion score: {0:.2f}%\n".format(accuracy_mean*100))
    return accuracy,accuracy_mean

def chi2_feature_comparison(X_train,y_train,X_val,y_val,n_features,  vectorizer, classifier):
    '''Feature scoring with chi square: without crossvalidation
    INPUT:
     	vectorizer:  vectorization function
        classifier: classification model
        n_features: list with intervals for iteration of number of features and accuracy test, with empty list it takes all without multiple iterations
        stopword= stopwords list
        Training and validation set
    OUTPUT:
        ch2_result = chi square result
        chi2_loops = results in a list for plotting
    '''
    X_train_chi2 = vectorizer.fit_transform(X_train)
    X_val_chi2=vectorizer.transform(X_val)

    chi2score = chi2(X_train_chi2, y_train)[0]
    chi2score[np.isnan(chi2score)]=0

    '''top chi2 scores'''
    plot_top_values(vectorizer.get_feature_names(),chi2score)
    
    ch2_result = []
    chi2_loops=[]
    
    print('Selecting features using chi2 scores:\n')
    for n in n_features:
        ch2 = SelectKBest(chi2, k=n)
        
        X_train_chi2_selected = ch2.fit_transform(X_train_chi2, y_train)
        
        chi2_loops.append(ch2)
        
        X_val_chi2_selected = ch2.transform(X_val_chi2)
        clf = classifier
        clf.fit(X_train_chi2_selected, y_train)
        score = clf.score(X_val_chi2_selected, y_val)
        ch2_result.append(score)
        print('Validation score for {0} features: {1}\n'.format(n,score))
        
    return ch2_result, chi2_loops

def chi2_feature_comparison_crossval(X_train,y_train,n_features, vectorizer, classifier,scoring):
    '''Feature scoring with chi square: with crossvalidation
    INPUT:
     	vectorizer:  vectorization function
        classifier: classification model
        n_features: list with intervals for iteration of number of features and accuracy test, with empty list it takes all without multiple iterations
        stopword= stopwords list
        Training set
    OUTPUT:
        ch2_result = chi square result
        chi2_loops = results in a list for plotting
'''
    
    X_train_chi2 = vectorizer.fit_transform(X_train)

    chi2score = chi2(X_train_chi2, y_train)[0]
    chi2score[np.isnan(chi2score)]=0

    '''top chi2 scores'''
    plot_top_values(vectorizer.get_feature_names(),chi2score)
    
    ch2_result = []
    chi2_loops=[]
    print('Selecting features using chi2 scores:\n')
    for n in n_features:
        ch2 = SelectKBest(chi2, k=n)
        
        X_train_chi2_selected = ch2.fit_transform(X_train_chi2, y_train)
        chi2_loops.append(ch2)
        clf = classifier
        accuracy=cross_val_score(clf, X_train_chi2_selected, y_train, scoring=scoring,cv=5)
        accuracy_mean=accuracy.mean()
        
        ch2_result.append(accuracy_mean)
        
        print('Crossvalidation score for {0} features: {1}\n'.format(n,accuracy_mean))
        
    return ch2_result, chi2_loops

def plot_top_values(feature_names,values):
    
    plt.figure(figsize=(15,15))
    wscores = list(zip(feature_names, values))
    wvalues= sorted(wscores, key=lambda x:x[1])
    topvalues = list(zip(*wvalues[-100:]))
    x = range(len(topvalues[1]))
    labels = topvalues[0]
    plt.barh(x,topvalues[1], align='center', alpha=0.2)
    plt.plot(topvalues[1], x, '-o', markersize=5, alpha=0.8)
    plt.yticks(x, labels)
    plt.xlabel('$\chi^2$')
    plt.title('Top chi2 scores')

def plot_comparison(nfeatures,nfeatures_accuracy,ch2_result,features_range,scoring):

    plt.figure(figsize=(10,10))
    plt.plot(nfeatures, nfeatures_accuracy,label='term frequency',color='royalblue')
    plt.plot(features_range, ch2_result,label='chi2 score',linestyle=':', color='orangered')
    plt.title("vectorizer: features limited within term frequency VS term chi2 score")
    plt.xlabel("Number of features")
    plt.ylabel("Validation set [0] score".format(scoring))
    plt.legend()

def select_best_chi2_features_OLD(df,variables_pol,variables_return,anagrafica,label,vectorizer,vectorization_method, first_n_features_to_take):
    ''' Feature selection with chi-square, no separation between training and validation
    INPUT:
        DataFrame
        anagrafica: list of anagraphic columns
        label= field on which to perform the selection as string
        vectorizer: vectorization function (model)
        vectorization_method: count or tfidf, vectorization method
        first_n_features: number of features to select
        normalized_texts = normalized texts ready for analysis
    OUTPUT:
        DataFrame without non-signigficative variables
        variables: clean feature list 
    '''
    
    X=df[variables_pol]
    y=df[anagrafica]

    
    X_chi2 = vectorizer.fit_transform(X)
    chi2score = chi2(X_chi2, y[label])[0]
    chi2score[np.isnan(chi2score)]=0

    wscores = list(zip(vectorizer.get_feature_names(),chi2score))
    wvalues= sorted(wscores, key=lambda x:x[1])
    topvalues = list(zip(*wvalues[-first_n_features_to_take:]))


    normal_df,nomi=text_vectorizer(df[variables_pol],method=vectorization_method,start=0,end=-1)
    normal_df=pd.DataFrame(normal_df,index=df['ana_ticker'],columns=nomi)
    df=df.merge(normal_df,left_on='ana_ticker',right_on='ana_ticker')
    
    #df=df.drop('classe_y',axis=1) #ho trovato la parola 'classe' e quindi fa caos con la colonna chiamata 'classe'
    #df=df.rename(columns={'classe_x':'classe'})
    
    variables_pol=[x for x in topvalues[0]]
    df=df[anagrafica+variables_return+variables_pol]

    variables_mixed=list(df.columns[5:])

    return df,variables_mixed,variables_pol


def select_best_chi2_features(df,variables_pol,variables_return,anagrafica,vectorizer, label,first_n_features_to_take):
    
    X=df[variables_pol]
    y=df[anagrafica]
    
    X_chi2 = vectorizer.fit_transform(X).toarray()
    nomi=vectorizer.get_feature_names()
    df_pol=pd.DataFrame(X_chi2,columns=nomi)
    
    ch2 = SelectKBest(chi2, k=first_n_features_to_take)
        
    ch2.fit_transform(X_chi2, y[label])
    
    
    mask = ch2.get_support() #list of booleans
    new_features = [] # The list of your K best features
    for bool, feature in zip(mask, nomi):
       if bool:
           new_features.append(feature)
           
    df_pol=df_pol[new_features]
    variables_mixed=variables_return+new_features
    df=df.join(df_pol)
        
    return df,variables_mixed,new_features


def text_normalizer(texts,stopwords):
    '''selects kiid and uses bag of words method for word frequency distribution
    
    INPUT: texts list
           stopwords list
           vectorization method: count or tfidf
           start and end index
    OUTPUT: feature matrix (vectorized words)
            vectorized words list'''
    
    texts_list=[0]*len(texts)
        
    for i in range(len(texts)):
        #elenco strumenti di normalizzazione
        text=texts[i]
        text=fix_text(text)
        words=nltk.word_tokenize(text)
        words = remove_non_ascii(words)
        words = to_lowercase(words)
        words = remove_punctuation(words)
        #words = remove_stopwords(words,stopwords)
        #words = stem_words(words)
        #words=lemmatize_words(words)
        #words=remove_short_words(words)
        
        final_text =' '.join(words)
        final_text = "".join(filter(lambda x: not x.isdigit(), final_text))
        texts_list[i]=final_text
        
    return texts_list

def strip_html(text):
    '''removes html text'''
    soup = BeautifulSoup(text, "html.parser")
    
    return soup.get_text()

def remove_between_square_brackets(text):
    '''removes square brackets content'''
    return re.sub('\[[^]]*\]', '', text)

def denoise_text(text):
    text = strip_html(text)
    text = remove_between_square_brackets(text)
    return text

def remove_non_ascii(words):
    """Remove non-ASCII characters from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = unicodedata.normalize('NFKD', word).encode('ascii', 'ignore').decode('utf-8', 'ignore')
        new_words.append(new_word)
    return new_words

def to_lowercase(words):
    """Convert all characters to lowercase from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = word.lower()
        new_words.append(new_word)
    return new_words

def remove_punctuation(words):
    """Remove punctuation from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = re.sub(r'[^\w\s]', ' ', word)
        if new_word != '':
            new_words.append(new_word)
    return new_words


def remove_stopwords(words,stopwords):
    """Remove stop words from list of tokenized words"""
    new_words = []
    for word in words:
        if word not in stopwords:
            new_words.append(word)
        
            
    return new_words

def remove_short_words(words):
    """Remove short words from list of tokenized words"""
    new_words = []
    for word in words:
        if (len(word)>2):
            new_words.append(word)
                    
    return new_words

def stem_words(words):
    """Stems words in list of tokenized words"""
    #language and algorithm selection
    stemmer = SnowballStemmer('italian')
    #stemmer = SnowballStemmer('english')
    #stemmer = ItalianStemmer()
    #stemmer=LancasterStemmer()
    #stemmer=PorterStemmer()
    stems = []
    for word in words:
        stem = stemmer.stem(word)
        stems.append(stem)
    return stems
 
def lemmatize_words(words):
    """lemmatizes words in list of tokenized words"""
    lemmatizer = WordNetLemmatizer()
    lemmas = []
    for word in words:
        lemma = lemmatizer.lemmatize(word)
        lemmas.append(lemma)
    return lemmas

def fix_text(text):
    '''combination of previous functions for text handling'''
    text=strip_html(text)
    text=remove_between_square_brackets(text)
    text= denoise_text(text)
    
    return text

def google_translation(lista):
    '''
    input: list of all strings to translate
    '''
          
    lista.index=range(len(lista))
    for j in range(len(lista)):
        if(lista['testo_intero'][j]=='- -'):
            lista=lista.drop(j,axis=0)
    lista.index=range(len(lista))
        
    for j in range(len(lista)):
        lista['testo_intero'][j]=lista['testo_intero'][j].replace('&quot','')
            
    lista['traduzione']=0
    for j in range(len(lista)):
        try:
            translator= Translator()
            translated= translator.translate(lista['testo_intero'][j],dest='it')
            lista['traduzione'][j]=translated.text
        except:
            print('no')
    lista.to_csv('traduzioni.csv',sep=';')
        
    return lista

def word_counter(texts_list):
    ''' Counts how many times a word appears in the texts'''
    words_list=[]
    for i in range(len(texts_list)):
        text=texts_list[i]
        words=nltk.word_tokenize(text)
        words_list=words_list+words
    wordsfreq=Counter(words_list)
    return wordsfreq

def text_vectorizer(texts_list,method,start,end):
    '''selects kiid and uses bag of words method for word frequency distribution
    
    INPUT: texts list
           stopwords list
           vectorization method: count or tfidf
           start and end index
    OUTPUT: feature matrix (vectorized words)
            vectorized words list'''    
   
    if(method=='count'):
        vectorizer = CountVectorizer()
        vectorizer.fit(texts_list)
        nomi=vectorizer.get_feature_names()
        X=vectorizer.fit_transform(texts_list).toarray()
    elif(method=='tfidf'):
        vectorizer = TfidfVectorizer()
        vectorizer.fit(texts_list)
        nomi=vectorizer.get_feature_names()
        X = vectorizer.transform(texts_list).toarray()
    
    return X,nomi

'''############# DOC2VEC ##############'''

def labelize(texts,label):
    result = []
    prefix = label
    for i, t in zip(texts.index, texts):
        result.append(LabeledSentence(t.split(), [prefix + '_%s' % i]))
    return result

def labelize_bigrams(texts,label):
    #in text_vectorized_old -> altro codice dei bigrammi/trigrammi 
    
    tokenized_train = [t.split() for t in texts]
    phrases = Phrases(tokenized_train)
    bigram = Phraser(phrases)

    result = []
    prefix = label
    for i, t in zip(texts.index, texts):
        result.append(LabeledSentence(bigram[t.split()], [prefix + '_%s' % i]))
    return result



def get_vectors(model, corpus, size):
    vecs = np.zeros((len(corpus), size))
    n = 0
    for i in corpus.index:
        prefix = 'all_' + str(i)
        vecs[n] = model.docvecs[prefix]
        n += 1
    return vecs    



def doc2vec_test(model,vectorization,classifier, X_train,y_train,X_val,y_val,X_test,y_test):
    
    model.build_vocab(documents=[x for x in tqdm(vectorization)])
    for epoch in range(100):
        model.train(utils.shuffle([x for x in tqdm(vectorization)]), total_examples=len(vectorization), epochs=1)
        model.alpha -= 0.0005
        model.min_alpha = model.alpha
    train_vecs_dbow = get_vectors(model, X_train, model.vector_size)
    validation_vecs_dbow = get_vectors(model, X_val, model.vector_size)
    test_vecs_dbow = get_vectors(model, X_test,model.vector_size)
    classifier.fit(train_vecs_dbow, y_train)
    print('score for the validation set: {0}'.format(classifier.score(validation_vecs_dbow, y_val)))
    print('score for the test set: {0}'.format(classifier.score(test_vecs_dbow, y_test)))
    
    return model



def combining_doc2vec_models(model1,model2,classifier,X_train,y_train,X_val,y_val,X_test,y_test):
    
    train_vecs_combination = get_concat_vectors(model1,model2, X_train, model1.vector_size+model2.vector_size)
    validation_vecs_combination = get_concat_vectors(model1,model2, X_val, model1.vector_size+model2.vector_size)
    test_vecs_combination = get_concat_vectors(model2,model1, X_test, model1.vector_size+model2.vector_size)

    classifier = supervised.LogisticRegression()
    classifier.fit(train_vecs_combination, y_train)
    
    print('score for the validation set: {0}'.format(classifier.score(validation_vecs_combination, y_val)))
    print('score for the test set: {0}'.format(classifier.score(test_vecs_combination, y_test)))

def doc2vec_test_crossval(model,vectorization,classifier, X_train,y_train,alpha_drop,scoring):
    
    model.build_vocab(documents=[x for x in vectorization])
    for epoch in range(100):
        model.train(utils.shuffle([x for x in vectorization]), total_examples=len(vectorization), epochs=1)
        model.alpha -= alpha_drop
        model.min_alpha = model.alpha
        
    train_vecs_dbow = get_vectors(model, X_train, model.vector_size)
    
    accuracy=cross_val_score(classifier, train_vecs_dbow, y_train, scoring=scoring,cv=5)
    accuracy_mean=accuracy.mean()

    print('mean score for the cross-validation set: {0}\n'.format(accuracy_mean))
    
    return model


def doc2vec_get_vectors(model,df,variables_pol,stop_words,anagrafica,alpha_drop):
    
    '''Returns vectors generated by doc2vec on the entire dataframe, following the model used
    INPUT: model:  doc2vec model
           df:dataframe
           variables: feature list to be used lista di variabili da usare come features
           testo: textual variable for vectorization
           stopwords: stopwords list
           anagrafica: list of anagrapic variables
           alpha drop: variable for doc2vec optimized descent
           
    OUTPUT: dataframes and variables to be used as  features'''
    
    
    vectorization = labelize(df[variables_pol], 'all') 

    
    model.build_vocab(documents=[x for x in vectorization])
    for epoch in range(100):
        model.train(utils.shuffle([x for x in vectorization]), total_examples=len(vectorization), epochs=1)
        model.alpha -= alpha_drop
        model.min_alpha = model.alpha
        
    vecs = get_vectors(model, df[variables_pol], model.vector_size)
    
    vecs=pd.DataFrame(vecs)
    variables_pol=list(vecs.columns)
    print(vecs)
    df_output=df[anagrafica].join(vecs)
    
    
    return df_output,variables_pol


def combining_doc2vec_models_crossval(model1,model2,classifier,X_train,y_train,scoring):
    
    train_vecs_combination = get_concat_vectors(model1,model2, X_train, model1.vector_size+model2.vector_size)

    accuracy=cross_val_score(classifier, train_vecs_combination, y_train, scoring=scoring,cv=5)
    accuracy_mean=accuracy.mean()
    
    print('score for the  cross-validation set: {0}\n'.format(accuracy_mean))

def get_concat_vectors(model1,model2, corpus, size):
    vecs = np.zeros((len(corpus), size))
    n = 0
    for i in corpus.index:
        prefix = 'all_' + str(i)
        vecs[n] = np.append(model1.docvecs[prefix],model2.docvecs[prefix])
        n += 1
    return vecs

