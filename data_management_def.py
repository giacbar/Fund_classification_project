# -*- coding: utf-8 -*-
"""
Created on Tue Aug  6 10:53:35 2019

@author: fida-stage1
"""
"""load the packages"""

import os 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import seaborn as sns
from langdetect import detect

'''############### FUNZIONI CARICAMENTO DATI SPECIFICHE ################'''

def import_excel(name,path):
    """imports the xlxs file
    inputs: -name of the xlxs file
             """
    owd=os.getcwd()                     
    #path=".\dati"
    os.chdir(path)
    xls=pd.ExcelFile(name)
    os.chdir(owd)
    return xls


def initiate_table_from_excel(source,path,sheets,columns,merge_on):
    """initiates the dataframe by importing the xlsx file, selecting target sheets, columns 
        and the column label used to merge the sheets
     inputs: -xlsx filename
            -list of sheets to select
            -list of columns to select 
            -column label to merge upon 
            output: -dataframe  """
    
    xlsx = import_excel(source,path)
    dflist=[0]*len(sheets)
    for i in range(len(sheets)):
        dflist[i]=pd.read_excel(xlsx,sheet_name=sheets[i])
        dflist[i]=dflist[i].filter(columns[i])
    start=dflist[0]    
    if (len(sheets)>1):
        for i in range(len(dflist)-1):
            df=pd.merge(start,dflist[i+1], left_on=merge_on,right_on=merge_on)
            start=df
            df=df[df["sol_elab"]=='2019-01-31']  
            df=df.drop(columns=["sol_elab"])
            df.index = range(len(df))
    else:
        df=pd.DataFrame(start)
    return df



def initiate_dataframe_from_multiple_csv(names,path):
    """initiates the dataframe by importing the csv files selected by names and merges them
    inputs: -csv filenames
    output: -dataframe indexed by ticker"""
    owd=os.getcwd()        
    os.chdir(path)
    dflist=[0]*len(names)
    for i in range(len(names)):
        dflist[i]=pd.read_csv(names[i],sep="|",decimal=".")
        start=dflist[0]
    for i in range(len(dflist)-1):
        importdata=pd.merge(start,dflist[i+1],how='outer',on='Unnamed: 0')
        #importdata.columns=importdata['Unnamed: 0']
        #importdata=importdata.drop(['Unnamed: 0'])
        start=importdata
        
    df=importdata
    df.index=df['Unnamed: 0']
    
    df.columns=df.columns.str.replace('_EUR_TR', '')
    df=df.transpose()
    df=df.drop('Unnamed: 0')
    #df = df.assign(ana_ticker=df.index.str.replace('_EUR_TR', ''))
    #df.index=df["ana_ticker"]
    #df=df.drop(['ana_ticker'],axis=1)
    os.chdir(owd)
    
    return df

def indicator_function(x):
    """indicator function for values greater than 1"""
    if(x > 0):
        indicator = 1
    else:
        indicator = 0
    return(indicator)

def split_dataframe(df,step):
    """splits the dataframe by column with a specific step, creating a list of dataframes"""
    indicator=indicator_function((len(df.columns)%step))
    dflist=[0]*((len(df.columns)//step)+indicator)
    start=0
    for i in range((len(df.columns)//step)+indicator):
        dflist[i]=pd.DataFrame()
        dflist[i]=df[df.columns[start:]]
        start=start+step
    return 

'''############ FUNZIONI OLD ##############'''
def remove_NaNs(dataframe):
     """remove NaN rows from dataframe"""
     dataframe=dataframe.dropna()
     dataframe.index = range(len(dataframe))
     return dataframe


def category_selection(wordsin,wordsout,dataframe,column):
    """subsets the dataframe selecting only the funds where the column 
    contains the wordsin and removes the rows containing wordsout
    input: -words in
           -words out
           -dataframe   """
    final=pd.DataFrame()
    for i in range(len(wordsin)):
        final=final.append(dataframe[dataframe[column].str.contains(wordsin[i])])
    for j in range(len(wordsout)):
        final=final[~final[column].str.contains(wordsout[j])]
    final.index = range(len(final))
    return final

'''############ FUNZIONI CONTEGGIO FONDI-CAT/CLASSI/SUP-CLASSI ###########'''

def category_count(df,category):
    """counts the number of funds in each category(class) in the dataframe
    INPUT: DataFrame, category= cat_descrizone, classe, super_classe as string
    OUTPUT: DataFrame con cat e numero fondi
    """
    count=Counter(df[category])
    lista1=[[0] for i in range(len(count))]
    lista2=[[0] for i in range(len(count))]
    for i in range(len(count)):
        lista1[i]=list(count.items())[i][1]
        lista2[i]=list(count.items())[i][0]
    ordine_categorie=pd.DataFrame(data={category:lista2,"quantity":lista1})
    #ordino per numerosità categoria
    ordine_categorie=ordine_categorie.sort_values(by="quantity",ascending=False) 
    return ordine_categorie

def class_size_threshold(df,label,X,keep):
    '''seleziono solo le categorie che hanno almeno X fondi,
        le categorie scartate le metto in una unica di chiusura
        input: dataframe
               label (categoria)
               X (threshold numerosità cateogrie)
               keep: se True fa confluire le categorie con bassa numerosità in 
                     una categoria nulla'''
    
    num_cat=category_count(df,label)
    num_cat=num_cat[num_cat['quantity']>=X]
    
    if(keep==True):
        for i in range(len(df)):
            if( np.any(num_cat[label].isin([df[label][i]]))):
                df[label][i]=df[label][i]
            else:
                df[label][i]='null_category'
    else:
         for i in range(len(df)):
            if( np.any(num_cat[label].isin([df[label][i]]))):
                df[label][i]=df[label][i]
            else:
                df=df.drop(i)
         df.index=range(len(df))
    return df

def class_size_threshold_all_labels(df,X,keep):
    '''seleziono solo le categorie che hanno almeno X fondi,
        le categorie scartate le metto in una unica di chiusura
        input: dataframe
               label (categoria)
               X (threshold numerosità cateogrie)
               keep: se True fa confluire le categorie con bassa numerosità in 
                     una categoria nulla'''
    
    num_cat=category_count(df,'cat_descrizione')
    num_cat=num_cat[num_cat['quantity']>=X]

    if(keep==True):
        for i in range(len(df)):
            if( np.any(num_cat['cat_descrizione'].isin([df['cat_descrizione'][i]]))):
                df['cat_descrizione'][i]=df['cat_descrizione'][i]
                df['classe'][i]=df['classe'][i]
                df['super_classe'][i]=df['super_classe'][i]
            else:
                df['super_classe'][i]='null_super_classe'
                df['classe'][i]='null_classe'
                df['cat_descrizione'][i]='null_categoria'
              
    else:
         for i in range(len(df)):
            if( np.any(num_cat['cat_descrizione'].isin([df['cat_descrizione'][i]]))):
                df['cat_descrizione'][i]=df['cat_descrizione'][i]
            else:
                df=df.drop(i)
         df.index=range(len(df))
    return df


'''############# GESTIONE DATI MANCANTI #############'''

def no_NaN_rows(df,x):
    """prints the columns who have at least x% non-nan values and prints the number
    of their non-nan rows
    -inputs: -dataframe
             -number of non-nan values desired"""
    for i in df.columns:
        if((len(df)-df[i].isna().sum())/len(df)>x):
            print('date: {0}, non missing values: {1:.2f}'.format(i,(len(df)-df[i].isna().sum())/len(df)))

def seleziona_date_partenza(df,data):
    
    variables=pd.DataFrame(df.columns,index=df.columns)
    variables=list(variables[data:].index)
    
    return variables #restituisce la lista di colonne dalla data specificata in poi

def minimum_time_series_length(df,variables,threshold):
    
    df=df.dropna(subset=variables,thresh=int(len(variables)*threshold))
    df.index=range(len(df))
    
    return df #restituisce df già pulito delle serie storiche troppo corte

def seleziona_gerarchia(df,ngerarchia):
    if(ngerarchia==1):
        df= df[(df["gerarchia"]==1)]
    if(ngerarchia==2):
        df= df[(df["gerarchia"]==1) | (df['gerarchia']==2)]
    if(ngerarchia==3):
        df= df[(df["gerarchia"]==1) | (df['gerarchia']==2)| (df['gerarchia']==3)]
    if(ngerarchia==0):
        df= df[(df["gerarchia"]==1) | (df['gerarchia']==2)| (df['gerarchia']==3)|(df['gerarchia']==0)]

    df.index=range(len(df))
    df=df.drop('gerarchia',axis=1)
    return df

'''######### GESTIONE OUTLIERS ##########'''

def delete_out_of_IQR(df,variables, yes):
    '''Elimina righe con elementi al di fuori dell'intervallo interquantile'''
    if (yes==True):
        Q1 = df[variables].quantile(0.25)
        Q3 = df[variables].quantile(0.75)
        IQR = Q3 - Q1
        df = df[~((df[variables] < (Q1 - 1.5 * IQR)) |(df[variables] > (Q3 + 1.5 * IQR))).any(axis=1)]
    return df

'''########### GRAFICI ###########'''

def class_histogram(df,label):
    
    cnt_pro = df[label].value_counts()
    plt.figure(figsize=(10,10))
    sns.barplot(cnt_pro.index, cnt_pro.values, alpha=0.8)
    plt.ylabel('Number of Occurrences', fontsize=15)
    plt.xlabel(label, fontsize=15)
    plt.xticks(rotation=90)
    plt.show();
    
def my_autopct(pct):
    return ('%.1f' % pct) if pct > 5 else ''

def pie_plot_lingue(dfpol):
    class_amount=category_count(dfpol,'lingua')
    class_amount['frequency']=class_amount['quantity']/len(dfpol)
    class_amount.index=range(len(class_amount))
    
    for i in class_amount.index:
        if (class_amount.iloc[i]['frequency']<0.05):
            class_amount.iloc[i]['lingua']=' '
    
    fig1, ax1 = plt.subplots( figsize=(8,8))
    ax1.pie(class_amount['frequency'], labels=class_amount['lingua'],   wedgeprops=dict(width=0.3, edgecolor='w'),autopct= my_autopct,
            startangle=90)
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

    plt.show()

    
def returns_boxplot(df,variables,startdate,enddate):
    plt.figure(figsize=(13,13))
    fig=sns.boxplot(data=df[df[variables].columns[df[variables].columns.get_loc(startdate):df[variables].columns.get_loc(enddate)]]).set(ylabel="daily yields")
    plt.xlabel("Date of recording")
    plt.ylabel("Yields")
    plt.show(fig)

def heatmap(df,variables,threshold):
    plt.figure(figsize=(25,25))
    c= df[variables].corr()
    sns.heatmap(c[(c >= 0.5) | (c <= -0.5)],cmap="RdBu",annot=True)

def scatterplot_columns(df,xcolumn,ycolumn):
    
    fig, ax = plt.subplots(figsize=(10,6))
    ax.scatter(df[xcolumn], df[ycolumn])
    ax.set_xlabel(xcolumn)
    ax.set_ylabel(ycolumn)
    plt.show()
    
def pairplot(df, variables, value):
    if(value=='yes'):
        sns.pairplot(df[variables].dropna())


def histograms_columns(df,variables,size):
    
    df[variables].hist(figsize=(size, size), bins=50, xlabelsize=8, ylabelsize=8);
    
def show_nans(df,variables):
    
    plt.figure(figsize=(13,13))
    fig=sns.heatmap(df[variables].isnull(), cbar=False)
    plt.show(fig)

'''########### GESTIONE TESTI ############'''



def controllo_testi_stranieri(dfpol):
    '''Plot istogramma con testi divisi per lingua'''
    lingue=[0]*len(dfpol)
    for i in range(len(dfpol)):
        lingue[i]=detect(dfpol['pol_testo'][i])
    dfpol['lingua']=lingue #aggiunge variabile lingua
    pie_plot_lingue(dfpol)
    #class_histogram(dfpol,'lingua')
    dfpol=dfpol.fillna(' ')
    
    return dfpol

def traduci_testi_stranieri(dfpol,value,file_traduzioni,gerarchia):
    
    testi_italiani=dfpol[dfpol['lingua']=='it']

    if(value=='traduci'):
        testi_stranieri_tradotti=pd.read_excel(file_traduzioni)
        
        if(gerarchia==1):
            testi_stranieri_tradotti=testi_stranieri_tradotti[(testi_stranieri_tradotti['gerarchia']==1)]
        elif(gerarchia==2):
            testi_stranieri_tradotti=testi_stranieri_tradotti[(testi_stranieri_tradotti['gerarchia']==1)|(testi_stranieri_tradotti['gerarchia']==2)]
        elif(gerarchia==3):
            testi_stranieri_tradotti=testi_stranieri_tradotti[(testi_stranieri_tradotti['gerarchia']==1)|(testi_stranieri_tradotti['gerarchia']==2)|(testi_stranieri_tradotti['gerarchia']==3)]

        testi_stranieri_tradotti=testi_stranieri_tradotti.drop('testo_intero',axis=1)
        testi_stranieri_tradotti=testi_stranieri_tradotti.rename(columns={"traduzione": "testo_intero", "B": "c"})
        testi_stranieri_tradotti['testo_intero']=testi_stranieri_tradotti['testo_intero']+testi_stranieri_tradotti['ana_name']
        dfpol=testi_italiani.append(testi_stranieri_tradotti)
        dfpol=dfpol.drop('Unnamed: 0',axis=1)
    elif(value=='solo italiani'):
        dfpol=testi_italiani
    dfpol.index=range(len(dfpol))
    dfpol=dfpol.drop(['pol_testo','pol_finalita','lingua'],axis=1)
    return dfpol

