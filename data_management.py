# -*- coding: utf-8 -*-
"""
Created on Fri Dec 20 17:52:04 2019

@author: python
"""


'''Essential modules'''
import os 
import pandas as pd

"""Selects the directory containing the libraries"""
os.chdir('E:\python') #chiavetta
#os.chdir("C:\\Users\\Windows 10\\Desktop\\Stage\\python") #mio portatile
#os.chdir("\\\\salaanalisisrv\\ufficio_studi\\python\\Cluster\\barilaro\\python") #analisti1
#os.chdir('\\\\salaanalisisrv\\ufficio_studi\\python\\Cluster\\Giulio\\def') #fonte libreria

'''Modules for data and text'''
import data_management_def as data
import warnings
warnings.filterwarnings("ignore")





'''1 - DATA INITIALIZATION'''


'''1.1 - Loads the dataframe with all the data'''
df_mixed=pd.read_excel('dataframe_iniziale.xlsx',index=False)


'''1.2 - Selects anagraphic and features columns to be used'''

variables_return=data.seleziona_date_partenza(df_mixed,'2000-01-31') #taglio l'elenco delle colonne dei rendimenti da una certa data in poi 

variables_pol=['pol_testo','pol_finalita'] #selezione colonne con campi testuali

anagrafica=["ana_name","ana_ticker","cat_descrizione","super_classe","classe","gerarchia"] #seleziono colonne con campi anagrafici


'''1.3 - Divides initial dataframe into yields and documents dataframes'''

df_return = df_mixed[anagrafica+variables_return] #datafrme rendimenti
df_pol = df_mixed[anagrafica+variables_pol] #dataframe testi


'''1.4 - Selects hierarchy'''

gerarchia=1 #se inserisco gerarchia superiore a 1, seleziona anche le precedenti (livello minimo di gerarichia)

df_pol=data.seleziona_gerarchia(df_pol,gerarchia) #estraggo fondi di interesse

df_return=data.seleziona_gerarchia(df_return,gerarchia) #elimino dall'elenco di variabili anagrafica la colonna gerarchia

anagrafica=["ana_name","ana_ticker","cat_descrizione","super_classe","classe"] 


#########   END OF INPUT PREPARATION   ############


'''2 - DATA EXPLORATIVE ANALYSIS'''


'''2.1 - Show the yields via plot to see where the missing data is'''
data.show_nans(df_return,variables_return)


'''3.3 - Time series plot of yields'''
grouped=pd.DataFrame(df_mixed.groupby('super_classe')[variables_return].mean())
trasposta=grouped.transpose()
ax = trasposta.plot(figsize=(13,13),ylim=(-0.21,0.21),grid=True,fontsize=13)
ax.set_xlabel("Date")
ax.set_ylabel("Returns")



'''2.2 - Verify the % of non-missing data for each dataframe column''' 
percentuale_minima=0.5
data.no_NaN_rows(df_return[variables_return],percentuale_minima)


'''2.3 - Selects subset of initial dataframe from a defined date'''
variables_return=data.seleziona_date_partenza(df_return,'2016-01-31') #riduco il time frame di analisi a quello con abbastanza dati
df_return=df_return[anagrafica+variables_return]


'''2.4 - Selects only rows with a certain % of non-missing data'''
percentuale_minima=0.5
df_return = data.minimum_time_series_length(df_return,variables_return,percentuale_minima) #tolgo i fondi che hanno troppi pochi dati


'''2.5 - Histogram of the classes'''
label='classe' #definisco quale livello gerarchico del db da utilizzare (specificare nome del campo): categoria, classe o superclasse
data.class_histogram(df_pol,label)
data.class_histogram(df_return,label)


'''2.6 - Yields boxplot'''
data.returns_boxplot(df_return,variables_return,'2016-01-31','2016-12-31')
data.returns_boxplot(df_return,variables_return,'2017-01-31','2017-12-31')
data.returns_boxplot(df_return,variables_return,'2018-01-31','2018-12-31')
data.returns_boxplot(df_return,variables_return,'2017-12-31','2018-12-31')


'''2.7 - Removes data outside the interquantile interval (outlier removal)''' 
df_return=data.delete_out_of_IQR(df_return,variables_return,yes=False) #yes=True toglie gli outliers
#data.show_nans(df_return,variables_return)


'''2.8 - Heatmap for yield correlations by date'''
correlazione_minima=0.5 # fa vedere correlazioni con valore pari o superiore
#correlazione calcolata non tra cat ma tra diverse date
#data.heatmap(df_return,variables_return,correlazione_minima)


'''2.9 - Scatterplot on two columns'''
data.scatterplot_columns(df_return,'2016-06-30','2016-12-31') #scatterplot con date selezionate a piacere


'''2.10 - Pairplot of the yields'''
data.pairplot(df_return,variables_return,'no') #con no non printa, particolarmente pesante 


'''2.11 - Histograms, with bins = 20 '''
data.histograms_columns(df_return,variables_return,20) 


'''2. 12 - Language distribution of the documents'''
'''Execute before translation'''

df_pol=data.controllo_testi_stranieri(df_pol) 
df_pol['testo_intero']=df_pol['pol_testo']+df_pol['pol_finalita']+df_pol['ana_name'] 
variables_pol=['testo_intero']

'''2.13 - Selects only funds with italian documents or translated to italian'''
'''If modalità=traduci it adds the translated texts as well, if =solo italiani it only keeps the italian ones'''

modalità='traduci' 
file_traduzioni='fondi_tradotti_in_italiano.xlsx' #dataset with translated text

df_pol=data.traduci_testi_stranieri(df_pol,modalità,file_traduzioni,gerarchia)






'''3 - FINAL DATAFRAMES'''



'''3.1 - Words and yields dataframe'''
df_mixed=df_pol.merge(df_return,left_on=anagrafica,right_on=anagrafica)
variables_mixed=variables_return+variables_pol


'''3.2 - Optional: selects only classes with X funds'''
'''The smaller classes are put into a single one'''

keep=True #if true it puts the smaller classes into a closure ones 
x=2 #smallest allowed number of funds for each class
df_mixed=data.class_size_threshold_all_labels(df_mixed,x,keep)
#df_pol=data.class_size_threshold_all_labels(df_pol,x,keep)
#df_return=data.class_size_threshold_all_labels(df_return,x,keep) 

#data.class_histogram(df_return,label)
#data.class_histogram(df_pol,label)
#data.class_histogram(df_mixed,label)

