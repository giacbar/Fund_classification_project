# -*- coding: utf-8 -*-
"""
Created on Fri Dec 20 17:52:04 2019

@author: python
"""


'''essential modules'''
import os 
import pandas as pd

"""seleziono la directory desiderata per caricare le librerie"""
os.chdir('E:\python') #chiavetta
#os.chdir("C:\\Users\\Windows 10\\Desktop\\Stage\\python") #mio portatile
#os.chdir("\\\\salaanalisisrv\\ufficio_studi\\python\\Cluster\\barilaro\\python") #analisti1
#os.chdir('\\\\salaanalisisrv\\ufficio_studi\\python\\Cluster\\Giulio\\def') #fonte libreria

'''modules for data and text'''
import data_management_def as data
import warnings
warnings.filterwarnings("ignore")





'''1 - INIZIALIZZAZIONE DEI DATI'''


'''1.1 - Carico il dataframe con tutti i dati'''
df_mixed=pd.read_excel('dataframe_iniziale.xlsx',index=False)


'''1.2 -indico le colonne di anagrafica  e quelle delle variabili (features) da utilizzare'''

variables_return=data.seleziona_date_partenza(df_mixed,'2000-01-31') #taglio l'elenco delle colonne dei rendimenti da una certa data in poi 

variables_pol=['pol_testo','pol_finalita'] #selezione colonne con campi testuali

anagrafica=["ana_name","ana_ticker","cat_descrizione","super_classe","classe","gerarchia"] #seleziono colonne con campi anagrafici


'''1.3 - Divido il dataframe in quello dei rendimenti e quello dei documenti'''

df_return = df_mixed[anagrafica+variables_return] #datafrme rendimenti
df_pol = df_mixed[anagrafica+variables_pol] #dataframe testi


'''1.4 - Seleziono gerarchia'''

gerarchia=1 #se inserisco gerarchia superiore a 1, seleziona anche le precedenti (livello minimo di gerarichia)

df_pol=data.seleziona_gerarchia(df_pol,gerarchia) #estraggo fondi di interesse

df_return=data.seleziona_gerarchia(df_return,gerarchia) #elimino dall'elenco di variabili anagrafica la colonna gerarchia

anagrafica=["ana_name","ana_ticker","cat_descrizione","super_classe","classe"] 


#########   FINE PREPARAZIONE INPUT   ############


'''2 - ANALISI ESPLORATIVA DEI DATI'''


'''2.1 - Visualizza graficamente i rendimenti per capire dove mancano i dati'''
data.show_nans(df_return,variables_return)


'''3.3 - plot della serie storica dei rendimenti, medie di super_classe'''
grouped=pd.DataFrame(df_mixed.groupby('super_classe')[variables_return].mean())
trasposta=grouped.transpose()
ax = trasposta.plot(figsize=(13,13),ylim=(-0.21,0.21),grid=True,fontsize=13)
ax.set_xlabel("Date")
ax.set_ylabel("Returns")



'''2.2 - Verifica quanto è la percentuale di dati non mancanti per ogni colonna del dataframe''' 
percentuale_minima=0.5
data.no_NaN_rows(df_return[variables_return],percentuale_minima)


'''2.3 - Seleziona sottoinsieme del dataframe iniziale a partire da una data (riduco colonne)'''
variables_return=data.seleziona_date_partenza(df_return,'2016-01-31') #riduco il time frame di analisi a quello con abbastanza dati
df_return=df_return[anagrafica+variables_return]


'''2.4 - Seleziono solo le righe che hanno almneno un certa percentuale di dati non mancanti'''
percentuale_minima=0.5
df_return = data.minimum_time_series_length(df_return,variables_return,percentuale_minima) #tolgo i fondi che hanno troppi pochi dati


'''2.5 - Istogramma della numerosità delle categorie nei dataframe'''
label='classe' #definisco quale livello gerarchico del db da utilizzare (specificare nome del campo): categoria, classe o superclasse
data.class_histogram(df_pol,label)
data.class_histogram(df_return,label)


'''2.6 - Boxplot per rendimenti'''
data.returns_boxplot(df_return,variables_return,'2016-01-31','2016-12-31')
data.returns_boxplot(df_return,variables_return,'2017-01-31','2017-12-31')
data.returns_boxplot(df_return,variables_return,'2018-01-31','2018-12-31')
data.returns_boxplot(df_return,variables_return,'2017-12-31','2018-12-31')


'''2.7 - Elimino i dati esterni all'intervallo interquantile (per togliere outliers)''' 
df_return=data.delete_out_of_IQR(df_return,variables_return,yes=False) #yes=True toglie gli outliers
#data.show_nans(df_return,variables_return)


'''2.8 - Heatmap che evidenzia le correlazioni tra i rendimenti nelle varie date'''
correlazione_minima=0.5 # fa vedere correlazioni con valore pari o superiore
#correlazione calcolata non tra cat ma tra diverse date
#data.heatmap(df_return,variables_return,correlazione_minima)


'''2.9 - Scatterplot su due colonne'''
data.scatterplot_columns(df_return,'2016-06-30','2016-12-31') #scatterplot con date selezionate a piacere


'''2.10 - Pairplot sull'intero dataframe di rendimenti'''
data.pairplot(df_return,variables_return,'no') #con no non printa, particolarmente pesante 


'''2.11 - histograms, con bins = 20 (larghezza dei rettangoli'''
data.histograms_columns(df_return,variables_return,20) 


'''2. 12 - distribuzione dei testi tra le lingue'''
'''eseguire prima della traduzione'''
#faccio istogramma con distribuzione lingue e aggiungo colonna con specificazione
#della lingua (it, en, de...)
df_pol=data.controllo_testi_stranieri(df_pol) 
#creo variabile unica per il testo
df_pol['testo_intero']=df_pol['pol_testo']+df_pol['pol_finalita']+df_pol['ana_name'] 
variables_pol=['testo_intero']

'''2.13 - seleziono solo fondi con testo in italiano o seleziono anche le traduzioni di google'''
'''se modalità=traduci aggiunge anche i testi stranieri tradotti, se =solo italiani lascia solo quelli in italiano'''

modalità='traduci' 
file_traduzioni='fondi_tradotti_in_italiano.xlsx' #file con db documenti tradotti

df_pol=data.traduci_testi_stranieri(df_pol,modalità,file_traduzioni,gerarchia)






'''3 - DATAFRAME FINALI'''



'''3.1 - dataframe con parole e rendimenti'''
df_mixed=df_pol.merge(df_return,left_on=anagrafica,right_on=anagrafica)
variables_mixed=variables_return+variables_pol


'''3.2 - Opzionale: seleziono solo le categorie che hanno almeno X fondi'''
'''le categorie scartate le metto inuna unica di chiusura'''
'''valutare se l'accorpamento ex post non si apiù efficace, considerando il fatto 
che l'algoritmo valuta la vicinanza e una categoria con molta vairanza ha poche 
chance di essere quindi scelta'''
keep=True #se inserire o meno i fondi insufficientemente numeroso in una categoria nulla (esempio=Varie)
x=2 #numero minimo di fondi appartenenti alla categorie
df_mixed=data.class_size_threshold_all_labels(df_mixed,x,keep)
#df_pol=data.class_size_threshold_all_labels(df_pol,x,keep)
#df_return=data.class_size_threshold_all_labels(df_return,x,keep) 

#data.class_histogram(df_return,label)
#data.class_histogram(df_pol,label)
#data.class_histogram(df_mixed,label)

