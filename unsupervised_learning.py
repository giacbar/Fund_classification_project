# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 12:41:05 2019

@author: fida-stage1
"""
import numpy as np
import pandas as pd


import unsupervised_learning_def as unsupervised
import supervised_learning_def as supervised
from sklearn.decomposition import PCA


'''seleziono dataframe e variabili'''
df=df_mixed
variables=variables_mixed

'''scelgo il numero di cluster'''

'''basandomi sul numero di categorie presenti'''
label='cat_descrizione'
k=df[label].nunique()
''' scelta discrezionale'''
#k=15 es. di numero scelto discrezionalmente


'''imputazione di valori mancanti'''
method='kneighbors'
#df_arr=np.array(df[variables_return])
df_arr=supervised.impute_missing_values(df[variables_return],df[anagrafica],variables_return,label,method)  
df_arr=pd.DataFrame(df_arr,columns=variables_return)
#df=df[anagrafica+variables_pol].join(df_arr[variables_return])
df=df[anagrafica+variables_pol].join(df_arr[variables_return])


"""HIERARCHICAL clustering"""

'''realizza i cluster'''
clust_hier=unsupervised.hierarchical_clustering(df,variables,"clust_hier.xlsx",k,method="ward",metric="euclidean", plot=True,ndim=3)
'''punteggio consistenza delle categorie originali'''
unsupervised.silhouette_score (df[variables],df["cat_descrizione"], metric='euclidean')
'''punteggio consistenza delle categorie create dal clustering'''
unsupervised.silhouette_score (df[variables],clust_hier[1], metric='euclidean')

'''confronta cluster conc ategorie e dice la % dei fondi che all'interno dei cluster appartengono ad una stessa categoria dominante'''
unsupervised.purity(clust_hier[3])[1]

from sklearn.metrics.cluster import normalized_mutual_info_score
veri_cluster=unsupervised.lista_cluster(df)
normalized_mutual_info_score(clust_hier[3],veri_cluster)

"""KMEANS clustering"""

'''realizza i cluster x calcolo punteggio'''
'''generazione 500 volte cluster con centroidi random poi fa media per calcolare punteggio'''
'''per collocare i fondi occorre scegliere una delle soluzioni o fare l'intersezione o altro'''

multiple_kmeans(df,variables,5)

'''realizza i cluster'''
clust_k=unsupervised.kmeans_clustering(df,variables,"clust_k.xlsx",k, True,ndim=3)
'''punteggio consistenza delle categorie originali'''
unsupervised.silhouette_score (df[variables],df["cat_descrizione"], metric='euclidean')
'''punteggio consistenza delle categorie create dal clustering'''
unsupervised.silhouette_score (df[variables],clust_k[1], metric='euclidean')

'''confronta cluster conc ategorie e dice la % dei fondi che all'interno dei cluster appartengono ad una stessa categoria dominante'''
unsupervised.purity(clust_k[3])[1]






"""Clustering con intersezione: in fase di aggiornamento"""
"""Intersection procedure by selecting different variables and clustering on these, then verify if 
   the clustering are consistent each time"""
   
'''sistemare codice in unica procedura che calcola cluster poi fa intersezioni opzionali'''


variables1=variables_return
variables2=variables_pol

clust_hier1=unsupervised.hierarchical_clustering(df,variables1,"clust_hier.xlsx",100,method="ward",metric="euclidean", plot=False, ndim=3)

clust_hier2=unsupervised.hierarchical_clustering(df,variables2,"clust_hier.xlsx",100,method="complete",metric="euclidean", plot=False, ndim=3)


clust_k1=unsupervised.kmeans_clustering(df,variables1,"clust_k.xlsx",k, False)

clust_k2=unsupervised.kmeans_clustering(df,variables2,"clust_k.xlsx",k, False)



'''scelata cluster da intersecare'''
lista=[clust_hier1[2],clust_hier2[2]]

titoli=df_return["ana_name"]

intersection = unsupervised.cluster_intersection(lista,titoli)
intersectionlist = unsupervised.intersection_df(intersection,df)


'''lista cluster, sono tanti perchè quelli da 1 andrebbero accorpati'''
intersectionlist[0].to_excel(excel_writer="intersection.xlsx")

unsupervised.purity(intersectionlist[2])[1]

unsupervised.silhouette_score(df.sort_values("ana_name")[variables],intersectionlist[0].sort_values("ana_name")["new_cat_ID"], metric='euclidean')






"""trying some PCA"""
pca = PCA(n_components=3)

pca.fit(df[variables])  
X=pca.transform(df[variables])
print(pca.explained_variance_ratio_)  
sum(pca.explained_variance_ratio_)
print(pca.singular_values_)  
pca.components_.round(2)


unsupervised.PCA_plot(df,variables,2)
unsupervised.PCA_plot(df,variables,3)



















