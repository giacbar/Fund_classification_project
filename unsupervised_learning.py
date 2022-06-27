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

'''Selecting dataframes and variables'''
df=df_mixed
variables=variables_mixed

'''Selecting cluster number based on number of existing classes'''
label='cat_descrizione'
k=df[label].nunique()

'''Alternative: manual choice of cluster number'''
#k=15 es. di numero scelto discrezionalmente

'''Missing values imputation methods'''
method='kneighbors'
#df_arr=np.array(df[variables_return])
df_arr=supervised.impute_missing_values(df[variables_return],df[anagrafica],variables_return,label,method)  
df_arr=pd.DataFrame(df_arr,columns=variables_return)
#df=df[anagrafica+variables_pol].join(df_arr[variables_return])
df=df[anagrafica+variables_pol].join(df_arr[variables_return])

"""HIERARCHICAL clustering"""

clust_hier=unsupervised.hierarchical_clustering(df,variables,"clust_hier.xlsx",k,method="ward",metric="euclidean", plot=True,ndim=3)

'''Silhouette score with respect to real classes'''
unsupervised.silhouette_score (df[variables],df["cat_descrizione"], metric='euclidean')
'''Silhouette score with respect to clusters'''
unsupervised.silhouette_score (df[variables],clust_hier[1], metric='euclidean')

'''Comparing clusters against classes and find % of funds that inside a cluster belong to the same dominant class (purity score)'''
unsupervised.purity(clust_hier[3])[1]

from sklearn.metrics.cluster import normalized_mutual_info_score
veri_cluster=unsupervised.lista_cluster(df)
normalized_mutual_info_score(clust_hier[3],veri_cluster)

"""KMEANS clustering"""

multiple_kmeans(df,variables,5)

clust_k=unsupervised.kmeans_clustering(df,variables,"clust_k.xlsx",k, True,ndim=3)
'''Silhouette score with respect to real classes'''
unsupervised.silhouette_score (df[variables],df["cat_descrizione"], metric='euclidean')
'''Silhouette score with respect to clusters'''
unsupervised.silhouette_score (df[variables],clust_k[1], metric='euclidean')

'''Comparing clusters against classes and find % of funds that inside a cluster belong to the same dominant class (purity score)'''
unsupervised.purity(clust_k[3])[1]



"""Experiment: intersection clustering """
"""Intersection procedure by selecting different variables and clustering on these, then verify if 
   the clustering is consistent with the past ones each time"""
   
variables1=variables_return
variables2=variables_pol

clust_hier1=unsupervised.hierarchical_clustering(df,variables1,"clust_hier.xlsx",100,method="ward",metric="euclidean", plot=False, ndim=3)
clust_hier2=unsupervised.hierarchical_clustering(df,variables2,"clust_hier.xlsx",100,method="complete",metric="euclidean", plot=False, ndim=3)

clust_k1=unsupervised.kmeans_clustering(df,variables1,"clust_k.xlsx",k, False)
clust_k2=unsupervised.kmeans_clustering(df,variables2,"clust_k.xlsx",k, False)


'''Choosing the clusters to intersecate'''
lista=[clust_hier1[2],clust_hier2[2]]
titoli=df_return["ana_name"]

intersection = unsupervised.cluster_intersection(lista,titoli)
intersectionlist = unsupervised.intersection_df(intersection,df)
intersectionlist[0].to_excel(excel_writer="intersection.xlsx")

'''Comparing clusters against classes and find % of funds that inside a cluster belong to the same dominant class (purity score)'''
unsupervised.purity(intersectionlist[2])[1]

'''Silhouette score with respect to real classes'''
unsupervised.silhouette_score(df.sort_values("ana_name")[variables],intersectionlist[0].sort_values("ana_name")["new_cat_ID"], metric='euclidean')


"""Experimenting with PCA (Principal Components Analysis)"""

pca = PCA(n_components=3)

pca.fit(df[variables])  
X=pca.transform(df[variables])
print(pca.explained_variance_ratio_)  
sum(pca.explained_variance_ratio_)
print(pca.singular_values_)  
pca.components_.round(2)

unsupervised.PCA_plot(df,variables,2)
unsupervised.PCA_plot(df,variables,3)


