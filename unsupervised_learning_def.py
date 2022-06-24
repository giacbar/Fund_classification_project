# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 12:39:47 2019

@author: fida-stage1
"""
import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
import sklearn.decomposition
import scipy.cluster.hierarchy as sch
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from collections import Counter
import os
from sklearn.preprocessing import LabelEncoder
from random import randint
from mpl_toolkits.mplot3d import Axes3D


#alternative color selection method
#not in use
color_k2 = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w', 'aliceblue',
               'antiquewhite', 'aqua', 'aquamarine', 'azure', 'beige',  
               'bisque', 'black', 'blanchedalmond', 'blue', 'blueviolet',
               'brown', 'burlywood', 'cadetblue', 'chartreuse', 'chocolate',
               'coral', 'cornflowerblue', 'cornsilk', 'crimson', 'cyan',
               'darkblue', 'darkcyan', 'darkgoldenrod', 'darkgray',
               'darkgreen', 'darkgrey', 'darkkhaki', 'darkmagenta',
               'darkolivegreen', 'darkorange', 'darkorchid', 'darkred',
               'darksalmon', 'darkseagreen', 'darkslateblue', 'darkslategray',
               'darkslategrey', 'darkturquoise', 'darkviolet', 'deeppink',
               'deepskyblue', 'dimgray', 'dimgrey', 'dodgerblue', 'firebrick',
               'floralwhite', 'forestgreen', 'fuchsia', 'gainsboro',
               'ghostwhite', 'gold', 'goldenrod', 'gray', 'green',
               'greenyellow', 'grey', 'honeydew', 'hotpink', 'indianred', 
               'indigo', 'ivory', 'khaki', 'lavender', 'lavenderblush', 
               'lawngreen', 'lemonchiffon', 'lightblue', 'lightcoral',
               'lightcyan', 'lightgoldenrodyellow', 'lightgray', 'lightgreen',
               'lightgrey', 'lightpink', 'lightsalmon', 'lightseagreen', 
               'lightskyblue', 'lightslategray', 'lightslategrey',
               'lightsteelblue', 'lightyellow', 'lime', 'limegreen',
               'linen', 'magenta', 'maroon', 'mediumaquamarine', 'mediumblue',
               'mediumorchid', 'mediumpurple', 'mediumseagreen', 
               'mediumslateblue', 'mediumspringgreen', 'mediumturquoise', 
               'mediumvioletred', 'midnightblue', 'mintcream', 'mistyrose', 
               'moccasin', 'navajowhite', 'navy', 'oldlace', 'olive', 
               'olivedrab', 'orange', 'orangered', 'orchid', 'palegoldenrod',
               'palegreen', 'paleturquoise', 'palevioletred', 'papayawhip',
               'peachpuff', 'peru', 'pink', 'plum', 'powderblue', 'purple',
               'rebeccapurple', 'red', 'rosybrown', 'royalblue', 'saddlebrown',
               'salmon', 'sandybrown', 'seagreen', 'seashell', 'sienna', 
               'silver', 'skyblue', 'slateblue', 'slategray', 'slategrey',
               'snow', 'springgreen', 'steelblue', 'tan', 'teal', 'thistle',
               'tomato', 'turquoise', 'violet', 'wheat', 'white', 'whitesmoke',
               'yellow', 'yellowgreen','cloudy blue']




'''######## Data management #########'''

def lista_cluster(df):
    """lista_cluster =  list of lists containing names, ordered by cluster
       input: -dataframe  """
    k=df["cat_descrizione"].nunique()
    df["cat_ID"]=pd.factorize(df['cat_descrizione'])[0]
    vera_lista_cluster = [0.0]*(k)
    for i in range(0,k):
            for j in range(0,len(df)):
                if (int(df["cat_ID"][j]) == i):                        
                    vera_lista_cluster[i] = np.append(vera_lista_cluster[i],df["cat_descrizione"][j])              
    for i in range(0,k):
            vera_lista_cluster[i] = np.delete(vera_lista_cluster[i],0)
                # eliminiamo il primo elemento di ogni lista che per costruzione 
                # è 0.0
            vera_lista_cluster[i] = list(vera_lista_cluster[i])
    return vera_lista_cluster

'''######### Plots ##########'''

def PCA_plot(df,variables,ndim,label):
    
    '''scatter plot of the funds with the colors of their category
    
    input: dataframe
           features (lista)
           number of dimensions (2 o 3)
           label (cat_descrizione,classe,super_classe)'''
    color_k = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w', 'aliceblue',
               'antiquewhite', 'aqua', 'aquamarine', 'azure', 'beige',  
               'bisque', 'black', 'blanchedalmond', 'blue', 'blueviolet',
               'brown', 'burlywood', 'cadetblue', 'chartreuse', 'chocolate',
               'coral', 'cornflowerblue', 'cornsilk', 'crimson', 'cyan',
               'darkblue', 'darkcyan', 'darkgoldenrod', 'darkgray',
               'darkgreen', 'darkgrey', 'darkkhaki', 'darkmagenta',
               'darkolivegreen', 'darkorange', 'darkorchid', 'darkred',
               'darksalmon', 'darkseagreen', 'darkslateblue', 'darkslategray',
               'darkslategrey', 'darkturquoise', 'darkviolet', 'deeppink',
               'deepskyblue', 'dimgray', 'dimgrey', 'dodgerblue', 'firebrick',
               'floralwhite', 'forestgreen', 'fuchsia', 'gainsboro',
               'ghostwhite', 'gold', 'goldenrod', 'gray', 'green',
               'greenyellow', 'grey', 'honeydew', 'hotpink', 'indianred', 
               'indigo', 'ivory', 'khaki', 'lavender', 'lavenderblush', 
               'lawngreen', 'lemonchiffon', 'lightblue', 'lightcoral',
               'lightcyan', 'lightgoldenrodyellow', 'lightgray', 'lightgreen',
               'lightgrey', 'lightpink', 'lightsalmon', 'lightseagreen', 
               'lightskyblue', 'lightslategray', 'lightslategrey',
               'lightsteelblue', 'lightyellow', 'lime', 'limegreen',
               'linen', 'magenta', 'maroon', 'mediumaquamarine', 'mediumblue',
               'mediumorchid', 'mediumpurple', 'mediumseagreen', 
               'mediumslateblue', 'mediumspringgreen', 'mediumturquoise', 
               'mediumvioletred', 'midnightblue', 'mintcream', 'mistyrose', 
               'moccasin', 'navajowhite', 'navy', 'oldlace', 'olive', 
               'olivedrab', 'orange', 'orangered', 'orchid', 'palegoldenrod',
               'palegreen', 'paleturquoise', 'palevioletred', 'papayawhip',
               'peachpuff', 'peru', 'pink', 'plum', 'powderblue', 'purple',
               'rebeccapurple', 'red', 'rosybrown', 'royalblue', 'saddlebrown',
               'salmon', 'sandybrown', 'seagreen', 'seashell', 'sienna', 
               'silver', 'skyblue', 'slateblue', 'slategray', 'slategrey',
               'snow', 'springgreen', 'steelblue', 'tan', 'teal', 'thistle',
               'tomato', 'turquoise', 'violet', 'wheat', 'white', 'whitesmoke',
               'yellow', 'yellowgreen','cloudy blue']

   # color_k = []
   # n = 300
   #for i in range(n):
   #    color_k.append('#%06X' % randint(0, 0xFFFFFF))    

    LE = LabelEncoder()
    df['code'] = LE.fit_transform(df[label])
    
    pca = sklearn.decomposition.PCA(n_components=ndim)
    pca.fit(df[variables])  
    X=pca.transform(df[variables])
    columns=[0]*ndim
    for i in range(0,ndim):
        columns[i]='PC'+str(i)
    pcas=pd.DataFrame(X, columns=columns)
    color=[0]*len(df)
    for i in range(len(df)):
        for clu in range(df[label].nunique()):
            if(df['code'][i]==clu):
                color[i]=color_k[clu]
    if(ndim==2):         
        plt.figure(figsize=(20, 20))
        plt.scatter(pcas.iloc[:,0],pcas.iloc[:,1], c=color)
        plt.show()      
    elif(ndim==3):
        fig = plt.figure(figsize=(10,10))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(pcas.iloc[:,0],pcas.iloc[:,1], pcas.iloc[:,2],s=10,c=color)
        plt.show()

'''#########  Clustering functions ######### '''

def hierarchical_clustering(completedataframe,columns,excelname,k,method, metric, plot,ndim):
    """Runs the hierarchical clustering algorithm for the dataframe.
       INPUTS: -dataframe
               -columns on which to cluster observations
               -method of computing distance between clusters (complete, single..)
               -metric to be used for distance computation (euclidean..)
               -k: desired number of clusters
               -excelname: name of the excel file to create
               -plot: Y/N
       OUTPUTS: -clustering labels 
                -dataframe of funds with category ordered by clusters
                -lists of clusters (name and category)
                -dendrogram  """
                
    dataframe=completedataframe[columns]           
    linkage=sch.linkage(dataframe,method, metric)
    labels = sch.fcluster(linkage, k, 'maxclust')
    clusterdf=pd.DataFrame(completedataframe,columns=["ana_name","cat_descrizione"])
    clusterdf["kclust"]=labels
    clusterdf["kclust"]=labels
    clusterdf=clusterdf.sort_values(by="kclust")
    #clusterdf.to_excel(excel_writer=excelname)

    ct_k = linkage[-(k-1),2]
    if(plot==True):
        plt.figure(figsize=(12, 12))
        plt.title('Hierarchical Clustering Dendrogram')
        plt.ylabel('distance')
        sch.dendrogram(
                linkage,
                leaf_rotation=90.,  # rotates the x axis labels
                #leaf_font_size=8.,  # font size for the x axis labels
                color_threshold=ct_k
                )
        plt.show()
        
        color_k = []
        n = 500
        for i in range(n):
            color_k.append('#%06X' % randint(0, 0xFFFFFF))
        
        pca = sklearn.decomposition.PCA(n_components=ndim)
        pca.fit(dataframe)  
        X=pca.transform(dataframe)
        columns=[0]*ndim
        for i in range(0,ndim):
            columns[i]='PC'+str(i)
        pcas=pd.DataFrame(X, columns=columns)
        color=[0]*len(dataframe)
        for i in range(len(dataframe)):
            for clu in range(k):
                if(labels[i]==clu):
                    color[i]=color_k[clu]
        if(ndim==2):         
            plt.figure(figsize=(12, 12))
            plt.scatter(pcas.iloc[:,0],pcas.iloc[:,1], c=color)
            plt.show()      
        elif(ndim==3):
            fig = plt.figure(figsize=(12, 12))
            
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(pcas.iloc[:,0],pcas.iloc[:,1], pcas.iloc[:,2],s=8,c=color)
            plt.show()    
       
    lista_cluster_nome = [0.0]*(k)
    for i in range(0,k):
            for j in range(0,len(labels)):
                if (int(labels[j])-1 == i):
                    lista_cluster_nome[i] = np.append(lista_cluster_nome[i],completedataframe["ana_name"][j])              
    for i in range(0,k):
            lista_cluster_nome[i] = np.delete(lista_cluster_nome[i],0)
            lista_cluster_nome[i] = list(lista_cluster_nome[i])
            
    
    lista_cluster_cate= [0.0]*(k)
    for i in range(0,k):
            for j in range(0,len(labels)):
                if (int(labels[j])-1 == i):
                    lista_cluster_cate[i] = np.append(lista_cluster_cate[i],completedataframe["cat_descrizione"][j])              
    for i in range(0,k):
            lista_cluster_cate[i] = np.delete(lista_cluster_cate[i],0)
            lista_cluster_cate[i] = list(lista_cluster_cate[i])       
            
    return clusterdf,labels, lista_cluster_nome, lista_cluster_cate



def kmeans_clustering(completedataframe,columns,excelname,k,grafico,ndim):
    """runs the k-means clustering algorithm for the dataframe.
       INPUTS: -dataframe to cluster the observation
               -columns of the variables you want to cluster
               -dataframe of the columns to be plotted
               -number of clusters
               -plot Y/N
               -excelname: name of the excel file to create

               
       OUTPUTS: -clustering labels 
                -dataframe of funds with category ordered by clusters
                -lists of clusters (name and category)
                -plot of the clustering   """
    path=os.getcwd()
    os.chdir(path)
    dataframe=completedataframe[columns]  
    kmeansalg=KMeans(n_clusters=k).fit(dataframe)
    labels=kmeansalg.labels_
    clusterdf=pd.DataFrame(completedataframe,columns=["ana_name","cat_descrizione"])
    clusterdf["kclust"]=labels
    clusterdf=clusterdf.sort_values(by="kclust")
    #clusterdf.to_excel(excel_writer=excelname)
    if(grafico==True):
        
        color_k = []
        n = 500
        for i in range(n):
            color_k.append('#%06X' % randint(0, 0xFFFFFF))

        pca = sklearn.decomposition.PCA(n_components=ndim)
        pca.fit(dataframe)  
        X=pca.transform(dataframe)
        columns=[0]*ndim
        for i in range(0,ndim):
            columns[i]='PC'+str(i)
        pcas=pd.DataFrame(X, columns=columns)
        color=[0]*len(dataframe)
        for i in range(len(dataframe)):
            for clu in range(k):
                if(labels[i]==clu):
                    color[i]=color_k[clu]
                    
        if(ndim==2):         
            plt.figure(figsize=(12, 12))
            plt.scatter(pcas.iloc[:,0],pcas.iloc[:,1], c=color)
            plt.show()      
        elif(ndim==3):
            fig = plt.figure(figsize=(12, 12))
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(pcas.iloc[:,0],pcas.iloc[:,1], pcas.iloc[:,2],s=1,c=color)
            plt.show()
            
    lista_cluster_nome = [np.array([0])]*k
    for i in range(0,len(completedataframe)):
            lista_cluster_nome[labels[i]] = np.append(lista_cluster_nome[labels[i]],completedataframe["ana_name"][i])
    for i in range(0,len(lista_cluster_nome)):
            lista_cluster_nome[i] = np.delete(lista_cluster_nome[i],0,0)
            lista_cluster_nome[i] = list(lista_cluster_nome[i])
    for j in range(0,len(lista_cluster_nome[i])):
            lista_cluster_nome[i][j] = str(lista_cluster_nome[i][j])
            
    
    lista_cluster_cate = [np.array([0])]*k
    for i in range(0,len(completedataframe)):
            lista_cluster_cate[labels[i]] = np.append(lista_cluster_cate[labels[i]],completedataframe["cat_descrizione"][i])
    for i in range(0,len(lista_cluster_cate)):
            lista_cluster_cate[i] = np.delete(lista_cluster_cate[i],0,0) 
            lista_cluster_cate[i] = list(lista_cluster_cate[i])
    for j in range(0,len(lista_cluster_cate[i])):
            lista_cluster_cate[i][j] = str(lista_cluster_cate[i][j]) 
              
    return clusterdf, labels, lista_cluster_nome,lista_cluster_cate

def multiple_kmeans(df,variables,iterations,k):
    '''conduce il kmeans clusterin #iterations volte, printa il purity score medio
    INPUT: df, variables, numero di iterazioni
    OUTPUT'''
    
    results=[0]*iterations
    clust_k=[0]*iterations
    for i in range(len(results)):
        clust_k[i]=kmeans_clustering(df,variables,"clust_k.xlsx",k, False, ndim=3)
        results[i]=purity(clust_k[i][3])[1]
    print('mean purity score:{0}'.format(np.mean(results)))




def combination_search(lista_cluster,tutti_titoli):
    """
    Function that takes as input a list whose elemtns are the cluster compositions
    built using the algorithms multiple times using different methods (different periods metrics..)
    and finds the combinations of repeating elements
    
     input: list of lists of clusters
            list of all fund names

    output: list of fund combinations
            matrix A for summarising the cluster combinations
    """
    
    color_k = []
    n = 500
    for i in range(n):
        color_k.append('#%06X' % randint(0, 0xFFFFFF))

    A = pd.DataFrame(np.zeros((len(lista_cluster)+1,len(tutti_titoli))))
    A.columns = tutti_titoli 
    for i in range(0,len(lista_cluster)):
        for j in range(0,len(lista_cluster[i])):
            A.loc[i, lista_cluster[i][j]] = color_k[j]   
    lista = []
    for i in A.columns:
        if(A.loc[len(lista_cluster),i] == int(0)):
    
            A.loc[len(lista_cluster),i] = '1'
            temp = [i]     
            for j in A.columns: # per ogni colonna
                if(i!=j): # evitiamo di controllare titolo i con il titolo i
                    if(A[i][:-1].equals(A[j][:-1])):
                        temp.append(j)
                        A.loc[len(lista_cluster),j] = '1'             
            lista.append(temp) 
    preds=[]
    for cluster_index, cluster in enumerate(lista):
        for fund_index, fund in enumerate(cluster):
            preds.append([cluster_index])
    
    return lista,A,preds





def cluster_intersection(lista_cluster,tutti_titoli):
    """
    outputta solo una parte della precedente funzione
    
     input: lista i cui elementi sono le liste dei cluster
            lista contenenti i nomi di tutti i titoli considerati

    output: 
            matrice A che sintetizza le combinazioni dei cluster ottenuti ad
            ogni esecuzione dell'algoritmo
    """
    
    color_k = []
    n = 500
    for i in range(n):
        color_k.append('#%06X' % randint(0, 0xFFFFFF))

        # vettore contenente tutti i possibili colori           
    A = pd.DataFrame(np.zeros((len(lista_cluster),len(tutti_titoli))))
    A.columns = tutti_titoli 
    
    for i in range(0,len(lista_cluster)):
        for j in range(0,len(lista_cluster[i])):
            A.loc[i, lista_cluster[i][j]] = color_k[j]  
    A=A.transpose()
    A["ana_name"]=A.index
    A.index = range(A.shape[0])
    #A=A.rename(columns={0:"clust1", 1:"clust2",2:"clust3",3:"clust4",4:"clust5"})
    
    return A




def intersection_df(intersection,df):
    """shows which funds have clustered together in the previous clustering runs"""
    
    inte=intersection.sort_values(list(intersection.columns[:-1]))
    #intersection.groupby(by=list(intersection.columns))
    inte["new_cat_ID"]=[0]*len(inte)
    inte.index = range(len(inte))
    
    for i in range(len(inte)):
        if(i+1==len(inte)):
            break
    
        if(list(inte.loc[i+1,:][:-2] != inte.loc[i,:][:-2]) == [False]*(len(inte.columns)-2)):
             inte["new_cat_ID"][i+1]=inte["new_cat_ID"][i]
        else:
             inte["new_cat_ID"][i+1]=inte["new_cat_ID"][i]+1   
    k=inte["new_cat_ID"].nunique()
    lista_cluster_nomi= [0]*k
    for i in range(0,k):
            for j in range(0,len(inte)):
                if (int(inte["new_cat_ID"][j]) == i):                        
                    lista_cluster_nomi[i] = np.append(lista_cluster_nomi[i],inte["ana_name"][j])              
    for i in range(0,k):
            lista_cluster_nomi[i] = np.delete(lista_cluster_nomi[i],0)
                # eliminiamo il primo elemento di ogni lista che per costruzione 
                # è 0.0
            lista_cluster_nomi[i] = list(lista_cluster_nomi[i])
                    
    inte=pd.merge(inte,df[["cat_descrizione","ana_name"]],how="inner", left_on="ana_name",right_on="ana_name",validate="one_to_one")
    #inte.to_excel(excel_writer="intersection.xlsx")
    
    lista_cluster_cate= [np.array([0])]*k
    for i in range(0,len(inte)):
            lista_cluster_cate[inte["new_cat_ID"][i]] = np.append(lista_cluster_cate[inte["new_cat_ID"][i]],inte["cat_descrizione"][i])
    for i in range(0,len(lista_cluster_cate)):
            lista_cluster_cate[i] = np.delete(lista_cluster_cate[i],0,0) 
            lista_cluster_cate[i] = list(lista_cluster_cate[i])
    for j in range(0,len(lista_cluster_cate[i])):
            lista_cluster_cate[i][j] = str(lista_cluster_cate[i][j]) 
              

    return inte, lista_cluster_nomi, lista_cluster_cate


def from_clusters_combinations_to_dataframe(combination, clusterings):
    """takes the inputs from the combination function and from the clustering functions, 
        yields a dataframe of the funds with the labels of the clusters assigned with each clustering procedure
        INPUT: -output of the combination_search function
               - a list of clustering results from hierarchical or kmeans (of the type clust_k[2])
        OUTPUT: -dataframe with funds and related cluster labels for each iteration of the clustering procedure  """
    
    preds=pd.DataFrame()
    for cluster_index, cluster in enumerate(combination[0]):
        for fund_index, fund in enumerate(cluster):
            inputs=[combination[0][cluster_index][fund_index], clusterings[0][1][fund_index],clusterings[1][1][fund_index],clusterings[2][1][fund_index]]
            stuff=np.array([inputs])
            thing=pd.DataFrame(stuff,columns=["fund_name","cluster_1","cluster_2","cluster_3"])
            preds=preds.append(thing)
     
    return preds
 





def two_step(D,titles,k1,k2):
    """
    funzione che effettua la procedura two step per trovare i clusters, prima hierarchical poi kmeans
    
     input: DataFrame dei rendimenti
            numero di cluster che vogliamo ottenere nel primo step
            lista dei codici dei titoli
            numero tra [0-100] che corrisponde alla riduzione percentuale 
            dell'SSE necessaria per selezionare il numero ottimale di cluster
            che vogliamo ottenere nel secondo step
    """
    z1 =hierarchical_clustering(D,titles,"clust_hier.xlsx",k1,method="complete",metric="euclidean", plot=False)[2]
    z2 = [0.0]*k1
    index=D["ana_name"]+" "+D["cat_descrizione"]
    B=D.set_index(index)

    for i in range(0,k1):   
        data1 = pd.DataFrame(np.transpose(np.array([np.std(B[titles].loc[z1[i]]),np.mean(B[titles].loc[z1[i]])])))
        #temp_label = B[titles].loc[z1[i]].columns
        if(data1.shape[0] == 1):
            data1[data1.columns[0]] = data1[data1.columns[0]]
            data1[data1.columns[1]] = data1[data1.columns[1]]
        else:
            data1[data1.columns[1]] = (
                    data1[data1.columns[1]]-np.min(data1[data1.columns[1]])
                    )/(
                            np.max(data1[data1.columns[1]])-np.min(data1[data1.columns[1]])
                            ) 
            data1[data1.columns[0]] = data1[data1.columns[0]]/np.max(data1[data1.columns[0]]) 
        # abbiamo normalizzato i dati in modo che media e std siano compresi
        # tra 0 e 1


        z2[i] = kmeans_clustering(D,titles,"clust_k.xlsx",k2, False)[2]    
        #for j in range(0,len(z2[i])):
           # for k in range(0,len(z2[i][j])):
               # z2[i][j][k] = temp_label[z2[i][j][k]]      
    return z2


'''########## funzioni per punteggi############'''

def checkEqual(iterator):
    """funzionale per checkList"""
    iterator = iter(iterator)
    try:
        first = next(iterator)
    except StopIteration:
        return True
    return all(first == rest for rest in iterator)

def checkList(cluster_list):
    """controlla il numero di cluster con tutti i fondi della stessa categoria"""
    vec=[0]*len(cluster_list)
    for i in range(len(cluster_list)):
        vec[i]=checkEqual(cluster_list[i])
    
    print(vec.count(True))
    
    
    
def frequency_scores(lista, percentage):
    """INPUT: Lista dei cluster, livello di percentuale desiderato per fondi della stessa categoria in un cluster
       OUTPUT: True: il cluster ha >percentuale di fondi della stessa cat.
               False: il cluster non ha..................................."""
    results=[]
    for i in range(len(lista)):
        counter=Counter(lista[i])
        times=list(counter.values())
        times.sort(reverse=True)
        if (times[0]/len(lista[i])>percentage):
            results.append(True)
        else:
            results.append(False)
    return results


def recursive_len(item):
    """ funzione accessoria per purity"""
    if type(item) == list:
        return sum(recursive_len(subitem) for subitem in item)
    else:
        return 1
    
    
def purity(lista):
    """computes the purity score of the list of clusters:
       sum of most common categories in each cluster/all the funds"""
    results=pd.DataFrame(np.zeros((len(lista),2)))
    
    for i in range(len(lista)):
            most_common,num_most_common = Counter(lista[i]).most_common(1)[0]
            results[0][i]=most_common
            results[1][i]=num_most_common
    score=results[1].sum()/recursive_len(lista)
    print(score)
    return results, score
        
       
    return results

def silhouette(lista,df,titles):
    """INPUT: Lista dei clusterings, df, colonne
       OUTPUT: lista con silhouette scores globali per ogni clustering"""
    
    punteggi=[0]*len(lista)
    for i in range(len(lista)):
        preds=[]
        for cluster_index, cluster in enumerate(lista[i]):
            for fund_index, fund in enumerate(cluster):
                preds.append([cluster_index])
        s=silhouette_score (df[titles],preds, metric='euclidean')
        punteggi[i]=s
    return punteggi

