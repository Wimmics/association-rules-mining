from base64 import encode
import warnings
warnings.filterwarnings('ignore')

from flask import Flask, request, jsonify, render_template

import pandas as pd

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# import SPARQLWrapper
import json
# from SPARQLWrapper import SPARQLWrapper, JSON

import networkx as nx
from cdlib import algorithms,viz

import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from scipy.cluster.hierarchy import dendrogram, linkage, fcluster, ward
import scipy.linalg.blas

from mlxtend.frequent_patterns import association_rules
from mlxtend.frequent_patterns import fpgrowth, fpmax

from keras.models import Model
from keras.layers import Input, Dense
from keras import regularizers

from stellargraph.data import BiasedRandomWalk
from stellargraph import StellarGraph


from gensim.models import Word2Vec

from utils import *

import argparse, sys
from os.path import exists

parser = argparse.ArgumentParser()

parser.add_argument('--endpoint', help='The endpoint from where retrieve the data (identified through codes: issa, covid).', required=False) 
parser.add_argument('--input', help='If available, path to the file containing the input data', required=False) # to reduce the data import time when using the same data
parser.add_argument('--graph', help='In case there is a graph where to get the data from in the endpoint, provide (valid for issa: agrovoc, geonames, wikidata, dbpedia)', required=False)
parser.add_argument('--lang', help='The language of the labels', required=False) # language of labels 
parser.add_argument('--filename', help='The output file name. If not provided, it will be automatically generated based on the input information.', required=False) 
parser.add_argument('--conf', help='Minimum confidence of rules. Default is .7, rules with less than x confidence are filtered out.', default=0.7, required=False) 
parser.add_argument('--int', help='Minimum interestingness (serendipity, rarity) of rules. Default is .3, rules with less than x interestingess are filtered out.', default=0.3, required=False) 
parser.add_argument('--occurrence', help='Keep only terms co-occurring more than x times. Default is 5', default=5, required=False) # keep only terms co-occurring more than x times, default is 5

args = parser.parse_args()

## load the queries 
with open('queries.json') as f:
    queriesData = json.load(f)

app = Flask(__name__)

def query() :
    """
    Query all articles published between two years

    Parameters :
        year1 : int
            The oldest year of publication
        year2 : int
            The latest year of publication
        length : int
            Maximum length of the list containing the DataFrames (for example if we want maximum 50 000 rows, the list has a length of 5)

    Returns :
        df_total : DataFrame
            The DataFrame with all the request responses
    """

    offset = 0
    if (args.graph != None):
        query = queriesData['queries'][args.endpoint][args.graph]
    else:
        query = queriesData['queries'][args.endpoint]

    print("query = ", query)

    complete_query = query % (args.lang, offset) if args.lang else query % (offset)
    df_query = sparql_service_to_dataframe(queriesData['endpoints'][args.endpoint], complete_query)

    ## List with all the request responses ##
    list_total = [df_query]
    ## Get all data by set the offset at each round ##
    while (df_query.shape[0] > 0):
        print("offset = ", offset)
        offset += 10000
        complete_query = query % (args.lang, offset) if args.lang else query % (offset)
        df_query = sparql_service_to_dataframe(queriesData['endpoints'][args.endpoint], complete_query)
        list_total.append(df_query)

    ## Concatenate all the dataframe from the list ##
    df_total = pd.concat(list_total)

    datafile = 'data/input_data_' + args.endpoint + ('_' + args.graph if args.graph else '') + ('_' + args.lang if args.lang else '') + '.csv'
    df_total.to_csv(datafile, sep=',', index=False, header=list(df_total.columns), mode='w')

    return df_total

# Création de la matrice de co-occurences qui est notre jeu de données pour le clustering 
def getMatrixCooccurrences(df_article_sort):
    ### Découpage en TRAIN/TEST des données ### 

    # train_index, test_index  = train_test_split(df_article_sort[['article']].drop_duplicates(), test_size=0.2) #,stratify=df_article_sort[['article','year']].drop_duplicates()['year']
    # train = df_article_sort[df_article_sort['article'].isin(train_index['article'].unique() )]
    # test = df_article_sort[df_article_sort['article'].isin(test_index['article'].unique() )]

    ### METTRE TOUT EN string + spécifier que Label et year sont des catégories pour le one-hot-encoding ###


    df_article_sort[['label']].drop_duplicates()

    train = df_article_sort.astype({"article" : str, "label":str})
    train["label"] = train["label"].astype('category')
    # train["year"] = train["year"].astype('category')

    # test = test.astype({"article" : str, "label":str})
    # test["label"] = test["label"].astype('category')
    # test["year"] = test["year"].astype('category')

    ##One hot encoding train set (5000 by 5000 articles) + Sparse type to reduce the memory 

    one_hot = pd.get_dummies(train[train['article'].isin(train['article'].unique()[0:5000])].drop_duplicates().\
                    set_index('article')).sum(level=0).apply(lambda y : y.apply(lambda x : 1 if x>=1 else 0)).\
                    astype("Sparse[int]")
    i = 5000
    while(one_hot.shape[0] < len(train['article'].unique())):
        one_hot = one_hot.append(pd.get_dummies(train[train['article'].isin(train['article'].unique()[i:i+5000])].drop_duplicates().\
                    set_index('article')).sum(level=0).apply(lambda y : y.apply(lambda x : 1 if x>=1 else 0)).\
                    astype("Sparse[int]"))
        i = i+5000

    ### Remplacer les NaN par 0 et supprimer les lignes avec que des 0 ###

    one_hot = one_hot.fillna(0)
    one_hot = one_hot.loc[:, (one_hot != 0).any(axis=0)]

    ### Passer la matrice en type Sparse pour accélérer les calculs ###

    one_hot =one_hot.astype("Sparse[int]")

    ## Supprimer les variables "year" si on ne veut pas les prendre en compte dans l'analyse ##

    drop = [x for x in one_hot.columns if not x.startswith('label_')]
    one_hot_label = one_hot.drop(drop,axis=1)
    one_hot_label.columns = list(pd.DataFrame(one_hot_label.columns)[0].apply(lambda x : x.split('_')[-1]))

    return one_hot_label

### Réduction du nombre de variables + Clustering
#L'autoencoder permet de réduire la dimension et de pouvoir appliquer la CAH qui n'est pas robuste face à un nombre trop importants de variables
def applyAutoencoder(one_hot_matrix):
    ### Autoenconder  ###

    input_dim = one_hot_matrix.shape[1]
    encoding_dim = 128
    # Number of neurons in each Layer [8, 6, 4, 3, ...] of encoders
    input_layer = Input(shape=(input_dim, ))
    encoder_layer_1 = Dense(2048, activation="tanh")(input_layer)
    encoder_layer_2 = Dense(1024, activation="tanh")(encoder_layer_1)
    encoder_layer_3 = Dense(256, activation="tanh")(encoder_layer_2)
    encoder_layer_4 = Dense(encoding_dim, activation="tanh",kernel_regularizer=regularizers.l1_l2(l1=0.001, l2=0.01))(encoder_layer_3)
    encoder = Model(inputs=input_layer, outputs=encoder_layer_4)

    # Use the model to predict the factors which sum up the information of interest rates.
    encoded_data = pd.DataFrame(encoder.predict(one_hot_matrix))

    encoded_data.index = one_hot_matrix.index
    
    return encoded_data

def clusteringCAH(encoded_data):

    nb_cluster,groupe,index = elbow_method(encoded_data,10,"cosine")

    groupe[groupe.index.isin(index)].groupby([groupe[groupe.index.isin(index)].index]).count()

    ### Visualisation des labels les plus fréquents dans le groupe 1 ###

    # count = train[train['article'].isin(list(groupe[groupe.index == 1]['article']))]['label'].value_counts()[:20].sort_values(ascending=True)
    # count.plot(kind='barh').set_title('The 15 most frequent named entities')

    # Apply again elbow method to the groups with more than 500 articles #

    new_cluster,index_of_cluster = repeat_cluster(encoded_data,groupe,index,500,5)

    return groupe, new_cluster, index, index_of_cluster

### Community alogrithms (Walk Trap) 
def applyWalkTrap(one_hot_label):
    ## Co-occurencies matrix
    coooc_s = coocc_matrix_Label(one_hot_label)

    ## Créer des tuples avec co-occurences différentes de 0 ##

    labels = one_hot_label.columns
    tuple_list = []
    for i in range(len(coooc_s)) : 
        no_zero = np.where(coooc_s[i]!=0)
        for j in no_zero[0]:
            tuple_list.append([labels[i], labels[j],coooc_s[i][j]])
    
    print(tuple_list)
    ## Create Graph ##
    G=nx.Graph()

    for egde in tuple_list:
        G.add_edge(egde[0], egde[1], weight=egde[2])

    com_wt = algorithms.walktrap(G)
    return com_wt.communities
    

def rulesNoClustering(one_hot_matrix):
    regles_fp = fp_growth(one_hot_matrix, 3 , float(args.conf))

    print("No clustering | Number of rules before filtering = " + str(regles_fp.shape[0]))

    ### POST-PROCESSING : interestingness + règles redondantes ###
    regles_fp = interestingness_measure(regles_fp, one_hot_matrix)
    regles_fp = delete_redundant(regles_fp)
    print ("No clustering | Number of rules after redundancy filter = " + str(regles_fp.shape[0]))
    regles = create_rules_df(regles_fp, float(args.int))
    print ("No clustering | Number of rules after interestingness filter = " + str(regles_fp.shape[0]))
   
    regles['cluster'] = "no_clustering"

    print("No clustering | Number of rules : " + str(regles.shape[0]))

    return regles

def rulesCommunities(one_hot_label, communities_wt):
    # drop = [x for x in one_hot.columns if not x.startswith('label_')]
    # one_hot_community = one_hot.drop(drop,axis=1)
    # one_hot_community.columns = list(pd.DataFrame(one_hot_community.columns)[0].apply(lambda x : x.split('_')[-1]))

    regles_communities_wt = fp_growth_with_community(one_hot_label,communities_wt, 3, float(args.conf))
    print("Communities clustering | Number of rules before filtering = " + str(pd.concat(regles_communities_wt).shape[0]))
    regles_communities_wt = interestingness_measure_community(regles_communities_wt,one_hot_label,communities_wt)
    regles_communities_wt = delete_redundant_community(regles_communities_wt)
    print("Communities clustering | Number of rules after redundancy filter = " + str(pd.concat(regles_communities_wt).shape[0]))
    regles_wt = create_rules_df_community(regles_communities_wt, float(args.int))
    print("Communities clustering | Number of rules after interestingness filter = " + str(pd.concat(regles_communities_wt).shape[0]))
    # print("Communities clustering | Number of rules = " + str(pd.concat(regles_communities_wt).shape[0]))

    for i in range(len(regles_wt)):
        regles_wt[i]['cluster'] = 'wt' + "_community" + str(i+1)
        
    all_rules_wt = pd.concat(regles_wt)
    #Number of rules
    print("Communities clustering | Number of rules = " + str(all_rules_wt.shape[0]))

    return all_rules_wt

def rulesClustering(one_hot_label, groupe, index, new_cluster, index_of_cluster):
    regles_fp_clustering = fp_growth_with_clustering(one_hot_label, groupe, index, 3, float(args.conf))

    print("Clustering | Number of rules before filtering = " + str(pd.concat(regles_fp_clustering).shape[0]))

    ### POST_PROCESSING ###

    regles_fp_clustering = interestingness_measure_clustering(regles_fp_clustering, one_hot_label, groupe, index)
    regles_fp_clustering = delete_redundant_clustering(regles_fp_clustering)
    print("Clustering | Number of rules after redundancy filter = " + str(pd.concat(regles_fp_clustering).shape[0]))
    regles_clustering = create_rules_df_clustering(regles_fp_clustering, float(args.int))
    print("Clustering | Number of rules after interestingness filter = " + str(pd.concat(regles_fp_clustering).shape[0]))

    ### ASSOCIER CHAQUE REGLE AU CLUSTER ###

    regles_clustering_final = pd.DataFrame()
    for i in range(len(regles_clustering)):
        regles_clustering[i]['cluster'] = "clust" + str(i+1)
        regles_clustering_final = regles_clustering_final.append(regles_clustering[i])

    #Number of rules
    print("Clustering | Number of rules = " + str(regles_clustering_final.shape[0]))

    # regles_clustering_final.head()
    return regles_clustering_final


def rulesNewCluter(one_hot_label, new_cluster, index_of_cluster):
    ### SI ON A REPETE LE CLUSTERING POUR DIMINUER LE NOMBRE D'ARTICLES DANS CERTAINES CLASSES ALORS ON APPLIQUE SUR CES NOUVELLES CLASSE ###

    regles_fp_clustering_reclust = []
    for i in range(len(new_cluster)) :
        if(len(new_cluster[i][0]) != 0):
            rules = fp_growth_with_clustering(one_hot_label, new_cluster[i][1], new_cluster[i][2], 4, float(args.conf))
            print("Clustering " + str(i) + " | Number of rules = " + str(pd.concat(rules).shape[0]))
            rules = interestingness_measure_clustering(rules,one_hot_label,new_cluster[i][1],new_cluster[i][2])
            rules = delete_redundant_clustering(rules)
            print("Clustering " + str(i) + " | Number of rules after redundancy filter = " + str(pd.concat(rules).shape[0]))
        else : 
            rules = pd.DataFrame([])
        
        regles_fp_clustering_reclust.append(rules)


    ### POST PROCESSING ###
    regles_reclustering = []
    for i in range(len(regles_fp_clustering_reclust)) : 
        regles_reclustering.append(create_rules_df_clustering(regles_fp_clustering_reclust[i], float(args.int)))
        print("Clustering | Post-processing step " + str(i))
     
    ### ASSOCIER REGLES AU CLUSTER -> Attention ici on a deux cluster : ###
    ### celui trouvé en premier puis celui trouvé en réappliquant la clusterisation ###
    ### (e.g : clust1_clust1 + clust1_clust2 + clust1_clust3) ### 

    regles_reclustering_final = []
    for i in range(len(regles_reclustering)):
        if(len(regles_reclustering[i])!=0):
            for j in range(len(regles_reclustering[i])):
                regles_reclustering[i][j]['cluster'] =  "_clust" + str(index_of_cluster[i]+1) + "_clust" + str(j+1)
                regles_reclustering_final.append(regles_reclustering[i][j])

    return pd.concat(regles_reclustering_final)

def listToString(df):
    # transform lists into strings to use in drop_duplicates
    df['antecedents'] = [','.join(map(str, l)) for l in df['antecedents']]
    df['consequents'] = [','.join(map(str, l)) for l in df['consequents']]
    df['source'] = [','.join(map(str, l)) for l in df['source']]
    df['target'] = [','.join(map(str, l)) for l in df['target']]

def stringToList(df):
    df['antecedents'] = [ x.split(',') for x in df['antecedents'] ]
    df['consequents'] = [ x.split(',') for x in df['consequents']]
    df['source'] = [ x.split(',') for x in df['source']]
    df['target'] = [ x.split(',') for x in df['target']]

def combineClusterRules(regles_clustering_final, regles_reclustering_final):
    ### REGROUPEMENT DE TOUTES LES REGLES DES CLUSTERS  + SUPPRESSION SI MEME REGLE DANS PLUSIEURS CLUSTERS###

    rules_clustering = regles_clustering_final.append(regles_reclustering_final)
    rules_clustering.reset_index(inplace=True, drop=True)
    print("Clustering | Total number of rules = " + str(rules_clustering.shape[0]))
    
    # transform lists into strings to use in drop_duplicates
    listToString(rules_clustering)
    
    # remove duplicates, keeping only the duplicate with highest confidence
    rules_clustering = rules_clustering.sort_values('confidence').drop_duplicates(subset=['antecedents', 'consequents'], keep='last').sort_index()

    # transform strings back into lists for exporting
    stringToList(rules_clustering)

    print("Clustering | Total number of rules after duplicate filter = " + str(rules_clustering.shape[0]))

    return rules_clustering


# Application à Community detection + Clustering (on regroupe article et label)
def rulesCommunityCluster(one_hot, communities_wt):
    all_rules_clustering_wt =  rules_clustering_communities_autoenconder(one_hot, communities_wt, 20, "cosine", 3, float(args.conf), float(args.int))
    # all_rules_clustering_wt = remove_identical_rules(all_rules_clustering_wt)

    # transform lists into strings to use in drop_duplicates
    listToString(all_rules_clustering_wt)
    
    # remove duplicates, keeping only the duplicate with highest confidence
    all_rules_clustering_wt = all_rules_clustering_wt.sort_values('confidence').drop_duplicates(subset=['antecedents', 'consequents'], keep='last').sort_index()

    # transform strings back into lists for exporting
    stringToList(all_rules_clustering_wt)

    print("Clustering article/label | Number of rules = " + str(all_rules_clustering_wt.shape[0]))

    return all_rules_clustering_wt

def fileName(cluster):
    if (args.filename) :
        return args.filename + '_' + cluster + '.json'

    graph = "_" + args.graph if  args.graph else ""
    lang = "_" + args.lang if args.lang else ""
    dataset = '_' + args.endpoint if args.endpoint else ""
    return "data/rules" + dataset + graph + lang + '_' + cluster + '.json'

def exportRules(rules_df, cluster):
    if (args.lang):
        rules_df['lang'] = args.lang
    if (args.graph):
        rules_df['graph'] = args.graph

    print(rules_df)
    rules_df['source'] = rules_df['antecedents']
    rules_df['target'] = rules_df['consequents']

    # rules_df.to_csv(header=True, index=False, path_or_buf=filename, sep=';', mode = mode)
    rules_df.to_json(path_or_buf=fileName(cluster), orient='records')
    
    

if __name__ == '__main__':
    with app.app_context():

        print ('Running algorithm with parameters:')
        print ('SPARQL endpoint = ' + ('Not informed' if args.endpoint == None else queriesData['endpoints'][args.endpoint] + ' (' + args.endpoint + ')'))
        print ('Graph = ' + str(args.graph))
        print ('Language = ' + str(args.lang))
        print ('Minimum confidence = ' + str(args.conf))
        print ('Minimum interestingness = ' + str(args.int))
        print ('Minimum occurrence = ' + str(args.occurrence))
        print ('Input data path = ', str(args.input))
        print ('Output data file = ', str(args.filename))

        if (args.input != None):
            df_total = pd.read_csv(args.input)
        else: 
            ## retrieve the data from SPARQL endpoint 
            df_total = query()

        print(df_total.shape[0])

        ### PREPARATION DES DONNEES : les articles avec un nombre d'entités nommées > 1, on trie par article, entités nommées en minuscule, etc. ###
        
        df_article_sort = transform_data(df_total, int(args.occurrence))

        print("Number of unique items (articles) : " + str(len(df_article_sort['article'].unique())))
        print("Number of unique labels (e.g. named entities) : " + str(len(df_article_sort['label'].unique())))

        matrix_one_hot = getMatrixCooccurrences(df_article_sort)
        encoded_data = applyAutoencoder(matrix_one_hot)

        rules_no_clustering = rulesNoClustering(matrix_one_hot)

        exportRules(rules_no_clustering, 'no_cluster')

        communities_wt = applyWalkTrap(matrix_one_hot)
        rules_communities = rulesCommunities(matrix_one_hot, communities_wt)

        exportRules(rules_no_clustering, 'communities')
        
        ## generate clusters from labels
        groupe, new_cluster, index, index_of_cluster = clusteringCAH(encoded_data)
        ## generate rules from clusters
        rules_clustering = rulesClustering(matrix_one_hot, groupe, index, new_cluster, index_of_cluster)
        exportRules(rules_clustering, 'clustering')

        ## find sub-clusters, if any, and generate rules from them
        rules_reclustering = rulesNewCluter(matrix_one_hot, new_cluster, index_of_cluster)
        exportRules(rules_reclustering, 'reclustering')

        ## combine all rules generated from clustering and remove duplicates (possible rules find in several clusters), keeping only the most relevant
        rules_clustering_total = combineClusterRules(rules_clustering, rules_reclustering)
        exportRules(rules_clustering_total, 'clustering_final')

        all_rules_clustering_wt = rulesCommunityCluster(matrix_one_hot, communities_wt)

        exportRules(all_rules_clustering_wt, 'communities_clustering')

        all_rules = rules_no_clustering.append(rules_clustering_total).append(rules_communities).append(all_rules_clustering_wt)
        all_rules.reset_index(inplace=True, drop=True)
        
        print('All rules | Number of rules = ', str(all_rules.shape[0]))
        listToString(all_rules)
        all_rules = all_rules.drop_duplicates(subset=['antecedents', 'consequents', 'isSymmetric'])
        stringToList(all_rules)

        print('All rules | Number of rules after symmetric duplicate filter = ', str(all_rules.shape[0]))
        exportRules(all_rules, 'all_rules')
        

        filename = 'data/config_' + args.endpoint + '.json'
        # verify if config file exists before
        if (exists(filename)):
            config = pd.read_json(filename)
        else:
            config = {
                "lang": [],
                "graph": [],
                "min_interestingness": float(args.int),
                "min_confidence": float(args.conf),
                "methods": [
                    {"label": "No clustering method", "key": "no_clustering"},
                    {"label": "Clusters of labels", "key": "clust_"},
                    {"label": "Communities of articles", "key": "wt_community"},
                    {"label": "Combination of clusters and communities", "key": "communities"}
                ]
            }

        config['lang'].append(args.lang)
        config['graph'].append(args.graph)
        
        with open(filename, "w") as outfile:
            json.dump(config, outfile, indent=4, sort_keys=False)
        

        
        