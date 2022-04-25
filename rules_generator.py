import warnings

warnings.filterwarnings('ignore')

# from flask import Flask, request, jsonify, render_template

import pandas as pd

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# import SPARQLWrapper
import json
# from SPARQLWrapper import SPARQLWrapper, JSON

import networkx as nx
from cdlib import algorithms, viz

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

parser = argparse.ArgumentParser()

parser.add_argument('--endpoint', help='The endpoint to query')
parser.add_argument('--graph', help='The graph you want to query', required=False)
parser.add_argument('--lang', help='The language of the labels', required=False)
parser.add_argument('--filepath', help='The path to the output datafile', required=False)
parser.add_argument('--conf', help='The minimal confidence for generating the rules', default=0.7, required=False)
parser.add_argument('--int', help='The minimal interestingness for generating the rules', default=0.3, required=False)

parser.add_argument('--min-entity-freq', help='The minimal frequency of entities for inclusion', type=int, dest="min_entity_freq",
                    default=4, required=False)

args = parser.parse_args()

## load the queries 
with open('queries.json') as f:
    queriesData = json.load(f)


# app = Flask(__name__)

def query_between_dates(year1, year2, length):
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

    print(query)

    complete_query = query % (args.lang, offset) if args.lang else query % (offset)
    df_query = sparql_service_to_dataframe(queriesData['endpoints'][args.endpoint], complete_query)

    ## List with all the request responses ##
    list_total = [df_query]
    ## Get all data by set the offset at each round ##
    while (df_query.shape[0] == 10000 and len(list_total) <= length):
        offset = offset + 10000
        complete_query = query % (args.lang, offset) if args.lang else query % (offset)
        df_query = sparql_service_to_dataframe(queriesData['endpoints'][args.endpoint], complete_query)
        list_total.append(df_query)
    ## Concatenate all the dataframe from the list ##
    df_total = pd.concat(list_total)

    return df_total


# Création de la matrice de co-occurences qui est notre jeu de données pour le clustering 

def getMatrixCooccurrences(train):
    ##One hot encoding train set (5000 by 5000 articles) + Sparse type to reduce the memory

    one_hot = pd.get_dummies(
        train[train['article'].isin(train['article'].unique()[0:5000])].drop_duplicates().set_index('article')).sum(
        level=0).apply(lambda y: y.apply(lambda x: 1 if x >= 1 else 0)).astype("Sparse[int]")

    i = 5000
    while (one_hot.shape[0] < len(train['article'].unique())):
        one_hot = one_hot.append(
            pd.get_dummies(train[train['article'].isin(train['article'].unique()[i:i + 5000])].drop_duplicates(). \
                           set_index('article')).sum(level=0).apply(lambda y: y.apply(lambda x: 1 if x >= 1 else 0)). \
                astype("Sparse[int]"))
        i = i + 5000

    ### Remplacer les NaN par 0 et supprimer les lignes avec que des 0 ###

    one_hot = one_hot.fillna(0)
    one_hot = one_hot.loc[:, (one_hot != 0).any(axis=0)]

    ### Passer la matrice en type Sparse pour accélérer les calculs ###

    one_hot = one_hot.astype("Sparse[int]")

    ## Supprimer les variables "year" si on ne veut pas les prendre en compte dans l'analyse ##

    # drop = [x for x in one_hot.columns if not x.startswith('Label_')]
    # one_hot_label = one_hot.drop(drop,axis=1)
    one_hot.columns = list(pd.DataFrame(one_hot.columns)[0].apply(lambda x: x.split('_')[-1]))

    return one_hot


### Réduction du nombre de variables + Clustering
# L'autoencoder permet de réduire la dimension et de pouvoir appliquer la CAH qui n'est pas robuste face à un nombre trop importants de variables
def applyAutoencoder(one_hot_matrix):
    ### Autoenconder  ###

    input_dim = one_hot_matrix.shape[1]
    encoding_dim = 128
    # Number of neurons in each Layer [8, 6, 4, 3, ...] of encoders
    input_layer = Input(shape=(input_dim,))
    encoder_layer_1 = Dense(2048, activation="tanh")(input_layer)
    encoder_layer_2 = Dense(1024, activation="tanh")(encoder_layer_1)
    encoder_layer_3 = Dense(256, activation="tanh")(encoder_layer_2)
    encoder_layer_4 = Dense(encoding_dim, activation="tanh", kernel_regularizer=regularizers.l1_l2(l1=0.001, l2=0.01))(
        encoder_layer_3)
    encoder = Model(inputs=input_layer, outputs=encoder_layer_4)

    # Use the model to predict the factors which sum up the information of interest rates.
    encoded_data = pd.DataFrame(encoder.predict(one_hot_matrix))

    encoded_data.index = one_hot_matrix.index

    return encoded_data


def clusteringCAH(encoded_data):
    nb_cluster, groupe, index = elbow_method(encoded_data, 10, "cosine")

    groupe[groupe.index.isin(index)].groupby([groupe[groupe.index.isin(index)].index]).count()

    ### Visualisation des labels les plus fréquents dans le groupe 1 ###

    # count = train[train['article'].isin(list(groupe[groupe.index == 1]['article']))]['label'].value_counts()[:20].sort_values(ascending=True)
    # count.plot(kind='barh').set_title('The 15 most frequent named entities')

    # Apply again elbow method to the groups with more than 500 articles #

    new_cluster, index_of_cluster = repeat_cluster(encoded_data, groupe, index, 500, 5)

    return groupe, new_cluster, index, index_of_cluster


### Community alogrithms (Walk Trap)
def applyWalkTrap(one_hot_label):
    ## Co-occurencies matrix
    coooc_s = coocc_matrix_Label(one_hot_label)

    ## Créer des tuples avec co-occurences différentes de 0 ##

    labels = one_hot_label.columns
    tuple_list = []
    for i in range(len(coooc_s)):
        no_zero = np.where(coooc_s[i] != 0)
        for j in no_zero[0]:
            tuple_list.append([labels[i], labels[j], coooc_s[i][j]])

    ## Create Graph ##
    G = nx.Graph()

    for egde in tuple_list:
        G.add_edge(egde[0], egde[1], weight=egde[2])

    com_wt = algorithms.walktrap(G)
    return com_wt.communities


def rulesNoClustering(one_hot_matrix):
    regles_fp = fp_growth(one_hot_matrix, 3, float(args.conf))
    print(regles_fp.head())

    print("No clustering | Number of rules before filtering = " + str(regles_fp.shape[0]))

    ### POST-PROCESSING : interestingness + règles redondantes ###
    regles_fp = interestingness_measure(regles_fp, one_hot_matrix)
    print("No clustering | Number of rules after interestingness filter = " + str(regles_fp.shape[0]))
    regles_fp = delete_redundant(regles_fp)
    print("No clustering | Number of rules after redundancy filter = " + str(regles_fp.shape[0]))
    regles = create_rules_df(regles_fp, float(args.int))

    regles['cluster'] = "no_clustering"

    print("No clustering | Number of rules : " + str(regles.shape[0]))

    return regles


def rulesCommunities(one_hot, communities_wt):
    drop = [x for x in one_hot.columns if not x.startswith('label_')]
    one_hot_community = one_hot.drop(drop, axis=1)
    one_hot_community.columns = list(pd.DataFrame(one_hot_community.columns)[0].apply(lambda x: x.split('_')[-1]))

    regles_communities_wt = fp_growth_with_community(one_hot_community, communities_wt, 3, float(args.conf))
    print(
        "Communities clustering | Number of rules before filtering = " + str(pd.concat(regles_communities_wt).shape[0]))
    regles_communities_wt = interestingness_measure_community(regles_communities_wt, one_hot_community, communities_wt)
    print("Communities clustering | Number of rules after interestingness filter = " + str(
        pd.concat(regles_communities_wt).shape[0]))
    regles_communities_wt = delete_redundant_community(regles_communities_wt)
    print("Communities clustering | Number of rules after redundancy filter = " + str(
        pd.concat(regles_communities_wt).shape[0]))
    regles_wt = create_rules_df_community(regles_communities_wt, float(args.int))
    # print("Communities clustering | Number of rules = " + str(pd.concat(regles_communities_wt).shape[0]))

    for i in range(len(regles_wt)):
        regles_wt[i]['cluster'] = 'wt' + "_community" + str(i + 1)

    all_rules_wt = pd.concat(regles_wt)
    # Number of rules
    print("Communities clustering | Number of rules = " + str(all_rules_wt.shape[0]))

    return all_rules_wt


def rulesClustering(one_hot_label, groupe, index, new_cluster, index_of_cluster):
    regles_fp_clustering = fp_growth_with_clustering(one_hot_label, groupe, index, 3, float(args.conf))

    print("Clustering | Number of rules before filtering = " + str(pd.concat(regles_fp_clustering).shape[0]))

    ### POST_PROCESSING ###

    regles_fp_clustering = interestingness_measure_clustering(regles_fp_clustering, one_hot_label, groupe, index)
    print(
        "Clustering | Number of rules after interestingness filter = " + str(pd.concat(regles_fp_clustering).shape[0]))
    regles_fp_clustering = delete_redundant_clustering(regles_fp_clustering)
    print("Clustering | Number of rules after redundancy filter = " + str(pd.concat(regles_fp_clustering).shape[0]))
    regles_clustering = create_rules_df_clustering(regles_fp_clustering, float(args.int))

    ### ASSOCIER CHAQUE REGLE AU CLUSTER ###

    regles_clustering_final = pd.DataFrame()
    for i in range(len(regles_clustering)):
        regles_clustering[i]['cluster'] = "clust" + str(i + 1)
        regles_clustering_final = regles_clustering_final.append(regles_clustering[i])

    # Number of rules
    print("Clustering | Number of rules = " + str(regles_clustering_final.shape[0]))

    regles_clustering_final.head()

    ### SI ON A REPETE LE CLUSTERING POUR DIMINUER LE NOMBRE D'ARTICLES DANS CERTAINES CLASSES ALORS ON APPLIQUE SUR CES NOUVELLES CLASSE ###

    regles_fp_clustering_reclust = []
    for i in range(len(new_cluster)):
        if (len(new_cluster[i][0]) != 0):
            rules = fp_growth_with_clustering(one_hot_label, new_cluster[i][1], new_cluster[i][2], 4, 0.7)
            rules = interestingness_measure_clustering(rules, one_hot_label, new_cluster[i][1], new_cluster[i][2])
            rules = delete_redundant_clustering(rules)
        else:
            rules = pd.DataFrame([])

        regles_fp_clustering_reclust.append(rules)

    ### POST PROCESSING ###
    regles_reclustering = []
    for i in range(len(regles_fp_clustering_reclust)):
        regles_reclustering.append(create_rules_df_clustering(regles_fp_clustering_reclust[i], 0.3))

    ### ASSOCIER REGLES AU CLUSTER -> Attention ici on a deux cluster : ###
    ### celui trouvé en premier puis celui trouvé en réappliquant la clusterisation ###
    ### (e.g : clust1_clust1 + clust1_clust2 + clust1_clust3) ### 

    regles_reclustering_final = []
    for i in range(len(regles_reclustering)):
        if (len(regles_reclustering[i]) != 0):
            for j in range(len(regles_reclustering[i])):
                regles_reclustering[i][j]['cluster'] = "_clust" + str(index_of_cluster[i] + 1) + "_clust" + str(j + 1)
                regles_reclustering_final.append(regles_reclustering[i][j])

    ### REGROUPEMENT DE TOUTES LES REGLES DES CLUSTERS  + SUPPRESSION SI MEME REGLE DANS PLUSIEURS CLUSTERS###

    rules_clustering = regles_clustering_final.append(regles_reclustering_final)
    rules_clustering.reset_index(inplace=True, drop=True)
    rules_clustering = remove_identical_rules(rules_clustering)

    print("Clustering | Number of rules after repeating clustering (if applicable) = " + str(rules_clustering.shape[0]))

    return rules_clustering


# Application à Community detection + Clustering (on regroupe article et label)
def rulesCommunityCluster(one_hot, communities_wt):
    all_rules_clustering_wt = rules_clustering_communities_autoenconder(one_hot, communities_wt, 20, "cosine", 3,
                                                                        float(args.conf), float(args.int))
    all_rules_clustering_wt = remove_identical_rules(all_rules_clustering_wt)

    print("Clustering article/label | Number of rules = " + str(all_rules_clustering_wt.shape[0]))

    return all_rules_clustering_wt


def exportRules(rules_df):
    # all_rules = regles.append(rules_clustering).append(all_rules_wt).append(all_rules_clustering_wt)

    # Export rules as csv
    # rules_df['antecedents-list'] = rules_df['antecedents'].apply(lambda x : list(x))
    # rules_df['consequents-list'] = rules_df['consequents'].apply(lambda x : list(x))

    graph = "_" + args.graph if args.graph else "_"
    lang = "_" + args.lang if args.lang else ""
    PATH_RULES = "data/rules_" + args.endpoint + graph + lang + ".json"

    if (args.lang):
        rules_df['lang'] = args.lang
    if (args.graph):
        rules_df['graph'] = args.graph

    rules_df['source'] = rules_df['antecedents']
    rules_df['target'] = rules_df['consequents']

    # rules_df.to_csv(header=True,index=False,path_or_buf=PATH_RULES,sep=';')
    rules_df.to_json(path_or_buf=PATH_RULES, orient='records')


if __name__ == '__main__':
    #### Récupération de 30 000 articles (length=2) publiés entre 2019 et 2020

    print('Running algorithm with parameters:')
    print('Endpoint = ' + queriesData['endpoints'][args.endpoint] + ' (' + args.endpoint + ')')
    print('Graph = ' + args.graph)
    print('Language = ' + args.lang)
    print('Minimal confidence = ' + str(args.conf))
    print('Minimal interestingness = ' + str(args.int))

    df_total = query_between_dates(2019, 2020, 100)

    print(df_total.head())

    # count = df_total['Label'].value_counts()[:20].sort_values(ascending=True)
    # count.plot(kind='barh').set_title('The 15 most frequent named entities')

    ### PREPARATION DES DONNEES : les articles avec un nombre d'entités nommées > 1, on trie par article, entités nommées en minuscule, etc. ###

    df_article_sort = transform_data(df_total, entity_count_threshold=int(args.min_entity_freq))

    print("Nombre d'articles uniques : " + str(len(df_article_sort['article'].unique())))
    print("Nombre de label uniques : " + str(len(df_article_sort['label'].unique())))

    # ### Découpage en TRAIN/TEST des données ###

    # # train_index, test_index  = train_test_split(df_article_sort[['article','year']].drop_duplicates(), test_size=0.2) #,stratify=df_article_sort[['article','year']].drop_duplicates()['year']
    # # train = df_article_sort[df_article_sort['article'].isin(train_index['article'].unique() )]
    # # test = df_article_sort[df_article_sort['article'].isin(test_index['article'].unique() )]

    # ### METTRE TOUT EN string + spécifier que Label et year sont des catégories pour le one-hot-encoding ###
    df_article_sort[['label']].drop_duplicates()

    train = df_article_sort.astype({"article": str, "label": str})
    # #  "year":str})
    train["label"] = train["label"].astype('category')
    # # train["year"] = train["year"].astype('category')

    # # test = test.astype({"article" : str, "Label":str, "year":str})
    # # test["Label"] = test["Label"].astype('category')
    # # test["year"] = test["year"].astype('category')

    matrix_one_hot = getMatrixCooccurrences(train)
    encoded_data = applyAutoencoder(matrix_one_hot)
    rules_no_clustering = rulesNoClustering(matrix_one_hot)

    communities_wt = applyWalkTrap(matrix_one_hot)
    rules_communities = rulesCommunities(matrix_one_hot, communities_wt)

    groupe, new_cluster, index, index_of_cluster = clusteringCAH(encoded_data)
    rules_clustering = rulesClustering(matrix_one_hot, groupe, index, new_cluster, index_of_cluster)

    all_rules_clustering_wt = rulesCommunityCluster(matrix_one_hot, communities_wt)

    all_rules = rules_no_clustering.append(rules_clustering).append(rules_communities).append(all_rules_clustering_wt)
    print('Number total of rules = ' + str(all_rules.shape[0]))
    exportRules(all_rules)

    # print(rules_no_clustering.head(10))
