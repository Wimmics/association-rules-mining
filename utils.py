import pandas as pd

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import SPARQLWrapper
import json
from SPARQLWrapper import SPARQLWrapper, JSON


import networkx as nx
# from cdlib import algorithms,viz

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


# from stellargraph.data import BiasedRandomWalk
# from stellargraph import StellarGraph


# from gensim.models import Word2Vec


def sparql_service_to_dataframe(service, query):
    """
    Helper function to convert SPARQL results into a Pandas DataFrame.

    Credit to Ted Lawless https://lawlesst.github.io/notebook/sparql-dataframe.html
    """
    sparql = SPARQLWrapper(service)
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    result = sparql.query()

    processed_results = json.load(result.response)
    cols = processed_results['head']['vars']

    out = []
    for row in processed_results['results']['bindings']:
        item = []
        for c in cols:
            item.append(row.get(c, {}).get('value'))
        out.append(item)

    return pd.DataFrame(out, columns=cols)


def delete_Label_number(label):
    delete = False
    k = 0
    for i in label:
        if (i.isdigit()):
            k = k + 1
    if (k == len(label)):
        delete = True

    return delete


def transform_data(df, min_occur):
    """

    Pre-processing data :
        Keep articles with more than 1 named entity
        Sort data by article
        Labels in lower case
        Remove "," and "."
        Delete Labels containing only numbers
        Keep Labels occurring more than 4 times

    Parameters :
        df : DataFrame

    Returns :
        df_article_sort : DataFrame
    """

    url_article = df['article'].value_counts()[df['article'].value_counts() > 1].index
    df_article = df.loc[df['article'].isin(url_article)]
    df_article_sort = df_article.sort_values(by=['article'])

    df_article_sort = df_article_sort.astype({"article": str, "label": str})# , "year": str})
    #df_article_sort['Label'] = df_article_sort['Label'].apply(lambda x: x.lower())
    df_article_sort['label'] = df_article_sort['label'].apply(lambda x: x.replace('.', '').replace(',', ''))
    df_article_sort = df_article_sort.drop(df_article_sort[df_article_sort['label']. \
                                           apply(lambda x: delete_Label_number(x)) == True].index)
    df_article_sort = df_article_sort.loc[df_article_sort['label'].
        isin(list(df_article_sort['label'].value_counts(sort=True)
                  [df_article_sort['label'].value_counts(sort=True) > min_occur].index))]
    df_article_sort = df_article_sort.loc[~df_article_sort['label'].
        isin(list(df_article_sort['label'].value_counts(sort=True).head(15).index))]

    df_article_sort["label"] = df_article_sort["label"].astype('category')
    # df_article_sort["year"] = df_article_sort["year"].astype('category')
    #df_article_sort["body"] = df_article_sort["body"].astype('category')

    return df_article_sort


def coocc_matrix_Label(one_hot_label):
    """
     Create Labels co-occurencies matrix

     Parameters :
        one_hot_label : DataFrame
            One hot encoding DataFrame with only labels

    Returns :
        coocc : DataFrame
            The co-occurencies matrix
    """

    coocc = scipy.linalg.blas.dgemm(alpha=1.0, a=one_hot_label.T, b=one_hot_label.T, trans_b=True)
    np.fill_diagonal(coocc, 0)  # replace the diagonal by 0

    return coocc


def elbow_method(one_hot, nb_max_cluster, metric):
    """
    Use the elbow method and a rule (having at least 2 groups with more than 50 articles) to determine the number of clusters

    Return the number of clusters (k), a dataframe assigning a group for each article and the group index with more than 50 articles
    """

    Z = linkage(one_hot, method='complete', metric=metric)  # sokalmichener
    last = Z[-(nb_max_cluster + 2):, 2]
    acceleration = np.diff(last, 2)
    k = acceleration.argmax() + 2

    groupes_cah = fcluster(Z, t=k, criterion='maxclust')
    idg = np.argsort(groupes_cah)
    groupe = pd.DataFrame(one_hot.index[idg], groupes_cah[idg])
    groupe.columns = ['article']
    index = groupe.groupby([groupe.index]).count().index[groupe.groupby([groupe.index]).count()['article'] > 50]

    while ((groupe.groupby([groupe.index]).count() > 50).sum()['article'] < 2):
        acceleration[acceleration.argmax()] = 0
        k = acceleration.argmax() + 2
        groupes_cah = fcluster(Z, t=k, criterion='maxclust')
        idg = np.argsort(groupes_cah)
        groupe = pd.DataFrame(one_hot.index[idg], groupes_cah[idg])
        groupe.columns = ['article']
        index = groupe.groupby([groupe.index]).count().index[groupe.groupby([groupe.index]).count()['article'] > 50]
        if ((acceleration > 0).sum() == 0):
            k = 0
            groupe = groupe[0:0]
            index = []
            break

    return k, groupe, index


def repeat_cluster(one_hot, group, index_cluster, nb_max_article, nb_cluster):
    """

    Repeat elbow method for each group contaning more than nb_max_article.

    """

    count = group[group.index.isin(index_cluster)].groupby([group[group.index.isin(index_cluster)].index]).count()
    count.reset_index(inplace=True, drop=True)
    index_for_new_cluster = count[count['article'] >= nb_max_article].index
    new_cluster = []

    for i in index_for_new_cluster:
        one_hot_reclust = one_hot[one_hot.index.isin(group[group.index == index_cluster[i]]['article'])]
        nb_cluster_reclust, groupe_reclust, index_reclust = elbow_method(one_hot_reclust, nb_cluster, "cosine")

        if (len(index_reclust) != 0):
            new_cluster.append([one_hot_reclust, groupe_reclust, index_reclust])
        else:
            new_cluster.append([pd.DataFrame([]), pd.DataFrame([]), pd.DataFrame([])])

    return new_cluster, index_for_new_cluster


def fp_growth(one_hot, max_len, min_confidence):
    """

    Apply FP-Growth and generate rules with parameter maximum lenght and minimum confidence
    minimum support is computes in order to have at least 5 articles.

    """
# / one_hot.shape[0]
    print(one_hot.shape[0])
    frequent_itemsets_fp = fpgrowth(one_hot, min_support=5 / one_hot.shape[0], max_len=max_len, use_colnames=True)
    regles_fp = association_rules(frequent_itemsets_fp, metric="confidence", min_threshold=min_confidence).sort_values(
        by='lift', ascending=False)

    return regles_fp



def fp_growth_with_clustering(one_hot, group, index, max_len, min_confidence):
    """
    Apply FP-Growth algorithm and generate rules to each cluster.

    Parameters :
        one_hot : DataFrame
            One hot encoding DataFrame (rows : articles, columns : labels)
        group : DataFrame
            A DataFrame assigning a group for each article
        index : list of int
            Index of group
        max_len : int
            Maximum length of rules (e.g : 3 -> 2 antecedents and 1 consequents or 1 antecedents and 2 consequents)
        min_confidence : float
            Minimum confidence i.e the probability to have B when A occurs.

    Returns :
        regles_fp_clustering : list of DataFrame
            List of Association rules DataFrame
    """

    regles_fp_clustering = []

    for i in index:
        one_hot_cluster = one_hot[one_hot.index.isin(list(group[group.index == i]['article']))]
        frequent_itemsets_fp = fpgrowth(one_hot_cluster,
                                        min_support=5/one_hot_cluster.shape[0], max_len=max_len,use_colnames=True)
        if(len(frequent_itemsets_fp)!=0):
            regles_fp_clustering.append(association_rules(frequent_itemsets_fp, metric="confidence",
                                        min_threshold=min_confidence).sort_values(by='lift',ascending=False))
        else:
            regles_fp_clustering.append(pd.DataFrame([]))
    return regles_fp_clustering


def fp_growth_with_community(one_hot, communities, max_len, min_confidence):
    """
    Apply FP-Growth algorithm and generate rules for selected clusters
    index = the groups with more than 50 articles (see elbow method)
    """

    regles_fp_clustering = []

    for i in range(len(communities)):
        one_hot_cluster = one_hot.T[one_hot.columns.isin(communities[i])].T
        frequent_itemsets_fp = fpgrowth(one_hot_cluster, min_support=5 / one_hot_cluster.shape[0], max_len=max_len,
                                        use_colnames=True)
        if (len(frequent_itemsets_fp) != 0):
            regles_fp_clustering.append(
                association_rules(frequent_itemsets_fp, metric="confidence", min_threshold=min_confidence).sort_values(
                    by='lift', ascending=False))
        else:
            regles_fp_clustering.append(pd.DataFrame([]))
    return regles_fp_clustering


def isSymmetric(rule1, rule2):
    """
    Check if a rule is symmetric
    """

    isSymmetric = False

    if ((rule1.antecedents == rule2.consequents) & (rule1.consequents == rule2.antecedents)):
        isSymmetric = True

    return isSymmetric


def findSymmetric(x,rules):
    """
    Find a symmetric of x among the rules
    """
    for y in rules.itertuples() :
        if(isSymmetric(x,y)) :
            x['isSymmetric'] = True
            break
    return x


def interestingness_measure(regles_fp, one_hot):
    """
    Compute a measure of the interestingness
    """

    size = one_hot.shape[0]
    regles_fp['interestingness'] = ((regles_fp['support'] ** 2) /
                                    (regles_fp['antecedent support'] * regles_fp['consequent support'])) * (
                                               1 - (regles_fp['support'] / size))

    return regles_fp


def interestingness_measure_clustering(regles_fp_clustering, one_hot, group, index):
    """
    Apply interestingness_measure to each cluster
    """
    i = 0
    regles_fp_clustering_new = []
    for rules in regles_fp_clustering:
        if (rules.shape[0] != 0):
            one_hot_group = one_hot[one_hot.index.isin(group[group.index == index[i]]['article'])]
            rules = interestingness_measure(rules, one_hot_group)
            regles_fp_clustering_new.append(rules)
        else:
            regles_fp_clustering_new.append(pd.DataFrame([]))
        i = i + 1
    return regles_fp_clustering_new


def interestingness_measure_community(regles_fp_clustering, one_hot, communities):
    """
    Apply interestingness_measure to each cluster
    """
    i = 0
    regles_fp_clustering_new = []
    for rules in regles_fp_clustering:
        if (rules.shape[0] != 0):
            one_hot_group = one_hot.T[one_hot.columns.isin(communities[i])].T
            rules = interestingness_measure(rules, one_hot_group)
            regles_fp_clustering_new.append(rules)
        else:
            regles_fp_clustering_new.append(pd.DataFrame([]))
        i = i + 1
    return regles_fp_clustering_new


def create_rules_df(regles_fp, interestingness):
    """
    Create the final rules dataframe by keeping rules with a value of interestingness grater than a threshold
    and finding symmetric rules.
    """

    rules = regles_fp.loc[:, ['antecedents', 'consequents', 'confidence', 'interestingness', 'support']]
    rules = rules[rules['interestingness'] >= interestingness]
    rules.reset_index(inplace=True, drop=True)
    rules['isSymmetric'] = False
    rules = rules.apply(lambda x: findSymmetric(x, rules), axis=1)

    return rules


def create_rules_df_clustering(regles_fp_clustering, interestingness):
    """
    Apply create_rules_df to each cluster
    """

    rules_clustering = []
    for rules in regles_fp_clustering:
        if (len(rules) != 0):
            rules = create_rules_df(rules, interestingness)
            rules_clustering.append(rules)
        else:
            rules_clustering.append(pd.DataFrame([]))

    return rules_clustering


def create_rules_df_community(regles_fp_clustering, interestingness):
    """
    Apply create_rules_df to each cluster
    """

    rules_clustering = []
    for rules in regles_fp_clustering:
        if (len(rules) != 0):
            rules = create_rules_df(rules, interestingness)
            rules_clustering.append(rules)
        else:
            rules_clustering.append(pd.DataFrame([]))

    return rules_clustering


def delete_redundant(rules):
    """
    Delete redundant rules. A rule is redundant if there is a subset of this rule with the same or higher confidence.

    (A,B,C) -> D is redundant if (A,B) -> D has the same or higher confidence.

    """

    redundant = []
    for i in rules.itertuples():
        for j in rules.itertuples():
            if (((i.antecedents.issubset(j.antecedents))
                 and (i.consequents == j.consequents)
                 and (i.confidence >= j.confidence)
                 and (i.Index != j.Index)) or ((i.consequents.issubset(j.consequents))
                                               and (i.antecedents == j.antecedents)
                                               and (i.confidence >= j.confidence)
                                               and (i.Index != j.Index))):
                redundant.append(j.Index)

    redundant = list(dict.fromkeys(redundant))
    rules = rules.drop(redundant)
    return rules


def delete_redundant_clustering(rules_clustering):
    """
    Apply delete_redundant to each cluster
    """

    rules_without_redundant = []
    for rules in rules_clustering:
        rules = delete_redundant(rules)
        rules_without_redundant.append(rules)
    return rules_without_redundant


def delete_redundant_community(rules_clustering):
    """
    Apply delete_redundant to each cluster
    """

    rules_without_redundant = []
    for rules in rules_clustering:
        rules = delete_redundant(rules)
        rules_without_redundant.append(rules)
    return rules_without_redundant


#Remove identical rules with lower confidence
def remove_identical_rules(rules) :
    index = []
    for x in rules.itertuples() :
        for y in rules.itertuples() :
            print(x, y)
            if((x.Index not in index) and (x.Index != y.Index) and (x.antecedents == y.antecedents) and (x.consequents == y.consequents)) :
                if(x.confidence >= y.confidence) :
                    index.append(y.Index)
                else:
                    index.append(x.Index)
    return rules.drop(index)


def generate_article_rules(test, rules):
    """
    For each article in the test set,  the method checks if labels and pair of labels of the article
    are antecedent in the created rules. If yes, it adds the consequents to the list of new rules.

    Return a list of list of new rules for each article.

    """

    new_rules = []

    for article in test['article'].unique():
        new_rules_article = []
        for i in test[test['article'] == article]['label']:

            if (rules[rules['antecedents'].eq({i})].shape[0] != 0):
                new_rules_article.append(
                    list(rules[rules['antecedents'].eq({i})]['consequents']))

            for j in test[test['article'] == article]['label']:
                if (rules[rules['antecedents'].eq({i, j})].shape[0] != 0):
                    new_rules_article.append(
                        list(rules[rules['antecedents'].eq({i, j})]['consequents']))

        new_rules.append(new_rules_article)

    new_rules_list = []

    for i in range(len(new_rules)):
        rules_i = []
        for j in range(len(new_rules[i])):
            for k in range(len(new_rules[i][j])):
                rules_i.append(list(new_rules[i][j][k])[0])
        new_rules_list.append(list(dict.fromkeys(rules_i)))

    return new_rules_list


def elbow_method_community(one_hot, nb_max_cluster, metric):
    """
    Use the elbow method and a rule (having at least 2 groups with more than 50 articles) to determine the number of clusters

    Return the number of clusters (k), a dataframe assigning a group for each article and the group index with more than 50 articles
    """

    Z = linkage(one_hot, method='complete', metric=metric)  # sokalmichener
    last = Z[-(nb_max_cluster + 2):, 2]
    acceleration = np.diff(last, 2)
    k = acceleration.argmax() + 2

    groupes_cah = fcluster(Z, t=k, criterion='maxclust')
    idg = np.argsort(groupes_cah)
    groupe = pd.DataFrame(one_hot.index[idg], groupes_cah[idg])
    groupe.columns = ['Labels']
    index = groupe.groupby([groupe.index]).count().index[groupe.groupby([groupe.index]).count()['Labels'] > 20]

    while ((groupe.groupby([groupe.index]).count() > 20).sum()['Labels'] < 2):
        acceleration[acceleration.argmax()] = 0
        k = acceleration.argmax() + 2
        groupes_cah = fcluster(Z, t=k, criterion='maxclust')
        idg = np.argsort(groupes_cah)
        groupe = pd.DataFrame(one_hot.index[idg], groupes_cah[idg])
        groupe.columns = ['Labels']
        index = groupe.groupby([groupe.index]).count().index[groupe.groupby([groupe.index]).count()['Labels'] > 20]
        if ((acceleration > 0).sum() == 0):
            k = 0
            groupe = groupe[0:0]
            index = []
            break

    return k, groupe, index


def fp_growth_with_com_auto(one_hot, group, index, max_len, min_confidence):
    """
    Apply FP-Growth algorithm and generate rules to each cluster.

    Parameters :
        one_hot : DataFrame
            One hot encoding DataFrame (rows : articles, columns : labels)
        group : DataFrame
            A DataFrame assigning a group for each article
        index : list of int
            Index of group
        max_len : int
            Maximum length of rules (e.g : 3 -> 2 antecedents and 1 consequents or 1 antecedents and 2 consequents)
        min_confidence : float
            Minimum confidence i.e the probability to have B when A occurs.

    Returns :
        regles_fp_clustering : list of DataFrame
            List of Association rules DataFrame
    """

    regles_fp_clustering = []

    for i in index:
        one_hot_cluster = one_hot.loc[:,one_hot.columns.isin(list(group[group.index == i]['Labels']))]
        frequent_itemsets_fp = fpgrowth(one_hot_cluster,
                                        min_support=5/one_hot_cluster.shape[0], max_len=max_len,use_colnames=True)
        if(len(frequent_itemsets_fp)!=0):
            regles_fp_clustering.append(association_rules(frequent_itemsets_fp, metric="confidence",
                                        min_threshold=min_confidence).sort_values(by='lift',ascending=False))
        else:
            regles_fp_clustering.append(pd.DataFrame([]))
    return regles_fp_clustering


def interestingness_measure_com_auto(regles_fp_clustering, one_hot, group, index):
    """
    Apply interestingness_measure to each cluster
    """
    i = 0
    regles_fp_clustering_new = []
    for rules in regles_fp_clustering:
        if (rules.shape[0] != 0):
            one_hot_group = one_hot.loc[:, one_hot.columns.isin(list(group[group.index == i]['Labels']))]
            rules = interestingness_measure(rules, one_hot_group)
            regles_fp_clustering_new.append(rules)
        else:
            regles_fp_clustering_new.append(pd.DataFrame([]))
        i = i + 1
    return regles_fp_clustering_new


def rules_clustering_communities_autoenconder(one_hot, communities, nb_cluster, metrics,
                                              max_length, min_confidence, interestingness):
    """
    Generate Association rules after applying clustering method to
    one hot encoding matrix with only labels from the same community.

    Parameters :
        one_hot : DataFrame
            One hot encoding DataFrame (rows : articles, columns : labels)
        communities : list of int
            List of communities which each named entities belonged
        nb_cluster : int
            Maximum number of clusters
        metrics : string
            The metric used for the HAC
        nb_min_articles : int
            The minimum number of articles in a cluster
        max_length : int
            Maximum length of rules (e.g : 3 -> 2 antecedents and 1 consequents or 1 antecedents and 2 consequents)
        min_confidence : float
            Minimum confidence i.e the probability to have B when A occurs.
        interestingness: float
            Threshold for the interestingness measure

    Returns :
        all_rules_clustering_communities : DataFrame
            Association rules DataFrame
    """

    all_rules_clustering_communities = pd.DataFrame()

    for i in communities:
        label = [x for x in one_hot.columns if x.startswith('label_')]
        label_drop = [x for x in label if not x in ["label_" + s for s in i]]
        one_hot_cluster = one_hot.drop(label_drop, axis=1)

        input_dim = one_hot.shape[1]
        encoding_dim = 64
        # Number of neurons in each Layer [8, 6, 4, 3, ...] of encoders
        input_layer = Input(shape=(input_dim,))
        encoder_layer_1 = Dense(256, activation="tanh", activity_regularizer=regularizers.l1(10e-5))(input_layer)
        encoder_layer_2 = Dense(128, activation="tanh")(encoder_layer_1)
        encoder_layer_3 = Dense(encoding_dim, activation="tanh")(encoder_layer_2)
        encoder = Model(inputs=input_layer, outputs=encoder_layer_3)
        # Use the model to predict the factors which sum up the information of interest rates.
        encoded_data = pd.DataFrame(encoder.predict(one_hot))
        encoded_data.index = one_hot_cluster.index

        nb_cluster_communities, groupe_communities, index_communities = elbow_method(encoded_data, nb_cluster,
                                                                                     metrics)

        drop = [x for x in one_hot_cluster.columns if not x.startswith('label_')]
        one_hot_cluster = one_hot_cluster.drop(drop, axis=1)
        one_hot_cluster.columns = list(pd.DataFrame(one_hot_cluster.columns)[0].apply(lambda x: x.split('_')[-1]))

        regles_fp_clustering_communities = fp_growth_with_clustering(one_hot_cluster, groupe_communities,
                                                                     index_communities, max_length, min_confidence)
        print("Number of rules : " + str(pd.concat(regles_fp_clustering_communities).shape[0]))
        regles_fp_clustering_communities = interestingness_measure_clustering(regles_fp_clustering_communities,
                                                                              one_hot_cluster, groupe_communities,
                                                                              index_communities)
        regles_fp_clustering_communities = delete_redundant_clustering(regles_fp_clustering_communities)
        regles_clustering_communities = create_rules_df_clustering(regles_fp_clustering_communities, interestingness)

        regles_clustering_communities_final = pd.DataFrame()

        for j in range(len(regles_clustering_communities)):
            regles_clustering_communities[j]['cluster'] = "communities" + str(communities.index(i)) + "_clust" + str(
                j + 1)
            regles_clustering_communities_final = regles_clustering_communities_final.append(
                regles_clustering_communities[j])

        all_rules_clustering_communities = all_rules_clustering_communities.append(regles_clustering_communities_final)

    all_rules_clustering_communities.reset_index(inplace=True, drop=True)

    return all_rules_clustering_communities


def rules_clustering_communities_embedding_autoencoder(one_hot, groupe, index, nb_cluster, metrics,
                                                       max_length, min_confidence, interestingness):
    """
    Generate Association rules after applying clustering method to
    one hot encoding matrix with only labels from the same community.

    Parameters :
        one_hot : DataFrame
            One hot encoding DataFrame (rows : articles, columns : labels)
        communities : list of int
            List of communities which each named entities belonged
        nb_cluster : int
            Maximum number of clusters
        metrics : string
            The metric used for the HAC
        nb_min_articles : int
            The minimum number of articles in a cluster
        max_length : int
            Maximum length of rules (e.g : 3 -> 2 antecedents and 1 consequents or 1 antecedents and 2 consequents)
        min_confidence : float
            Minimum confidence i.e the probability to have B when A occurs.
        interestingness: float
            Threshold for the interestingness measure

    Returns :
        all_rules_clustering_communities : DataFrame
            Association rules DataFrame
    """

    all_rules_clustering_communities = pd.DataFrame()

    for i in index:
        label = [x for x in one_hot.columns if x.startswith('Label_')]
        label_drop = [x for x in label if not x in ["label_" + s for s in list(groupe[groupe.index == i]['Labels'])]]
        one_hot_cluster = one_hot.drop(label_drop, axis=1)
        input_dim = one_hot_cluster.shape[1]
        encoding_dim = 32
        # Number of neurons in each Layer [8, 6, 4, 3, ...] of encoders
        input_layer = Input(shape=(input_dim,))
        encoder_layer_1 = Dense(100, activation="tanh")(input_layer)
        encoder_layer_2 = Dense(encoding_dim, activation="tanh",
                                kernel_regularizer=regularizers.l1_l2(l1=0.001, l2=0.01))(encoder_layer_1)
        encoder = Model(inputs=input_layer, outputs=encoder_layer_2)
        # Use the model to predict the factors which sum up the information of interest rates.
        encoded_data = pd.DataFrame(encoder.predict(one_hot_cluster))
        encoded_data.index = one_hot_cluster.index

        nb_cluster_communities, groupe_communities, index_communities = elbow_method(encoded_data, nb_cluster,
                                                                                     metrics)

        drop = [x for x in one_hot_cluster.columns if not x.startswith('label_')]
        one_hot_cluster = one_hot_cluster.drop(drop, axis=1)
        one_hot_cluster.columns = list(pd.DataFrame(one_hot_cluster.columns)[0].apply(lambda x: x.split('_')[-1]))

        regles_fp_clustering_communities = fp_growth_with_clustering(one_hot_cluster, groupe_communities,
                                                                     index_communities, max_length, min_confidence)
        print("Number of rules : " + str(pd.concat(regles_fp_clustering_communities).shape[0]))
        regles_fp_clustering_communities = interestingness_measure_clustering(regles_fp_clustering_communities,
                                                                              one_hot_cluster, groupe_communities,
                                                                              index_communities)
        regles_fp_clustering_communities = delete_redundant_clustering(regles_fp_clustering_communities)
        regles_clustering_communities = create_rules_df_clustering(regles_fp_clustering_communities, interestingness)

        regles_clustering_communities_final = pd.DataFrame()

        for j in range(len(regles_clustering_communities)):
            regles_clustering_communities[j]['cluster'] = "communities" + str(i + 1) + "_clust" + str(
                j + 1)
            regles_clustering_communities_final = regles_clustering_communities_final.append(
                regles_clustering_communities[j])

        all_rules_clustering_communities = all_rules_clustering_communities.append(regles_clustering_communities_final)

    all_rules_clustering_communities.reset_index(inplace=True, drop=True)

    return all_rules_clustering_communities


def dataframe_difference(df1, df2):
    """Find rows which are equal between two DataFrames."""
    comparison_df = df1.merge(df2,
                              indicator=True,
                              how='outer')
    diff_df = comparison_df[comparison_df['_merge'] == 'both']

    return diff_df.shape[0]


def comparison(rules1,rules2) :
    print("Number of rules 1 : " + str(rules1.shape[0]))
    print("Number of rules 2 : " + str(rules2.shape[0]))
    print("Number of same rows : " + str(dataframe_difference(rules1.loc[:,['antecedents','consequents']],rules2.loc[:,['antecedents','consequents']])))
    print("Number of same rows among top 10 most interesting rules : " + str(dataframe_difference(rules1.sort_values(by=['confidence','interestingness'],ascending=False).loc[:,['antecedents','consequents']].head(10),rules2.sort_values(by=['confidence','interestingness'],ascending=False).loc[:,['antecedents','consequents']].head(10))))
    print("Number of same rows among top 20 most interesting rules : " + str(dataframe_difference(rules1.sort_values(by=['confidence','interestingness'],ascending=False).loc[:,['antecedents','consequents']].head(20),rules2.sort_values(by=['confidence','interestingness'],ascending=False).loc[:,['antecedents','consequents']].head(20))))
