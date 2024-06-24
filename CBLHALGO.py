import numpy as np
from sklearn.neighbors import NearestNeighbors, KDTree

import sys
sys.setrecursionlimit(100000)
from collections import defaultdict
from DPoint import DPoint
from FindConnectedPoints import expandAroundGobalHotSpot
import  pandas as pd
ind_to_name = {}
name_to_label = {}
name_to_dPoint = {}
label_counter = 1
data_matrix = None
label_dic = defaultdict(set)

def initClustering(min_cluster_size):
    global label_counter
    X = index_to_X()

    epsilon = search_epsilon(min_cluster_size, X)
    if (epsilon == None):
        for key, item in name_to_dPoint.items():
            name = item.name
            name_to_label[name] = label_counter
            label_dic[label_counter].add(name)
        return collectLabels()
    rearrange_name(X)
    globalhotspot = globalhotspotFind()
    clustering_around_globalhotspot = expandAroundGobalHotSpot(globalhotspot, ind_to_name, name_to_dPoint)
    # print('end', len(clustering_around_globalhotspot.neighbours), X.shape)
    for ind in clustering_around_globalhotspot.neighbours:
        tmp_name = ind_to_name[ind]
        name_to_label[tmp_name] = label_counter
        label_dic[label_counter].add(tmp_name)
        del name_to_dPoint[tmp_name]
    label_counter += 1
    size_item = len(name_to_dPoint.items())
    if size_item < min_cluster_size:
        for key, item in name_to_dPoint.items():
            name = item.name
            name_to_label[name] = label_counter
            label_dic[label_counter].add(tmp_name)
        return collectLabels()
    else:
        return initClustering(min_cluster_size)


def str_to_name(dPoint):
    name = ""
    for item in dPoint:
        name += str(item) + "x"
    return name


def init_2(X):
    for idx, item in enumerate(X):
        tmp_name = str_to_name(item)
        if tmp_name not in name_to_dPoint:
            d2Punkt = DPoint(item, tmp_name, idx)
            name_to_dPoint[tmp_name] = d2Punkt
    # print(count)


def init(X):
    global data_matrix
    data_matrix = X
    count = 0
    for idx, item in enumerate(X):
        tmp_name = str_to_name(item)
        if tmp_name not in name_to_dPoint:
            count += 1
            d2Punkt = DPoint(item, tmp_name, idx)
            name_to_dPoint[tmp_name] = d2Punkt
    # print(count)


def index_to_X():
    list = []
    for key, item in name_to_dPoint.items():
        list.append(item.coordinates)
    return np.array(list)


def search_epsilon(min_cluster_size, X):
    #('start', X.shape)
    global name_to_dPoint
    if min_cluster_size < X.shape[0]:
        factor = 1
        metricNN = NearestNeighbors(n_neighbors=min_cluster_size * factor + 1, leaf_size=min_cluster_size * factor + 1,
                                    n_jobs=-1).fit(X)
        distances, indices = metricNN.kneighbors(X)


        name_dic = {}
        epsilon = distances[0][min_cluster_size]


        for i, distance in enumerate(distances):

            # if i == iiii or i== vvv:
            #     for n___ in indices[i]:
            #         if i != n___ and n___ != vvv:
            #          plt.scatter(X[n___][0], X[n___][1], c= pal[2], s= 100)
            # if i==iiii:
            #    plt.scatter(X[n___][0], X[n___][1], c=pal[5], s = 30)
            tmp_epsilon = distance[min_cluster_size]
            tmp_distance = 0
            for dis in distance:
                tmp_distance += dis
            name = str_to_name(X[i])
            name_dic[i] = name
            name_to_dPoint[name].avg_k_distance = tmp_distance / min_cluster_size
            name_to_dPoint[name].neighbours = indices[i]
            if epsilon > tmp_epsilon:
                epsilon = tmp_epsilon
        for i in range(X.shape[0]):
            name = name_dic[i]
            for item, dist_n in zip(indices[i], distances[i]):
                name_neigh = name_dic[item]
                name_to_dPoint[name_neigh].rknn.add(i)
                name_to_dPoint[name_neigh].rknn_dist += dist_n

        li_peak  = []
        for  i in range(X.shape[0]):

            name = name_dic[i]
            cur_point = name_to_dPoint[name]
            cur_point.symmetric_knn = set([])

            for nn in cur_point.neighbours:
                if nn in cur_point.rknn:
                    cur_point.symmetric_knn.add(nn)
            #print(len(name_to_dPoint[name].rknn))
            if len(name_to_dPoint[name].rknn) > 0 and   name_to_dPoint[name].rknn_dist > 0:
               name_to_dPoint[name].rknn_dist =  name_to_dPoint[name].rknn_dist/len(name_to_dPoint[name].rknn)
               name_to_dPoint[name].densityPeak =len(name_to_dPoint[name].rknn)/ name_to_dPoint[name].rknn_dist
            li_peak.append(name_to_dPoint[name].densityPeak)


        for i in range(X.shape[0]):
            name = name_dic[i]
            cur_point = name_to_dPoint[name]
            heatness = 0
            for j in  cur_point.symmetric_knn:
                nn =  name_dic[j]
                heatness += name_to_dPoint[nn].densityPeak
            if cur_point.symmetric_knn:
             name_to_dPoint[name].densityPeak = heatness / len(cur_point.symmetric_knn)


        li_peak = sorted(li_peak)
        li_peak = np.array(li_peak)
        li_peak /=li_peak.max()

        return epsilon
    else:
        return None


def rearrange_name(X):
    for i, item in enumerate(X):
        tmp_name = str_to_name(item)
        ind_to_name[i] = tmp_name


def globalhotspotFind():
    sigma = None
    for key, d2Punkt in name_to_dPoint.items():
        if sigma == None:
            sigma = d2Punkt
        # print(d2Punkt.densityPeak, d2Punkt.avg_k_distance )
        tmp_avg = d2Punkt.densityPeak
        tmp_neighours_len = len(d2Punkt.neighbours)
        # if tmp_avg < sigma.avg_k_distance and tmp_nachbarn > len(sigma.neighbours):
        #     sigma = d2Punkt
        if tmp_avg > sigma.densityPeak:
            sigma = d2Punkt
        if tmp_avg == sigma.densityPeak:
            sigma = compare(sigma, d2Punkt)
    # print(sigma.name)
    return sigma


def compare(d1, d2):
    d2Coor = d2.coordinates
    for i, value in enumerate(d1.coordinates):
        if value == d2Coor[i]:
            continue
        elif value < d2Coor[i]:
            return d1
        else:
            return d2
    return d1

def compare_mod(d1, d2):
    for i, value in enumerate(d1):
        if value == d2[i]:
            continue
        elif value < d2[i]:
            return -1
        else:
            return 1
    return 0



def merge():
    label_list  = sorted(list(label_dic.keys()))
    para_dic = {}
    for item in label_list:
        para_dic[item] = item
    # for i in range(len(label_list)):
    #     for j in range(i+1, len(label_list)):
    #         for data in label_dic[i]:
    #             if data in label_dic[j]:
    #                 para_dic[j] = para_dic[i]
    return para_dic

def collectLabels():
    y_label = []
    para_dic = merge()
    for datapoint in data_matrix:
        name = str_to_name(datapoint)
        if name in name_to_label:
            y_label.append(para_dic[name_to_label[name]])
        else:
            y_label.append(0)
    global label_counter
    return np.array(y_label).reshape(-1)


def CBLH_Start(X, k=None):
    if k == None: k = 20
    global ind_to_name, name_to_dPoint, name_to_label, label_counter, data_matrix
    ind_to_name = {}
    name_to_label = {}
    name_to_dPoint = {}
    label_counter = 0
    data_matrix = None
    init(X)
    return initClustering(k)
