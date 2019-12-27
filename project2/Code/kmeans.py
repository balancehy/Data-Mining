# !usr/bin/python3

import numpy as np 
import pickle
import os
import pandas as pd
import sys
from util import caculate_jaccard_matrix, perform_jaccard_coefficient, plot_pca



class Kmeans():
    def __init__(self, k = 10, maxiter=20, num_seed=5, centor=None):
        self.num_cluster = k
        self.data = None
        self.maxiter = maxiter
        self.num_seed = num_seed
        self.inertia = 0
        self.given_centor = None if centor is None else np.array(centor, dtype=np.int32)
        
    def load_data(self, fpath):
        self.data = np.array(pd.read_csv(fpath, sep='\t', header=None))

    def distance(self, x, y):
        return np.sqrt(np.mean(np.square(x - y)))

    def assign_cluster(self, cluster_id_new, atts_c, att_p, gid_p, cluster_id=None):
        # atts_c is attributes of all centroid, atts_p is current point
        
        mindist = 10000000#float('inf')
        bestindex = -1
        for i in range(len(atts_c)):
            if cluster_id is None or len(cluster_id[i])!=0: # handle when no point is assigned to cluster[i]
                dist = self.distance(atts_c[i, :], att_p)
                if dist<mindist:
                    mindist = dist
                    bestindex = i
        self.inertia += mindist
        cluster_id_new[bestindex].append(gid_p)

    def init_centroid(self):
        n = len(self.data)
        index = np.arange(n) # centroid index
        hasEmptyClass = True
        while hasEmptyClass:
            self.inertia = 0
            if self.given_centor is None:
                np.random.shuffle(index)
                atts_centroid = self.data[index[0:self.num_cluster], 2:]
            else:
                atts_centroid = self.data[self.given_centor-1, 2:]
                print("The initial gene id for each cluster is: ", self.given_centor)
            
            cluster_id = [[] for i in range(self.num_cluster)]
            for i in range(len(self.data)): # every class has sorted gene id
                gid = self.data[i, 0]
                att = self.data[i, 2:]
                self.assign_cluster(cluster_id, atts_centroid, att, gid)
            
            hasEmptyClass = False
            for c in cluster_id:
                if len(c)==0:
                    hasEmptyClass = True
                    break
            

        return cluster_id

    def one_step(self, cluster_id):
        
        atts_centroid_new = np.zeros(shape=[len(cluster_id), len(self.data[1, 2:])])
        cluster_id_new = [[] for i in range(self.num_cluster)]
        # Update centroid for each class
        for i in range(len(cluster_id)):
            if len(cluster_id[i])!=0:
                atts_centroid_new[i, :] = np.mean(self.data[np.array(cluster_id[i], dtype=np.int32)-1, 2:], axis=0)

        # Assgin class to each point in data
        self.inertia = 0
        for i in range(len(self.data)): # every class has sorted gene id
            gid = self.data[i, 0]
            att = self.data[i, 2:]
            self.assign_cluster(cluster_id_new, atts_centroid_new, att, gid, cluster_id=cluster_id)

        return cluster_id_new, atts_centroid_new
    
    def is_same(self, cluster_id, cluster_id_new):
        
        for i in range(len(cluster_id)):
            
            for j in range(len(cluster_id[i])):
                if len(cluster_id[i])!=len(cluster_id_new[i]):
                    return False
                if cluster_id[i][j] != cluster_id_new[i][j]:
                    return False

        return True

    def fit(self):
        cluster_id_best = None
        dist_best = 10000000
        counter_best = -1
        for i in range(self.num_seed):
            # np.random.seed(i)
            cluster_id, atts_centroid, counter = self.train()
            # print("distance: ", self.inertia)
            if self.inertia < dist_best:
                cluster_id_best = cluster_id
                dist_best = self.inertia
                counter_best = counter

        print("Number of iterations: ", counter_best)
        print("Final distance: ", dist_best)

        cluster_label = np.zeros((len(self.data), ))
        for ic, c in enumerate(cluster_id_best):
            for gid in c:
                cluster_label[int(gid)-1] = ic+1
        
        return cluster_label

    def train(self):
        
        cluster_id = self.init_centroid()
        atts_centroid_new = None
        counter = 0
        while(counter<self.maxiter):
            cluster_id_new, atts_centroid_new = self.one_step(cluster_id)
            # print(self.is_same(cluster_id, cluster_id_new))
            if self.is_same(cluster_id, cluster_id_new):
                break
            cluster_id = cluster_id_new
            counter += 1
        
        return cluster_id, atts_centroid_new, counter

if __name__ == "__main__":
    try:
        fpath = sys.argv[1]
    except IndexError:
        print("Please input file path")

    c = Kmeans(num_seed=1, maxiter=50, centor=None)
    c.load_data(fpath)
    print("Shape of data: ", c.data.shape)
    unique_class_truth = np.unique(c.data[:, 1])
    print("Class ground truth: ", unique_class_truth)
    if unique_class_truth[0]==-1:
        c.num_cluster = len(unique_class_truth)-1
    else:
        c.num_cluster = len(unique_class_truth)
    # atts_centroid, cluster_id = c.init_centroid()
    
    
    
    predict_label = c.fit()

    # from sklearn.cluster import KMeans
    # d_test = c.data[:, 2:]
    # obj = KMeans(n_clusters=10, random_state=0).fit(d_test)
    # predict_label = obj.labels_
    # print(obj.n_iter_)

    # print(predict_label-c.data[:, 1])
    predicted_matrix = caculate_jaccard_matrix(predict_label)
    truth_matrix = caculate_jaccard_matrix(c.data[:, 1])

    res = perform_jaccard_coefficient(truth_matrix, predicted_matrix)
    print("Rand index: " + str(res[0]) + " Jaccard Coefficient: " + str(res[1]))

    plot_pca(c.data[:, 2:], predict_label)
