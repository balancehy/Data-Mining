import numpy as np
import pandas as pd
import sys
from matplotlib import pyplot as plt
from sklearn.naive_bayes import MultinomialNB, GaussianNB

def evaluate(truth, predicted, verbose=True):

    tp, fp, tn, fn = 0, 0, 0, 0
    for i in range(len(truth)):
        t, p = truth[i], predicted[i]
        
        if t==1 and p==1:
            tp += 1
        if t==1 and p==0:
            fn += 1
        if t==0 and p==1:
            fp += 1
        if t==0 and p==0:
            tn += 1
    
    precision = tp/(tp + fp)
    recall = tp/(tp + fn)
    F1 = 2*precision*recall/(precision + recall)
    acc = (tp + tn)/(tp + fp + tn + fn)
    if verbose:
        print("tp, tn, fp, fn: ", tp, tn ,fp, fn)
        print("Precision: ", precision)
        print("Recall: ", recall)
        print("Accuracy", acc)
        print("F1: ", F1)
    
    return tp, fp, tn, fn, acc, F1

def cross_validation(x_np, y_np, n_folds=10, customer=True, verbose=False):
    
    n = len(x_np)
    idx = np.arange(n)
    np.random.shuffle(idx)
    
    step = int(n/n_folds)
    folds_idx = np.arange(0, n, step)
    # print(step, folds_idx)
    best_clf = None
    best_f1 = 0
    total_f1 = 0
    total_acc = 0
    for i in range(n_folds):
        left= folds_idx[i]
        if i==n_folds-1:
            right = n
        else:
            right = folds_idx[i+1]
        
        x_te = x_np[idx[left:right], :]
        y_te = y_np[idx[left:right], :]
        if i==0:
            x_tr = x_np[idx[right:], :]
            y_tr = y_np[idx[right:], :]
        elif i==n_folds-1:
            x_tr = x_np[idx[:left], :]
            y_tr = y_np[idx[:left], :]
        else:
            x_tr = np.concatenate((x_np[idx[:left], :], x_np[idx[right:], :]), axis=0)
            y_tr = np.concatenate((y_np[idx[:left], :], y_np[idx[right:], :]), axis=0)
        
        data_tr = np.concatenate([x_tr, y_tr], axis=1)
        data_te = np.concatenate([x_te, y_te], axis=1)

        if customer:
            clf = NBayes(data_tr, bins)
            y_pred, log_prob = clf.predict(data_te)
        else:
            clf = MultinomialNB()
            clf.fit(x_tr, y_tr.reshape(-1))
            y_pred = clf.predict(x_te)
        
        tp, fp, tn, fn, acc, F1 = evaluate(y_te, y_pred, verbose=verbose)
        if F1>best_f1:
            best_f1 = F1
            best_clf = clf
        total_f1 += F1
        total_acc += acc
    
    print("Mean accuracy: ", total_acc/n_folds)
    print("Mean F1: ", total_f1/n_folds)

    return best_clf, best_f1

def load_data(fpath, tr_ratio):
    # Load and split data into training and testing sets
    data = np.array(pd.read_csv(fpath, delimiter="\t", header=None))
    print("Shape of original data", data.shape)
    # Split into train and test dataset
    n = len(data)
    idx = np.arange(0, n)
    np.random.shuffle(idx)
    data_tr = data[idx[0 : int(n*tr_ratio)], :]
    data_te = data[idx[int(n*tr_ratio): ], :]
    # print(len(data_tr) + len(data_te))
    return data_tr, data_te 

def categorty_to_num(data):
    # Convert string column(categorical) to integer
    strcol = set()
    for i, c in enumerate(data[0, :]):
        if type(c) == np.str:
            strcol.add(i)
            for j, u in enumerate(np.unique(data[:, i])):
                data[np.where(data[:, i] == u)[0], i] = j
    data = np.array(data, dtype=np.float)
    
    return data, strcol

def discretization(data, strcol, bins=None, nbin=10, mode="train"):
    # Discretize continuous data into defined bins
    if mode == "train":
        bins = []
    elif mode == "query":
        if bins is None:
            raise ValueError("bins is not provided")
    else:
        raise ValueError("Only train and query mode are available")
    
    for c in range(len(data[0, :]) - 1): # not include last column(class label)
        if c in strcol: # not inlcude categorical column
            if mode == "train":
                bins.append(np.array([]))
            continue
        if mode == "train": # when training, do not know what bins are
            _, bi, _ = plt.hist(data[:, c], bins=nbin)
            bins.append(bi)
        elif mode == "query": # when query, provide bins
            bi = bins[c]

        for r in range(len(data[:, c])):
            if data[r, c]<bi[0] or data[r, c]>bi[len(bi)-1]:
                data[r, c] = nbin*2 # data attribute value out of bounds
            for i in range(len(bi)-1):
                if i < len(bi)-2:
                    if data[r, c]>=bi[i] and data[r, c]<bi[i+1]:
                        data[r, c] = i
                else:
                    if data[r, c]>=bi[i] and data[r, c]<=bi[i+1]:
                        data[r, c] = i
    if mode == "train":
        return data, bins
    elif mode == "query":
        return data

class NBayes():
    def __init__(self, data_tr, bins):
        self._data = data_tr
        # self._nbin = nbin
        # self._str2num = str2num
        self._bins = bins
        # self._strcol = strcol
    
    def category_to_onehot(self):
        
        strcol = []
        for i, c in enumerate(self._data[0, :]):
            if type(c) == np.str:
                strcol.append(i)
        for c in strcol:
            unique_idx = {}
            for i, u in enumerate(np.unique(self._data[:, c])):
                unique_idx[u] = i
            num_unique = len(unique_idx)
            for i, e in enumerate(self._data[:, c]):
                self._data[i, c] = unique_idx[e]
            # print(np.array(self._data[:, c], dtype=np.int))
            # print(np.eye(num_unique)[np.array(self._data[:, c], dtype=np.int), :])
            self._data = np.concatenate([self._data[:, 0:c], 
                                        np.eye(num_unique)[np.array(self._data[:, c], dtype=np.int), :], 
                                        self._data[:, c+1:]],
                                        axis=1)
                         
        self._data = np.array(self._data, dtype=np.float)
        print("Shape of data converted to one-hot", self._data)

    def predict(self, att):
        
        if type(att) is list:
            att.append(-1) # make len of query is the same as row of original data
            att = np.array(att, dtype=np.object).reshape(1, -1)
        
        # if self._str2num:
        #     self.categorty_to_num(att, mode="query")
        # self.discretization(att, mode="query")
        
        label_uni = np.unique(self._data[:, -1])
        log_probs = []
        for l in label_uni:
            class_count = len(np.where(self._data[:, -1] == l)[0])
            class_prior = float(class_count / len(self._data))
            prob_label = []
            for r in range(len(att)):
                log_prob = 0.0
                for c in range(len(att[0, :]) - 1):
                    bit = (self._data[:, c] == att[r][c]) & (self._data[:, -1] == l)
                    att_count = len(np.where(bit)[0])
                    if att_count == 0: # laplacian corrector
                        llh = float((att_count+1) / (class_count + len(label_uni)))
                    else:
                        llh = float(len(np.where(bit)[0]) / class_count)

                    log_prob += np.log(llh)
                    
                log_prob += np.log(class_prior) # not normalize by p(x)
                prob_label.append(log_prob)
            log_probs.append(prob_label)
        log_probs = np.array(log_probs).T # attribute * class
        # print(log_probs)
        best_class = np.argmax(log_probs, axis=1)
        max_prob = log_probs[:, best_class]

        return best_class, max_prob

if __name__ == "__main__":
    try:
        fpath = sys.argv[1]
    except:
        print("File path is not provided.")
        sys.exit(1)
    tr_split_ratio = 0.8
    num_bins = 10
    np.random.seed(1234)
    data_tr, data_te = load_data(fpath, tr_split_ratio)
    # print(data_tr.shape, data_te.shape)

    # Prep data set
    data_tr, strcol = categorty_to_num(data_tr)
    data_tr, bins = discretization(data_tr, strcol, bins=None, nbin=num_bins, mode="train")
    data_te, _ = categorty_to_num(data_te)
    data_te = discretization(data_te, strcol, bins=bins, nbin=num_bins, mode="query")

    # Predict labels in testing set
    # Use customized classifier
    # clf = NBayes(data_tr, bins)
    # label_pred, log_prob = clf.predict(data_te)

    # Use sklearn classifier
    # clf2 = GaussianNB()
    # clf2 = MultinomialNB()
    # clf2.fit(data_tr[:, 0:-1], data_tr[:, -1])
    # label_pred = clf2.predict(data_te[:, 0:-1])
    
    # tp, fp, tn, fn = evaluate(data_te, label_pred)
    # precision = tp/(tp + fp)
    # recall = tp/(tp + fn)
    # print("Precision: ", precision)
    # print("Recall: ", recall)
    # print("Accuracy", (tp + tn)/(tp + fp + tn + fn))
    # print("F1: ", 2*precision*recall/(precision + recall))

    # Cross validation
    print(data_tr.shape, data_te.shape)
    data = np.concatenate([data_tr, data_te], axis=0)
    x_np, y_np = data[:, 0:-1], data[:, -1:]
    # print(x_np.shape, y_np.shape)
    # clf2 = MultinomialNB()
    # clf2.fit(x_np[0:-2, :], y_np[0:-2, :].reshape(-1))
    cross_validation(x_np, y_np, n_folds=5, customer=True, verbose=False)
    # cross_validation(x_np, y_np, n_folds=5, customer=False, verbose=False)