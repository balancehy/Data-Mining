import numpy as np
import random

threshold = 0.1
force_grow = False
first = False


def metrics(actual, predicted):
    """metrics
    Return Acc, precision, recall, f-measure
    """
    a, b, c, d = 0, 0, 0, 0
    for i in range(len(actual)):
        if actual[i] == 1 and predicted[i] == 1:
            a += 1
        elif actual[i] == 1 and predicted[i] == 0:
            b += 1
        elif actual[i] == 0 and predicted[i] == 1:
            c += 1
        elif actual[i] == 0 and predicted[i] == 0:
            d += 1

    def acc(a, b, c, d):
        return (a + d) * 1.0 / (a + b + c + d)

    def p(a, c):
        if a + c == 0:
            return -1
        return a * 1.0 / (a + c)

    def r(a, b):
        if a + b == 0:
            return -1
        return a * 1.0 / (a + b)

    def f(a, b, c):
        if a + b + c == 0:
            return -1
        return 2 * a * 1.0 / (2 * a + b + c)

    return acc(a, b, c, d), p(a, c), r(a, b), f(a, b, c)


class Node(object):
    def __init__(self, idx, attr, encode):  # dtype=0:continuous,1:nomial
        self.branch_idx = idx
        self.child = {}
        self.attr = attr

        if idx not in encode:
            self.child = {"left": None, "right": None}
        else:
            for elem in attr:
                self.child[encode[idx][elem]] = None

    def __str__(self):
        return "idx: {} attr: {}, child{}".format(self.branch_idx, self.attr, self.child)

    def __repr__(self):
        return "idx: {} attr: {}, child{}".format(self.branch_idx, self.attr, self.child)


def fit_tree(tree, test_data, encode):
    pointer = tree
    prediction = []
    for test_sample in test_data:

        while isinstance(pointer, Node):
            idx = pointer.branch_idx
            value = test_sample[idx]

            if type_map[idx] == 0:
                if value < pointer.attr[0]:
                    pointer = pointer.child["left"]
                else:
                    pointer = pointer.child["right"]

            else:
                value = encode[idx][value]
                pointer = pointer.child[value]
        prediction.append(pointer)
        pointer = tree
    return prediction


def fit_forest(trees, test_data, encode):
    predictions = []
    actual_predict = []
    for tree in trees:
        prediction = fit_tree(tree, test_data, encode)
        predictions.append(prediction)
    predictions = np.asarray(predictions)
    for i in range(len(test_data)):
        label, cnt = np.unique(predictions[:, i], return_counts=True)
        most_cnt_label = label[np.argmax(cnt)]
        actual_predict.append(most_cnt_label)
    return actual_predict


def build_tree(data, test_data, type_map, encode, n_feature):
    available_col = list(range(len(data[0]) - 1))
    label = np.unique(data[:, -1])

    tree = select_attr(data, available_col, type_map, label, encode, n_feature, 0)
    prediction = fit_tree(tree, test_data, encode)
    # print(np.max(data[:,8]))
    return tree, prediction


def select_attr(data, available_col, type_map, label, encode, n_feature, first):
    chosen_attr = (-1, -1, 100)
    labels, cnt = np.unique(data[:, -1], return_counts=True)
    if len(labels) == 1:
        return labels[0]

    if len(available_col) == 0 or len(data) < 2:
        return labels[np.argmax(cnt)]

    origin_gini, higher = cnt_origin_gini(data[:, -1])
    # if origin_gini<0.1 and not force_grow:
    #     return higher
    sample_col = available_col
    if len(available_col) >= n_feature:
        sample_col = random.sample(available_col, n_feature)

    # print(sample_col)
    for idx in sample_col:
        data = data[np.argsort(data[:, idx])]
        data_attr = data[:, [idx, -1]]
        if type_map[idx] == 0:
            split_attr, gini = find_best_split(data_attr, label)
        else:
            split_attr, gini = find_best_nomial(data_attr, label)

        if gini < chosen_attr[2]:
            chosen_attr = (idx, split_attr, gini)

    if origin_gini < chosen_attr[2] and not force_grow and first != 0:
        return higher

    node = Node(chosen_attr[0], chosen_attr[1], encode)
    data_split = []
    available_col.remove(chosen_attr[0])
    if type_map[chosen_attr[0]] == 0:
        data_split.append(data[np.where(data[:, chosen_attr[0]] < chosen_attr[1][0])])
        data_split.append(data[np.where(data[:, chosen_attr[0]] >= chosen_attr[1][0])])
        node.child["left"] = select_attr(data_split[0], available_col.copy(), type_map, label, encode, n_feature, 1)
        node.child["right"] = select_attr(data_split[1], available_col.copy(), type_map, label, encode, n_feature, 1)
    else:
        if len(split_attr) == 1:
            return higher
        else:
            for elem in split_attr:
                temp = data[np.where(data[:, chosen_attr[0]] == elem)]
                node.child[encode[chosen_attr[0]][elem]] = select_attr(temp, available_col.copy(), type_map, label,
                                                                       encode, n_feature, 1)
    return node


def cnt_origin_gini(label_col):
    if len(label_col) < 2:
        return label_col[-1]
    label, cnt = np.unique(label_col, return_counts=True)
    higher = label[np.argmax(cnt)]
    # print(higher)
    mini_gini = min(cnt) / sum(cnt)
    return mini_gini, higher


def find_best_nomial(data_attr, label):
    split_attr = []
    cnt_map = {}

    for elem, cls in data_attr:
        if elem not in cnt_map:
            cnt_map[elem] = {i: 0 for i in label}
            split_attr.append(elem)
        cnt_map[elem][cls] += 1
    gini = cal_gini(cnt_map)
    return split_attr, gini


def find_best_split(data_attr, label):
    initial = data_attr[0, 0] - 1
    best_gini = 1
    best_split = (initial, best_gini)
    split_map = init_split(data_attr, label)
    begin = data_attr[0, 0]
    for elem, cls in data_attr:
        if elem != initial and elem != begin:
            temp_gini = cal_gini(split_map)
            if temp_gini < best_split[1]:
                best_split = (elem, temp_gini)
            initial = elem

        split_map[0][cls] += 1
        split_map[1][cls] -= 1
    # print(best_split)
    best_split = ([best_split[0]], best_split[1])
    return best_split


def init_split(data_attr, label):
    branch_str0 = 0
    branch_str1 = 1
    init_map = {branch_str0: {i: 0 for i in label}, branch_str1: {i: 0 for i in label}}
    for elem, label1 in data_attr:
        init_map[branch_str1][label1] += 1

    return init_map


def cal_gini(cnt_map):
    gini = 0
    total = 0
    impurity_arr = []
    for key in cnt_map:
        impurity, leaf_cnt = cal_gini_leaf(cnt_map[key])
        impurity_arr.append((impurity, leaf_cnt))
        total += leaf_cnt
    for elem, cnt in impurity_arr:
        gini += (cnt / total) * elem
    return gini


def cal_gini_leaf(leaf_map):
    total = 0
    impurity = 1
    for key in leaf_map:
        total += leaf_map[key]
    for key in leaf_map:
        impurity -= (leaf_map[key] / total) ** 2
    return impurity, total


def preprocess(data):
    encode_map = {}

    type_map = {}  # 0: continous, 1: nomial
    ret_data = np.zeros(shape=data.shape, dtype=float)
    for i in range(len(data[0])):

        try:
            ret_data[:, i] = data[:, i].astype(float)
            type_map[i] = 0
        except ValueError:
            cnt = 0
            encode_map[i] = {}
            for j in range(len(data[:, i])):
                elem = data[j, i]
                if elem not in encode_map[i]:
                    encode_map[i][elem] = cnt
                    ret_data[j, i] = float(cnt)
                    cnt += 1
                else:
                    ret_data[j, i] = float(encode_map[i][elem])

            type_map[i] = 1
    return ret_data, encode_map, type_map


def cross_validation(algorithm, data, n_fold, *args):
    length = len(data)
    even_split = length // n_fold
    accs = []
    ps = []
    rs = []
    fs = []
    best_tree = None
    best_acc = 0

    for i in range(n_fold):
        test_data = data[:even_split]
        train_data = data[even_split:]
        truth_label = test_data[:, -1]
        data = np.concatenate((train_data, test_data))
        tree, prediction = algorithm(train_data, test_data, *args)
        acc, p, r, f = metrics(truth_label, prediction)

        accs.append(acc)
        if p != -1:
            ps.append(p)
        if r != -1:
            rs.append(r)
        if f != -1:
            fs.append(f)
        if acc > best_acc:
            best_acc = acc
            best_tree = tree
    avg_acc = sum(accs) / n_fold
    avg_f = sum(fs) / len(fs)
    avg_r = sum(rs) / len(rs)
    avg_p = sum(ps) / len(ps)
    return best_tree, avg_acc, avg_p, avg_r, avg_f


def boostrap_data(data):
    ret_data = []
    for i in range(len(data)):
        ret_data.append(data[random.randint(0, len(data) - 1)])
    ret_data = np.asarray(ret_data)
    return ret_data


def random_forest(train_data, test_data, n_tree, type_map, decoder, n_feature):
    predictions = []
    actual_predict = []
    forest = []
    for i in range(n_tree):
        boostraped_data = boostrap_data(train_data)
        tree, prediction = build_tree(boostraped_data, test_data, type_map, decoder, n_feature)
        predictions.append(prediction)
        forest.append(tree)
    predictions = np.asarray(predictions)
    for i in range(len(test_data)):
        label, cnt = np.unique(predictions[:, i], return_counts=True)
        most_cnt_label = label[np.argmax(cnt)]
        actual_predict.append(most_cnt_label)
    score = metrics(test_data[:, -1], actual_predict)
    return forest, actual_predict


if __name__ == '__main__':
    filename = "project3_dataset1.txt"
    data_raw = np.genfromtxt(filename, delimiter="\t", dtype=str)
    data, encode_map, type_map = preprocess(data_raw)
    decoder = {}
    for key in encode_map:
        decoder[key] = {}
        for key1 in encode_map[key]:
            decoder[key][encode_map[key][key1]] = key1

    # Decision Tree without cross validation
    # n_feature = len(data[0])-1
    # tree,prediction = build_tree(data,data,type_map,decoder,n_feature)
    # print(decoder)
    # truth = data[:,-1]
    # print(tree)
    # print(metrics(truth,prediction))


    #decision Tree with cross validationx
    # n_fold = 10
    # n_feature = len(data[0])-1
    # tree, avg_acc, avg_p, avg_r, avg_f = cross_validation(build_tree, data, n_fold, type_map, decoder,n_feature)
    # print(
    #     "************Decision Tree********************\navg_acc：{}\nprecision: {}\nrecall: {}\nf1-measure: {}\n****************************".format(
    #         avg_acc, avg_p, avg_r, avg_f))

    # print(tree)

    #Random Forest without cross validation
    # n_feature = 2
    # n_tree = 30
    # forest,prediction = random_forest(data,data,n_tree,type_map,decoder,n_feature)
    # score = metrics(data[:, -1], prediction)
    # print(
    #     "*************random forest*******************\navg_acc：{}\nprecision: {}\nrecall: {}\nf1-measure: {}\n****************************".format(
    #         score[0], score[1], score[2], score[3]))
    # print(forest)

    # Random Forest with cross validation
    n_fold = 10
    n_feature = 6
    n_tree = 25
    tree, avg_acc, avg_p, avg_r, avg_f = cross_validation(random_forest, data, n_fold, n_tree,type_map, decoder, n_feature)
    print(
        "************Random Forest********************\navg_acc：{}\nprecision: {}\nrecall: {}\nf1-measure: {}\n****************************".format(
            avg_acc, avg_p, avg_r, avg_f))
    # print(tree)
