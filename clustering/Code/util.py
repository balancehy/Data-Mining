# !usr/bin/python3
import numpy as np

def caculate_jaccard_matrix(cluster):
    matrixs = [[0 for i in range(len(cluster))] for i in range(len(cluster))]
    for i in range(len(cluster)):
        for j in range(len(cluster)):
            if cluster[i] == cluster[j]:
                matrixs[i][j] = 1
    return matrixs

def perform_jaccard_coefficient(truth, cluster):
    """caculate the rand and jaccard index
    :parameter truth: ground truth labels, cluster: the labels from our algorithm
    rtype: rand index(float), jaccard (float)
    """
    same, diff, both_zero = 0, 0, 0
    for idx_x, row in enumerate(truth):
        for idx_y, value in enumerate(row):
            if truth[idx_x][idx_y] == 1 and truth[idx_x][idx_y] == cluster[idx_x][idx_y]:
                same += 1
            elif truth[idx_x][idx_y] != cluster[idx_x][idx_y]:
                diff += 1
            else:
                both_zero += 1
    return (same + both_zero) * 1.0 / (same + both_zero + diff), (same) * 1.0 / (same + diff)

def plot_pca(x, cluster_labels):
    from sklearn.decomposition import PCA
    import matplotlib.pyplot as plt
    data_pca = PCA(n_components=2).fit_transform(x)
    fig, ax = plt.subplots()
    fig.suptitle('DBSCAN of cho.txt')
    for group in np.unique(cluster_labels):
        idx = np.where(cluster_labels == group)
        ax.scatter(data_pca[:, 0][idx], data_pca[:, 1][idx], label = group)
    ax.legend(loc=1, ncol=3)
    plt.show()