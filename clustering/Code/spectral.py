import numpy as np

from sklearn import cluster


def load_data(filename):
    data = np.genfromtxt(filename, delimiter='\t', dtype=float)
    ground_truth = data[:, 1]
    data_raw = data[:, 2:]

    return ground_truth, data_raw


def gen_kernel(data, sigma):
    kernel = np.zeros(shape=(len(data), len(data)))

    for i in range(len(data)):
        for j in range(len(data)):
            if i == j:
                kernel[i][j] = 0
            else:
                kernel[i][j] = cal_dist(data[i], data[j], sigma)
    return kernel


def cal_dist(x1, x2, sigma):
    return np.exp(-(np.linalg.norm(x1 - x2)) ** 2 / sigma ** 2)


def gen_L(kernel):
    L_mat = np.zeros(shape=(len(kernel), len(kernel)))
    D_mat = np.zeros(shape=(len(kernel), len(kernel)))
    for i in range(len(kernel)):
        for j in range(len(kernel)):
            if i == j:
                L_mat[i][j] = sum(kernel[i, :])
                D_mat[i][j] = sum(kernel[i, :])+1e-9
            else:
                L_mat[i][j] = -kernel[i][j]
    return L_mat, D_mat


def define_gap(eigen_value):
    k = len(eigen_value) - 1
    max_gap = 0
    op_k = k
    while k > 0:
        temp = abs(eigen_value[k] - eigen_value[k - 1])
        if temp >= max_gap:
            max_gap = temp
            op_k = k
        k -= 1
    return op_k


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
    fig.suptitle('new_data_1.txt')
    for group in np.unique(cluster_labels):
        idx = np.where(cluster_labels == group)
        ax.scatter(data_pca[:, 0][idx], data_pca[:, 1][idx], label=group)
    ax.legend(loc=1, ncol=3)
    plt.savefig("new_data_1.png")
    plt.show()



if __name__ == '__main__':
    filename = "data/new_dataset_1.txt"
    sigma = 1
    truth, data = load_data(filename)
    kernel = gen_kernel(data, sigma)
    L_mat, D_mat = gen_L(kernel)
    L_mat_norm = np.matmul(np.linalg.inv(D_mat), L_mat)

    eigen_value, eigen_vector = np.linalg.eig(L_mat_norm)
    idx = eigen_value.argsort()
    eigen_value = eigen_value[idx]
    eigen_vector = eigen_vector[:, idx]

    # k = define_gap(eigen_value)
    k=3
    U = np.array(eigen_vector[:, :k])

    kmean = cluster.KMeans(init="k-means++", n_clusters=k)
    kmean.fit(U)
    c = kmean.labels_

    matrix_truth = caculate_jaccard_matrix(truth)
    matrix_c = caculate_jaccard_matrix(c)
    print(perform_jaccard_coefficient(matrix_truth, matrix_c))
    plot_pca(data, c)



    # print(eigen_value,eigen_vector)
