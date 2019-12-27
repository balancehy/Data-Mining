import numpy as np
from math import pi, sqrt, exp, e, log


class GMM:
    def __init__(self, filename, K, mu, cov, prior, max_iter=100, threshold=1e-9, smooth=1e-9):
        self.ground_truth, self.data = self.load_data(filename)
        self.K = K
        self.mu = mu
        if mu is None:
            self.mu = [self.data[i] for i in range(K)]
        self.cov = cov
        if cov is None:
            self.gen_cov()
        self.prior = prior
        if prior is None:
            self.prior = [1/K for i in range(K)]
        self.max_iter = max_iter
        self.threshold = threshold
        self.smooth = smooth
        self.apply_smooth()
        # print(self.data, self.ground_truth)
        gussian_mat, prob_mat = self.cal_gussian_prob()
        likelihold_old = -10000000000
        likelihold = self.cal_ln_likelihood(gussian_mat, prob_mat)

        while (abs(likelihold - likelihold_old) > self.threshold and max_iter > 0):
            if likelihold > likelihold_old:
                likelihold_old = likelihold
            self.update_pi(prob_mat)
            self.update_mu(prob_mat)
            self.update_cov(prob_mat)
            print("new prior is {}\n,new center is \n{}\n new cov is \n{}\n".format(np.array(self.prior),np.array(self.mu),np.array(self.cov)))
            self.apply_smooth()
            gussian_mat, prob_mat = self.cal_gussian_prob()
            likelihold = self.cal_ln_likelihood(gussian_mat, prob_mat)
            max_iter-=1
        cluster = []
        for i in range(len(prob_mat)):
            cluster.append(np.argmax(prob_mat[i]))

        plot_pca(self.data,cluster)
        matrix = caculate_jaccard_matrix(cluster)
        t = caculate_jaccard_matrix(self.ground_truth)
        print(perform_jaccard_coefficient(t,matrix))

    def gen_cov(self):
        matrix = []
        for i in range(self.K):
            sub_matrix = []
            for j in range(len(self.data[0])):
                sub_row = []
                for k in range(len(self.data[0])):
                    if j == k:
                        sub_row.append(i+1)
                    else:
                        sub_row.append(0)
                sub_matrix.append(sub_row)
            matrix.append(sub_matrix)
        self.cov = matrix

    def update_pi(self,prob_mat):
        for i in range(len(self.prior)):
            temp = 0
            for j in range(len(prob_mat)):
                 temp+= prob_mat[j][i]
            self.prior[i] = temp/len(prob_mat)
    def update_mu(self,prob_mat):
        for i in range(len(self.mu)):
            temp = 0
            total_prob = 0
            for j in range(len(self.data)):
                temp+=self.data[j]*prob_mat[j][i]
                total_prob+=prob_mat[j][i]
            self.mu[i] = temp/total_prob

    def update_cov(self,prob_mat):
        for i in range(len(self.mu)):
            temp = 0
            total_prob = 0

            for j in range(len(self.data)):
                temp+=np.matmul((self.data[j]-self.mu[i])[:,None],(self.data[j]-self.mu[i])[None,:]) *prob_mat[j][i]
                total_prob+=prob_mat[j][i]
            self.cov[i] = temp/total_prob

    def cal_gussian_prob(self):
        gussian_mat = []
        for data in self.data:
            gussian_mat_sub = []
            for i in range(self.K):
                gussian_mat_sub.append(self.mul_gussian(data, self.mu[i], self.cov[i]))
            gussian_mat.append(gussian_mat_sub)
        prob_mat = []
        for i in range(len(self.data)):
            total = 0
            prob_mat_sub = []
            for k in range(self.K):
                total += gussian_mat[i][k] * self.prior[k]
            for k in range(self.K):
                prob_mat_sub.append(gussian_mat[i][k] * self.prior[k] / total)
            prob_mat.append(prob_mat_sub)
        return gussian_mat, prob_mat

    def cal_ln_likelihood(self, gaussian_mat, prob_mat):
        likelihood = 0
        for i in range(len(gaussian_mat)):
            weigted_prob = 0
            for k in range(self.K):
                weigted_prob += prob_mat[i][k] * np.log(self.prior[k] * gaussian_mat[i][k]+self.smooth)
            likelihood += weigted_prob
        return likelihood

    def apply_smooth(self):
        for j in range(len(self.cov)):
            for i in range(len(self.cov[j])):
                self.cov[j][i][i] += self.smooth

    def mul_gussian(self, xi, mu, cov):

        return 1 / sqrt((2 * pi) ** len(xi) * (np.linalg.det(cov))) * exp(
            -np.matmul(np.matmul((xi - mu), np.linalg.inv(cov)), (xi - mu)[:,None]) / (2))

    def load_data(self, filename):
        data = np.genfromtxt(filename, delimiter='\t', dtype=float)
        ground_truth = data[:, 1]
        data_raw = data[:, 2:]

        return ground_truth, data_raw

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
    fig.suptitle('GMM of gmm.txt')
    for group in np.unique(cluster_labels):
        idx = np.where(cluster_labels == group)
        ax.scatter(data_pca[:, 0][idx], data_pca[:, 1][idx], label = group)
    ax.legend(loc=1, ncol=3)
    plt.savefig("GMM_gmm.png")
    plt.show()


if __name__ == '__main__':
    filename = "data/GMM.txt"

    k = 2
    mu = [[0, 0], [1, 1]]
    cov = [[[1, 0], [0, 1]], [[2, 0], [0, 2]]]
    prior = [0.5, 0.5]

    cluster = GMM(filename, k, mu, cov, prior)
    # filename = "data/cho.txt"
    # GMM(filename,5,None,None,None)


