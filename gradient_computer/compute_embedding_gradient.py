import numpy as np


class CNEGradient:
    @classmethod
    def get_node_hessian_inverse(cls, i, cne, node_hessian_val=None):
        posterior_prob_i_row = cne.get_posterior_row(i)
        adj_matrix = cne.get_adj_matrix()
        x = cne.get_embeddings()
        s1 = cne.get_s1()
        s2 = cne.get_s2()
        if node_hessian_val is None:
            node_hessian_val = cls.node_hessian(adj_matrix, i, posterior_prob_i_row, s1, s2, x)
        h_i_inverse = np.linalg.pinv(node_hessian_val)

        return h_i_inverse

    @classmethod
    def get_partial_derivative_node_wrt_link(cls, hi_inverse, i, k_list, cne):
        x = cne.get_embeddings()
        s1 = cne.get_s1()
        s2 = cne.get_s2()
        gamma = 1 / s1 ** 2 - 1 / s2 ** 2
        return gamma * (x[k_list] - x[i]).dot(-hi_inverse)

    @classmethod
    def get_partial_derivative_prob_wrt_xi(cls, i, j, cne):
        s1 = cne.get_s1()
        s2 = cne.get_s2()
        prior = cne.get_prior().get_row_probability(i, [j])
        emb_diff = cne.get_embeddings()[i, :] - cne.get_embeddings()[j, :]
        d_p2 = np.sum((emb_diff.T) ** 2, axis=0)
        prob = cne._posterior(1, d_p2, prior, s1, s2)[0]

        s_div = s1/s2
        s_diff = (1/s1**2 - 1/s2**2)
        return (-s_div*((1-prior[0])/prior[0])*emb_diff*s_diff*np.exp(d_p2/2*s_diff))*(prob**2)

    @classmethod
    def node_gradient_v2(cls, i, j, fi_xi_gradient_val, fj_xj_gradient_val, p_ij, a_ij, x, s1, s2):
        """
        assume adj_matrix is symmetric
        :param i:
        :param j:
        :param fi_xi_gradient:
        :param fj_xj_gradient:
        :param p_ij:
        :param a_ij:
        :param x:
        :param s1:
        :param s2:
        :return:
        """
        n, d = x.shape
        gamma = 1 / s1 ** 2 - 1 / s2 ** 2
        h = np.zeros((2 * d, 2 * d))
        h[0:d, 0:d] = fi_xi_gradient_val
        h[d:, d:] = fj_xj_gradient_val
        f_ij = cls.fi_xj_gradient(d, gamma, i, j, p_ij, a_ij, x)
        h[d:, 0:d] = f_ij
        h[0:d, d:] = f_ij
        h_inverse = np.linalg.pinv(h)
        f_grad = np.zeros((2 * d,))
        diff = x[i, :] - x[j, :]
        f_grad[:d] = (gamma * diff)
        f_grad[d:] = - (gamma * diff)
        result = -1 * h_inverse.dot(f_grad)
        print(result)
        return result[:d]

    @classmethod
    def node_gradient_wrt_link(cls, hi_inverse, x, i, k_list, s1, s2):
        gamma = 1 / s1 ** 2 - 1 / s2 ** 2
        return gamma * (x[k_list] - x[i]).dot(-hi_inverse)
        # return -hi_inverse.dot(x[k] - x[i])

    @classmethod
    def node_hessian_inverse(cls, adj_matrix, i, posterior_prob_i_row, s1, s2, x, node_hessian_val=None):
        if node_hessian_val is None:
            node_hessian_val = cls.node_hessian(adj_matrix, i, posterior_prob_i_row, s1, s2, x)
        h_i_inverse = np.linalg.pinv(node_hessian_val)

        return h_i_inverse

    @classmethod
    def node_hessian(cls, adj_matrix, i, posterior_prob_i_row, s1, s2, x):
        gamma = 1 / s1 ** 2 - 1 / s2 ** 2
        h_i = cls.fi_xi_gradient(adj_matrix, gamma, i, posterior_prob_i_row, x)
        return h_i

    @classmethod
    def fi_xi_gradient(cls, adj_matrix, gamma, i, posterior_prob_i_row, x):
        n, d = x.shape
        h_i_part1 = np.identity(d) * np.sum(
            posterior_prob_i_row[[j for j in range(len(posterior_prob_i_row)) if j != i]] - adj_matrix[
                i, [j for j in range(len(posterior_prob_i_row)) if j != i]])
        x_i_diff = x[i, :] - x
        p_ij_2 = np.multiply(posterior_prob_i_row, 1 - posterior_prob_i_row)
        h_i_part2 = np.zeros((d, d))
        for j in range(n):
            if j == i:
                continue
            h_i_part2 += np.outer(x_i_diff[j], x_i_diff[j]) * p_ij_2[j]
        h_i_part2 *= gamma
        # h_i_part2 = gamma * (np.sum([np.outer(x_i_diff[j], x_i_diff[j])*p_ij_2[j] for j in range(n)], axis=0))
        h_i = gamma * (h_i_part1 - h_i_part2)
        return h_i

    @classmethod
    def fi_xj_gradient(cls, d, gamma, i, j, p_ij, a_ij, x):
        h_i_part1 = -gamma * np.identity(d) * (p_ij - a_ij)
        diff = x[i, :] - x[j, :]
        h_i_part2 = (gamma ** 2) * np.outer(diff, diff) * (p_ij * (1 - p_ij))
        # h_i_part2 = gamma * (np.sum([np.outer(diff[j], diff[j])*p_ij_2[j] for j in range(n)], axis=0))
        h_i = h_i_part1 + h_i_part2
        return h_i
