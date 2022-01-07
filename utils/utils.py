import pandas as pd
import base64
import io
import numpy as np


class Utils:
    @classmethod
    def fig_to_base64(cls, fig):
        if fig:
            img = io.BytesIO()
            fig.savefig(img, format='png',
                        bbox_inches='tight')
            img.seek(0)
            # print(base64.b64encode(img.getvalue()))
            return base64.b64encode(img.getvalue())
        else:
            return base64.b64encode(b'')

    @classmethod
    def get_pareto_optimal_points(cls, points_ndarray):
        n, d = points_ndarray.shape
        l = list()
        for idx, e1 in enumerate(points_ndarray):
            max_dimensions_worse_than_others = np.max(np.sum(e1 < points_ndarray, axis=1))
            if max_dimensions_worse_than_others == d:
                l.append(0)
            else:
                l.append(1)
        return l

    @staticmethod
    def find_optimal_cutoff(target, predicted):
        """ Find the optimal probability cutoff point for a classification model related to event rate
        Parameters
        ----------
        target : Matrix with dependent or target data, where rows are observations

        predicted : Matrix with predicted data, where rows are observations

        Returns
        -------
        list type, with optimal cutoff value

        """
        from sklearn.metrics import roc_curve
        fpr, tpr, threshold = roc_curve(target, predicted)
        i = np.arange(len(tpr))
        roc = pd.DataFrame({'tf': pd.Series(tpr - (1 - fpr), index=i), 'threshold': pd.Series(threshold, index=i)})
        roc_t = roc.iloc[(roc.tf - 0).abs().argsort()[:1]]

        return list(roc_t['threshold'])[0]

    @staticmethod
    def get_neg_log_prob_distances(method, source_node_ids, target_node_ids):
        distances_new = np.zeros((len(source_node_ids), len(target_node_ids)))
        for idx, src_id in enumerate(source_node_ids):
            distances_new[idx] = -np.log(method.get_posterior_row(src_id)[target_node_ids])
        return distances_new

    @staticmethod
    def get_embedding_distances(source_embedding, target_embedding):
        H = source_embedding.shape[0]
        I = target_embedding.shape[0]

        distances = np.zeros((H, I))
        for i in range(H):
            for j in range(I):
                distances[i][j] = np.linalg.norm(source_embedding[i] - target_embedding[j])
        return distances

    @staticmethod
    def get_reembedding_max_iter(args_max_iter, multi_reembedding_batch_count):
        cne_max_iter = int(args_max_iter)
        if multi_reembedding_batch_count:
            cne_max_iter = int(max(min(cne_max_iter, multi_reembedding_batch_count * (int(args_max_iter) / 25)),
                                   2.5 * (int(args_max_iter) / 25)))
        return cne_max_iter
