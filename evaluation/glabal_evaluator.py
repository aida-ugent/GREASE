from imbalance_calculation.fractional_perfect_bmatching_calculator import fpbm_new


class GlobalEvaluator:
    @classmethod
    def evaluate(cls, embeddings, method, source_node_ids, target_node_ids, param_factory,
                 metrics=None):
        """

        :param metrics: a list. Options: emd, svm_rbf, emd_prob, sum_kde_log_ratio, emd_prob_cne_uniform_prior, emd_prob_cne_block_prior
        :param cost: emb_distance, neg_log_prob
        :return:
        """
        result = dict()
        if 'fpbm' in metrics:
            result['fpbm'] = cls.fpbm(embeddings, method, source_node_ids, target_node_ids, param_factory=param_factory)
        if 'emd' in metrics:
            result['emd'] = cls.emd(embeddings, method, source_node_ids, target_node_ids, param_factory)
        if 'emd_prob' in metrics:
            result['emd_prob'] = cls.emd_prob(embeddings, method, source_node_ids, target_node_ids, param_factory=param_factory)
        return result

    @staticmethod
    def emd(embeddings, method, source_node_ids, target_node_ids, param_factory):
        # emd_new = emd_calculator_optimized_kmeans(embeddings[source_node_ids], embeddings[target_node_ids])
        distances_new = param_factory.get_distances(source_node_ids, target_node_ids, embeddings=embeddings,
                                                    method=method)
        emd_new = fpbm_new(distances_new)
        print("*********************************************")
        print("cost: emb_distance")
        print("emd: %.7f" % (emd_new))
        print("*********************************************")
        return emd_new

    @staticmethod
    def fpbm(embeddings, method, source_node_ids, target_node_ids, param_factory):
        distances_new = param_factory.get_distances(source_node_ids, target_node_ids, embeddings=embeddings, method=method)
        fpbm_val = fpbm_new(distances=distances_new)
        print("*********************************************")
        print("cost: %s" % param_factory.get_cost_type())
        print("fpbm: %.7f" % (fpbm_val))
        print("*********************************************")
        return fpbm_val

    @staticmethod
    def emd_prob(embeddings, method, source_node_ids, target_node_ids, param_factory):
        """
        the same as fpbm
        """
        distances_new = param_factory.get_distances(source_node_ids, target_node_ids, embeddings=embeddings, method=method)
        emd_new = fpbm_new(distances=distances_new)
        print("*********************************************")
        print("cost: %s" % param_factory.get_cost_type())
        print("emd_prob: %.7f" % (emd_new))
        print("*********************************************")
        return emd_new
