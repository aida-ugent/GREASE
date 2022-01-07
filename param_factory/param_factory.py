from utils.utils import Utils


class ParamFactory():
    def __init__(self, cost_type=None):
        if not cost_type:
            cost_type = 'emb_distance'
        self.cost_type = cost_type

    def get_cost_type(self):
        return self.cost_type

    def get_distances(self, source_node_ids, target_node_ids, embeddings=None, method=None):
        """
        :param source_node_ids:
        :param target_node_ids:
        :param method: is none for embedding distance
        :param embeddings: is none for neg_log_prob
        :return:
        """
        if self.cost_type == 'neg_log_prob':
            distances_new = Utils.get_neg_log_prob_distances(method, source_node_ids, target_node_ids)
        elif self.cost_type == 'emb_distance':
            distances_new = Utils.get_embedding_distances(embeddings[source_node_ids], embeddings[target_node_ids])
        return distances_new