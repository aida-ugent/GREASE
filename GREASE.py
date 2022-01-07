# from pyomo.environ import *
import random
import time

import gurobipy as gp
from gurobipy import GRB

from copy import deepcopy
import numpy as np


class GREASE():
    def __init__(self, node_file_start, pre_sparsify, candidate_link_strategy, candidate_list, cuts, no_rel_heur_time, c_method, time_limit, mip_focus, heuristics, strategy_list,
                 flow_variables_ratio, network_embedding_method, gradient_computer_class, nodes_type_dict, source_node_ids,
                 target_node_ids,
                 destination_node_ids, param_factory):
        self.pre_batch_list = list()
        self.graph = None

        self.gradient_computer_class = gradient_computer_class
        self.destination_node_ids = destination_node_ids
        self.target_node_ids = target_node_ids
        self.source_node_ids = source_node_ids
        self.param_factory = param_factory
        self.nodes_type_dict = nodes_type_dict
        self.network_embedding_method = deepcopy(network_embedding_method)

        self.candidate_list = candidate_list
        self.c_method = c_method
        self.no_rel_heur_time = no_rel_heur_time
        self.cuts = cuts
        self.time_limit = time_limit
        self.heuristics = heuristics
        self.mip_focus = mip_focus
        self.flow_variables_ratio = flow_variables_ratio
        self.strategy_list = strategy_list
        self.method = deepcopy(self.network_embedding_method.ne_method)
        self.candidate_link_strategy = candidate_link_strategy
        self.node_file_start = node_file_start
        self.pre_sparsify = pre_sparsify
        self.starting_point_variables = None



    def get_distances(self):
        distances_new = self.param_factory.get_distances(self.source_node_ids, self.target_node_ids,
                                                         embeddings=self.method.get_embeddings(), method=self.method)
        return distances_new

    def set_initial_values(self, starting_point_variables):
        self.starting_point_variables = starting_point_variables

    def get_initial_values(self):
        return self.starting_point_variables

    def get_links(self, link_count, round_number=0):
        """
        :param link_count:
        :return:
        """
        if round_number > 0: # for getting the intermediate results
            return self.optimize_get_results(link_count)
        original_distances = self.get_distances()
        temp_method = deepcopy(self.network_embedding_method.ne_method)
        original_emb = deepcopy(temp_method.get_embeddings())
        adj_matrix = deepcopy(temp_method.get_adj_matrix())
        sid_link_destination_costs_list = list()
        s_id_d_ids_dict = dict()
        start_time = time.time()
        for s_id, d_id in self.candidate_list:
            l = s_id_d_ids_dict.get(s_id, list())
            l.append(d_id)
            s_id_d_ids_dict[s_id] = l
        for s_idx, s_id in enumerate(self.source_node_ids):
            d = {'s_idx': s_idx, 's_id': s_id, 'links': list()}
            dest_ids = s_id_d_ids_dict.get(s_id, list())
            for dest_id in dest_ids:
                self._add_destination_link(adj_matrix, d, dest_id, original_emb, s_id, temp_method)
            sid_link_destination_costs_list.append(d)
            # if s_idx > 2:
            #     break

        print("start find_links_globally_combining_objectives")
        return self.find_links_globally_combining_objectives(original_distances, sid_link_destination_costs_list,
                                                             link_count)

    def _add_destination_link(self, adj_matrix, d, dest_id, original_emb, s_id, temp_method):
        adj_matrix[s_id, dest_id] = 1
        adj_matrix[dest_id, s_id] = 1
        temp_method.set_adj_matrix(adj_matrix)
        temp_method.embed_partially([s_id])
        d.get('links').append({'dest_id': dest_id,
                               'distances': -np.log(temp_method.get_posterior_row(s_id)[self.target_node_ids])})
        adj_matrix[s_id, dest_id] = 0
        adj_matrix[dest_id, s_id] = 0
        temp_method.set_adj_matrix(adj_matrix)
        temp_method.set_embeddings(deepcopy(original_emb))

    def find_links_globally_combining_objectives(self, original_distances, sid_link_destination_costs_list, k):
        link_mapping = dict()
        link_mapping_reverse = dict()
        s_idx_dict_mapping = dict()
        d1 = original_distances.shape[0]
        d2 = original_distances.shape[1]
        w_i = 1.0 / float(d1)
        w_j = 1.0 / float(d2)

        flow_indices = self._get_flow_indices(
            d1, d2, original_distances)

        a_count = 0
        for item in sid_link_destination_costs_list:
            s_idx_dict_mapping[item.get('s_idx')] = item
            for link in item.get("links"):
                link_mapping[self._get_link_mapping_key(item, link)] = a_count
                link_mapping_reverse[a_count] = (item, link)
                a_count += 1

        # Model
        m = gp.Model("GraBV2")

        f = m.addVars(flow_indices, vtype=GRB.CONTINUOUS, lb=0.0, ub=min(w_i, w_j), name="f")
        si = m.addVars([i for i in range(d1)], vtype=GRB.CONTINUOUS, lb=0.0, ub=w_i, name="si")
        sj = m.addVars([j for j in range(d2)], vtype=GRB.CONTINUOUS, lb=0.0, ub=w_j, name="sj")
        a = m.addVars([i for i in range(a_count)], vtype=GRB.BINARY, name="a")

        self._add_constraints(si, sj, a, d1, d2, f, flow_indices, k, link_mapping, m, s_idx_dict_mapping, w_i, w_j)

        self._add_objective(si, sj, a, d1, d2, f, flow_indices, link_mapping, m, original_distances, s_idx_dict_mapping)

        self._set_start_point(si, sj, d1, d2, f, a, a_count, flow_indices, original_distances, w_i, w_j)


        if 'time_limit' in self.strategy_list:
            m.Params.TimeLimit = self.time_limit
        m.Params.MIPFocus = self.mip_focus
        m.Params.Heuristics = self.heuristics
        m.Params.Method = self.c_method
        m.Params.Cuts = self.cuts
        m.Params.NoRelHeurTime = self.no_rel_heur_time
        m.Params.NodefileStart = self.node_file_start
        m.Params.PreSparsify = self.pre_sparsify

        # for getting the intermediate results
        self.m = m
        self.a = a
        self.a_count = a_count
        self.link_mapping_reverse = link_mapping_reverse
        return self.optimize_get_results(k)

    def optimize_get_results(self, k):
        print("start solver ...")
        self.m.optimize()
        print("end solver ...")
        result_links = self._get_result_links(self.a, self.a_count, k, self.link_mapping_reverse)
        return result_links

    def _get_result_links(self, a, a_count, k, link_mapping_reverse):
        result_links = list()
        for i in range(a_count):
            # print(a[i].x)
            if a[i].x > 0.5:
                item, link = link_mapping_reverse.get(i)
                result_links.append((item.get('s_id'), link.get('dest_id')))
                if len(result_links) >= k:
                    break
        print("results_links length: ")
        print(len(result_links))
        return result_links

    def _add_objective(self, si, sj, a, d1, d2, f, flow_indices, link_mapping, m, original_distances,
                       s_idx_dict_mapping):
        M = 1000
        expr_list = []
        for i, j in flow_indices:
            item = s_idx_dict_mapping.get(i)
            if item:
                associated_a_list = [a[link_mapping[self._get_link_mapping_key(item, link)]] for link in
                                     item.get('links')]
            else:
                associated_a_list = []
            expr_list.append(f[i, j] * original_distances[i, j])
            if associated_a_list:
                expr_list.append(
                    -1 * f[i, j] * original_distances[i, j] * gp.quicksum(
                        (associated_a for associated_a in associated_a_list)))
            if associated_a_list:
                for link in item.get('links'):
                    expr_list.append(
                        f[i, j] * a[
                            link_mapping[self._get_link_mapping_key(item, link)]] * link.get('distances')[
                            j])
        m.setObjective(gp.quicksum(expr_list)+M*(gp.quicksum(si[i] for i in range(d1))+gp.quicksum(sj[j] for j in range(d2))))

    def _add_constraints(self, si, sj, a, d1, d2, f, flow_indices, k, link_mapping, m, s_idx_dict_mapping, w_i, w_j):
        m.addConstr(a.sum() <= k)
        m.addConstrs((si[i] + gp.quicksum((f[i, j] for j in range(d2) if
                                   self.flow_variables_ratio >= 1 or (i, j) in flow_indices)) == w_i for i in
                      range(d1)))
        m.addConstrs(
            (sj[j] + gp.quicksum((f[i, j] for i in range(d1) if
                                   self.flow_variables_ratio >= 1 or (i, j) in flow_indices)) == w_j for j in
             range(d2)))
        if self.candidate_link_strategy not in ('tdl', 'rpdl'):
            for i in range(d1):

                item = s_idx_dict_mapping.get(i)
                if item and len(item.get("links")) > 1:
                    m.addConstr(gp.quicksum(
                        [a[link_mapping[self._get_link_mapping_key(item, link)]] for link in item.get('links')]) <= 1)

    def _set_start_point(self, si, sj, d1, d2, f, a, a_count, flow_indices, original_distances, w_i, w_j):
        if self.starting_point_variables is None:
            M = 1000
            model_name = str(random.getrandbits(128))
            m = gp.Model(model_name)
            f2 = m.addVars(flow_indices, vtype=GRB.CONTINUOUS, lb=0.0, name="f" + model_name)
            si2 = m.addVars([i for i in range(d1)], vtype=GRB.CONTINUOUS, lb=0.0, ub=w_i, name="si" + model_name)
            sj2 = m.addVars([j for j in range(d2)], vtype=GRB.CONTINUOUS, lb=0.0, ub=w_j, name="sj" + model_name)

            m.addConstrs((si2[i] + gp.quicksum((f2[i, j] for j in range(d2) if
                                       self.flow_variables_ratio >= 1 or (i, j) in flow_indices)) == w_i for i in
                          range(d1)))
            m.addConstrs(
                (sj2[j] + gp.quicksum((f2[i, j] for i in range(d1) if
                                       self.flow_variables_ratio >= 1 or (i, j) in flow_indices)) == w_j for j in
                 range(d2)))

            expr_list = []
            for i, j in flow_indices:
                expr_list.append(f2[i, j] * original_distances[i, j])
            m.setObjective(gp.quicksum(expr_list)+M*(gp.quicksum(si2[i] for i in range(d1))+gp.quicksum(sj2[j] for j in range(d2))))
            m.optimize()
        else:
            print("shortcut in set starting point")
            m, si2, sj2, f2 = self.starting_point_variables
        for i in range(d1):
            si[i] = si2[i].x
        for j in range(d2):
            sj[j] = sj2[j].x
        for i in range(d1):
            for j in range(d2):
                if self.flow_variables_ratio >= 1 or (i, j) in flow_indices:
                    f[i, j].start = f2[i, j].x
        for i in range(a_count):
            a[i].start = 0

        self.starting_point_variables = (m, si2, sj2, f2)

    def _get_flow_indices(self, d1, d2, original_distances):
        if self.flow_variables_ratio >= 1:
            flow_indices = [(i, j) for i in range(d1) for j in range(d2)]
        else:
            flow_indices = self._get_flow_indices_percent(d1, d2, original_distances)

        flow_indices = set(flow_indices)
        return flow_indices

    def _get_flow_indices_percent(self, d1, d2, original_distances):
        flow_indices = list()
        j_count_to_keep = int(float(d2) * self.flow_variables_ratio)
        i_count_to_keep = int(float(d1) * self.flow_variables_ratio)
        for i in range(d1):
            flow_indices += [(i, j) for j in original_distances[i, :].argsort()[:j_count_to_keep]]
        for j in range(d2):
            flow_indices += [(i, j) for i in original_distances[:, j].argsort()[:i_count_to_keep]]
        return flow_indices

    @classmethod
    def _get_link_mapping_key(cls, item, link):
        return str(item.get('s_idx')) + '-' + str(link.get('dest_id'))
