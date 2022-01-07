import gurobipy as gp

import random
import time

import networkx as nx
import numpy as np
import argparse

from GREASE import GREASE
from gradient_computer.compute_embedding_gradient import CNEGradient
from embedding.embedding_computer import NetworkEmbeddingMethod
from evaluation.glabal_evaluator import GlobalEvaluator
from param_factory.param_factory import ParamFactory
from utils.utils import Utils
import os
os.environ["OMP_NUM_THREADS"] = "1" # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "1" # export OPENBLAS_NUM_THREADS=4


def main(args):
    try:
        network_embedding_method_name = args.network_embedding_method_name
    except:
        network_embedding_method_name = 'cne'
    nem = NetworkEmbeddingMethod(args, method=network_embedding_method_name)
    cne, embeddings, graph, node_types_dict = nem.compute_embeddings()
    print("connected components: %d " % len([_ for _ in nx.connected_components(graph)]))
    source_node_ids = list(sorted(node_types_dict.get(args.source_type).get('node_ids', [])))
    target_node_ids = list(sorted(node_types_dict.get(args.target_type).get('node_ids', [])))
    destination_node_ids = list(sorted(node_types_dict.get(args.destination_type).get('node_ids', [])))
    print("len(source_node_ids): %d " % len(source_node_ids))
    print("len(target_node_ids): %d" % len(target_node_ids))

    cost_type = _get_cost_type(args)
    param_factory = ParamFactory(cost_type)
    _evaluate(param_factory, embeddings, cne, source_node_ids, target_node_ids)

    adj_matrix = cne.get_adj_matrix()
    candidate_list = get_candidate_links(args.candidate_link_strategy, args.n_destination_link, source_node_ids,
                                         destination_node_ids, adj_matrix, cne)

    embedding_computer = CNEGradient
    grase = GREASE(args.node_file_start, args.pre_sparsify, args.candidate_link_strategy, candidate_list, args.cuts,
               args.no_rel_heur_time, args.c_method, args.time_limit,
               args.mip_focus, args.heuristics, args.strategies,
               args.flow_variables_ratio,
               nem,
               embedding_computer, node_types_dict, source_node_ids,
               target_node_ids,
               destination_node_ids, param_factory)

    start_time = time.time()
    selected_links = grase.get_links(args.link_count)
    end_time = time.time()
    _print_runtime(end_time, start_time)
    results = []
    embeddings_new, g, new_cne = _select_links_evaluate(param_factory, args.max_iter, graph, args.link_count, nem,
                                                                        results, selected_links, source_node_ids,
                                                                        target_node_ids)
    _print_final_evaluation(results)


def get_candidate_links(candidate_link_strategy, n_destination_link_orig, source_node_ids, destination_node_ids, adj_matrix, ne_method):
    candidate_links = list()
    for s_id in source_node_ids:
        dest_ids = []
        for dst_id in destination_node_ids:
            if adj_matrix[s_id, dst_id] > 0.5:
                continue
            dest_ids.append(dst_id)

        if dest_ids:
            if candidate_link_strategy == 'tdl':
                n_destination_link = 1
                # dest_id = np.array(dest_ids)[
                #     np.argmax(ne_method.get_posterior_row_cols(s_id, dest_ids))]
                # candidate_links.append((s_id, dest_id))
            elif candidate_link_strategy == 'tndl':
                n_destination_link = n_destination_link_orig
            if candidate_link_strategy == 'tdl' or candidate_link_strategy == 'tndl':
                candidate_dest_ids = np.array(dest_ids)[ne_method.get_posterior_row_cols(s_id, dest_ids).argsort()[-n_destination_link:][::-1]]
            elif candidate_link_strategy == 'all':
                candidate_dest_ids = dest_ids
            elif candidate_link_strategy == 'rpdl':
                candidate_dest_ids = [random.choice(dest_ids)]
            for dest_id in candidate_dest_ids:
                candidate_links.append((s_id, dest_id))
    return candidate_links


def _get_cost_type(args):
    try:
        cost_type = args.cost_type
    except:
        cost_type = 'emb_distance'
    return cost_type


def _print_final_evaluation(results):
    if results:
        result_sum = dict()
        for result in results:
            for key, val in result.items():
                if isinstance(val, tuple) or isinstance(val, list):
                    res = result_sum.get(key, [0 for _ in range(len(val))])
                    for idx, v in enumerate(val):
                        res[idx] += v
                    result_sum[key] = res
                else:
                    result_sum[key] = result_sum.get(key, 0) + val
        for key, val in result_sum.items():
            if isinstance(val, tuple) or isinstance(val, list):
                for idx, v in enumerate(val):
                    val[idx] /= float(len(results))
            else:
                result_sum[key] /= float(len(results))
        print(result_sum)


def _select_links_evaluate(param_factory, cne_max_iter, graph, link_count, nem, results, selected_links,
                           source_node_ids, target_node_ids):
    selected_links = selected_links[:link_count]
    print("selected_links: %s" % str(selected_links))
    print("selected_links length: %d" % len(selected_links))
    embeddings_new, g, new_cne = nem.apply_reembedding(graph,
                                                       selected_links, max_iter=cne_max_iter)
    # evaluation
    result_i = _evaluate(param_factory, embeddings_new, new_cne, source_node_ids, target_node_ids)
    results.append(
        result_i)
    print("selected_links")
    print(selected_links)
    print(len(selected_links))
    return embeddings_new, g, new_cne


def _evaluate(param_factory, embeddings_new, new_cne, source_node_ids, target_node_ids):
    if param_factory.get_cost_type() == "emb_distance":
        metrics = ['emd']
    elif param_factory.get_cost_type() == "neg_log_prob":
        metrics = ['fpbm']
    result_i = GlobalEvaluator.evaluate(embeddings_new, new_cne, source_node_ids, target_node_ids, param_factory=param_factory
                                        , metrics=metrics)
    return result_i


def _print_runtime(end_time, start_time):
    print("run_time: %d" % (end_time - start_time))


def parse_args():
    parser = argparse.ArgumentParser(description=".")
    # data params
    parser.add_argument('--data_directory', nargs='?',
                        default='data/weibo',
                        help='Input graph path')
    parser.add_argument('--delimiter', default=',',
                        help='The delimiter used to separate the edgelist.')
    parser.add_argument('--dimension', type=int, default=4,
                        help='')
    # network embedding params
    parser.add_argument('--k', type=float, default=100,
                        help='Sample size. Default is 100.')
    parser.add_argument('--s1', type=float, default=1,
                        help='Sigma 1. Default is 1.')
    parser.add_argument('--s2', type=float, default=2,
                        help='Sigma 2. Default is 2.')
    parser.add_argument('--prior', type=str, default='degree_per_block_eco',
                        help='cne prior: degree, degree_per_block, degree_per_block_eco, block or uniform')
    parser.add_argument('--max_iter', type=int, default=350,
                        help='max iter for CNE')
    parser.add_argument('--network_embedding_method_name', type=str, default='cne')

    # experiment params
    parser.add_argument('--save_method_pkl', type=int, default=1,
                        help='')
    parser.add_argument('--load_method_pkl', type=int, default=1,
                        help='')
    parser.add_argument('--method_pkl_file_name', type=str, default='cne4_degree_per_block_eco.pkl',
                        # cne_block_artificial_1.pkl cne2_degree_per_block_eco.pkl cne4_degree_per_block_eco.pkl
                        help='')
    parser.add_argument('--seed', type=int, default=1,
                        help='')
    parser.add_argument('--source_type', type=str, default='m',
                        help='')
    parser.add_argument('--target_type', type=str, default='f',
                        help='')
    parser.add_argument('--destination_type', type=str, default='T',
                        help='')
    parser.add_argument('--link_count', type=int, default=22,
                        help='comma separated link counts')
    parser.add_argument('--cost_type', type=str, default='neg_log_prob',
                        help='neg_log_prob or emb_distance')
    parser.add_argument('--candidate_link_strategy', type=str, default='tdl',
                        help='all, random_per_destination_link (rpdl), top_n_destination_link (tndl), top_destination_link (tdl)')
    parser.add_argument('--n_destination_link', type=int, default=5,
                        help='comma separated')

    # methods param
    parser.add_argument('--strategies', type=str, default='time_limit',
                        help='comma separated, options: time_limit,relax_int')
    parser.add_argument('--flow_variables_ratio', type=float, default=1.0,
                        help='ratio of top flow variables in matching to keep, comma separated')
    parser.add_argument('--mip_focus', type=float, default=3,
                        help='refer to https://www.gurobi.com/documentation/8.1/refman/mipfocus.html#parameter:MIPFocus')
    parser.add_argument('--time_limit', type=int, default=100,
                        help='time limit in seconds, comma separated')
    parser.add_argument('--heuristics', type=float, default=0.1,
                        help='amount of time spend on heuristics')
    parser.add_argument('--c_method', type=int, default=-1,
                        help='refer to https://www.gurobi.com/documentation/9.1/refman/method.html#parameter:Method')
    parser.add_argument('--node_file_start', type=float, default=0.5,
                        help='refer to https://www.gurobi.com/documentation/9.1/refman/nodefilestart.html#parameter:NodefileStart')
    parser.add_argument('--pre_sparsify', type=int, default=1,
                        help='refer to https://www.gurobi.com/documentation/9.5/refman/presparsify.html')
    parser.add_argument('--no_rel_heur_time', type=int, default=100,
                        help='refer to https://www.gurobi.com/documentation/9.1/refman/norelheurtime.html, , comma separated')
    parser.add_argument('--cuts', type=int, default=3,
                        help='refer to https://www.gurobi.com/documentation/9.1/refman/cuts.html#parameter:Cuts')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
