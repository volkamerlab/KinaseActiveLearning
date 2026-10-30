import pandas as pd
import statistics
import random
from operator import itemgetter
from scipy.special import softmax
from modAL.utils.data import modALinput
from modAL.acquisition import BaseLearner

import utils


def compound_filtering(mols, affinity, n_candidates, used):
    """Filter compounds based on docking score and diversity

    Args:
        mols (list of RDKit): list of explored molecules
        affinity (list of float): list of predicted affinity scores
        n_candidates (ind): number of candidates to select
        used (list of int): list of indexes (of molecules) that have already been selected in previous cycles (to be excluded)

    Returns:
        list of int: list of compounds to be selected for docking (indexes of mols)
    """
    filtered_compounds = []
    ap_labels, clusters = fragment_clusters(mols, affinity, used)
    cluster_scores = calculate_cluster_scores(clusters, affinity)
    probabilities = convert_to_probabilities(cluster_scores)

    for i in range(n_candidates):
        # fewer compounds available than requested
        if not clusters:
            break
        cluster_index = random.choices(range(len(clusters)), probabilities)[0]
        compound_index = clusters[cluster_index].pop(0)
        filtered_compounds.append(compound_index)

        if len(clusters[cluster_index]) == 0:
            del clusters[cluster_index]
            del cluster_scores[cluster_index]
            del ap_labels[cluster_index]
        else:
            # update sampled cluster with new score
            cluster_scores[cluster_index] = recalculate_cluster_score(
                clusters, cluster_index, affinity
            )
        # update probabilities
        probabilities = convert_to_probabilities(cluster_scores)
    return filtered_compounds


def fragment_clusters(mols, affinity, used):
    """Cluster compounds based on AP fragment

    Args:
        mols (list of RDKit): explored molecules
        affinity (list of float): predicted affinity for all molecules in mols
        used (list of int): list of indexes for molecules already validated (to be excluded)

    Returns:
        list of str: list of AP fragment names
        list of list of int: clusters, each cluster contains the indexes of molecules (mols). The content of each cluster is sorted according to predicted affinity
    """
    clusters = []
    ap_labels = []
    print(f"len mols: {len(mols)}, len affinity: {len(affinity)}")
    df = pd.DataFrame(
        {
            "ROMol": mols,
            "y": affinity,
            "frag_ids": [
                utils.sort_ligand_name(mol.GetProp("frag_ids")) for mol in mols
            ],
        }
    )
    df["AP"] = [frag.split(",")[0] for frag in list(df["frag_ids"])]
    ap_fragments = df.groupby("AP")
    for ap, ap_mols in ap_fragments:
        sorted = ap_mols.sort_values(by=["y"], ascending=False)
        new_cluster = [ind for ind in list(sorted.index) if ind not in used]
        if len(new_cluster) == 0:
            continue
        clusters.append(new_cluster)
        ap_labels.append(ap)
    return ap_labels, clusters


def calculate_cluster_scores(clusters, affinity):
    """Calculate representative score for each cluster (mean predicted affinity)

    Args:
        clusters (list of list of int): clusters based on AP fragments
        affinity (list of float): list of predicted affinity for explored molecules

    Returns:
        list of float: list of cluster scores
    """
    scores = []
    for cluster in clusters:
        cluster_scores = [affinity[i] for i in cluster]
        scores.append(statistics.mean(cluster_scores))
    return scores


def recalculate_cluster_score(clusters, index, affinity):
    """Recalculates the cluster score for one cluster (mean predicted affinity)

    Args:
        clusters (list of list of int): clusters based on AP fragments
        index (int): cluster index for which the score should be recalculated, since it might have changed
        affinity (list of float): list of predicted affinity for explored molecules

    Returns:
        float: new cluster score
    """
    cluster_scores = [affinity[i] for i in clusters[index]]
    return statistics.mean(cluster_scores)


def convert_to_probabilities(scores):
    """Converts the cluster scores into probabilities by applying a softmax

    Args:
        scores (list of float): list of cluster scores

    Returns:
        list of float: probabilities for each cluster score
    """
    return softmax([scores])[0]


def find_best_idx(predicted, used, num_to_choose):
    tmp_list = list(enumerate(predicted))
    tmp_list.sort(key=itemgetter(1), reverse=True)
    tmp_list = [x for x in tmp_list if x[0] not in used]
    tmp_list = [x[0] for x in tmp_list]
    return tmp_list[:num_to_choose]


def find_random_idx(predicted, used, num_to_choose):
    tmp_list = list(enumerate(predicted))
    random.shuffle(tmp_list)
    tmp_list = [x for x in tmp_list if x[0] not in used]
    tmp_list = [x[0] for x in tmp_list]
    return tmp_list[:num_to_choose]


def greedy(optimizer: BaseLearner, X: modALinput, n_instances=1, used=[], mols=[]):
    res = optimizer.predict(X)
    best_idx = find_best_idx(res, used, n_instances)
    return best_idx, [X[i] for i in best_idx]


def sample_diverse(
    optimizer: BaseLearner, X: modALinput, n_instances=1, used=[], mols=[]
):
    res = optimizer.predict(X)
    best_idx = compound_filtering(mols, res, n_instances, used)
    return best_idx, [X[i] for i in best_idx]


def random_selection(
    optimizer: BaseLearner, X: modALinput, n_instances=1, used=[], mols=[]
):
    """Baseline: picks molecules uniformly at random, without consulting the model.

    The model is still taught every cycle, so its predictions can be compared with the
    other strategies, but it has no influence on which molecules are sent to the oracle.
    """
    best_idx = find_random_idx(range(len(X)), used, n_instances)
    return best_idx, [X[i] for i in best_idx]


# import os
# import utils

# HERE = os.getcwd()
# affinity_file = (
#    HERE + "/data/recombination_library/final_recombined_library_skip_hyde.sdf"
# )
# mols = utils.read_sdf(affinity_file)

# subset_mols = mols[:500]
# for mol in subset_mols:
#    mol.SetProp(
#        "frag_ids",
#        utils.sort_ligand_name("_".join(mol.GetProp("_Name").split("_")[:-1])),
#    )
# subset_affinities = [
#    float(mol.GetProp("BIOSOLVEIT.DOCKING_SCORE")) for mol in subset_mols
# ]

# indexes = compound_filtering(
#    subset_mols, subset_affinities, 50, [1, 2, 3, 5, 7, 9, 10, 50, 23, 125]
# )
# print(indexes)
# mols_selected = utils.indexing(subset_mols, indexes)
# print([mol.GetProp("frag_ids") for mol in mols_selected])
# print([float(mol.GetProp("BIOSOLVEIT.DOCKING_SCORE")) for mol in mols_selected])

# print(sorted(subset_affinities, reverse=True)[:100])
