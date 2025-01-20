# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Imports
import math
import os
import argparse
from typing import List
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import pickle
import random
import math
import time
import pprint
from constants import DIST_METRIC_INDEX
from pathlib import Path


def init_memmap_embs(
    emb_memory_loc: str, dataset_size: int, emd_size: int = 512, dtype: str = "float32"
) -> np.memmap:
    """
    Initializes a memory-mapped NumPy array to read embeddings of examples.

    Args:
        emb_memory_loc (str): Path to the memory-mapped file.
        dataset_size (int): Size of the dataset.
        emd_size (int): Dimensionality of the embeddings.
        dtype (str): Data type of the embeddings.

    Returns:
        np.memmap: A memory-mapped NumPy array.
    """
    embs = np.memmap(
        emb_memory_loc, dtype=dtype, mode="r", shape=(dataset_size, emd_size)
    )
    return embs


# Each SLURMJob will run SemDeDup on number of clusters and save dataframe with which examples to keep from each cluster.
# - Parallelize shards across jobs so that preemption in the middle of an epoch isn't a problem and because we want to
#   keep the shard structure anyway.
# - Process more than one cluster per job=> run multiple taks inside each jobs.
# - Preempted jobs get resubmitted. Already precessed clusters get skipped internally.
class SemDeDup:
    def __init__(self, args):
        self.args = args
        random.seed(args.seed)

    def _contains_duplicates(self, arr):
        return len(np.unique(arr)) != len(arr)

    def semdedup(self, cluster, cluster_reps):
        start_time = time.time()
        ## -- compute pairwise cos sim between cluster items, then replace to diagonal with zeros to ignore self similarity
        pair_w_sim_matrix = cluster_reps @ (cluster_reps.T)
        pair_w_sim_matrix.fill_diagonal_(0.0)
        assert pair_w_sim_matrix.shape[0] == pair_w_sim_matrix.shape[1]

        ## -- get paths to cluster i images
        image_urls = cluster[:, 0]

        ## -- make sure all the paths are unique this ensure that the duplicates are really stored many time times on memory
        assert not self._contains_duplicates(image_urls)

        ## -- 2) compute the sum of all pairwise sim values exept the diagonal (diagonal items = 1.0)
        avg_sim_to_others_list = (1 / (pair_w_sim_matrix.shape[0] - 1)) * (
            torch.sum(pair_w_sim_matrix, dim=0)
        )  # -- array of shape (cluster_size x 1)

        ##-- 3) compute max pairwise similarity
        max_pair_w_sim_list = torch.max(pair_w_sim_matrix, dim=0)[
            0
        ]  # -- array of shape (cluster_size x 1)
        min_pair_w_sim_list = torch.min(pair_w_sim_matrix, dim=0)[
            0
        ]  # -- array of shape (cluster_size x 1)
        std_pair_w_sim = pair_w_sim_matrix.std()

        ## -- 4) average value of cos similarity to cluster centroid
        avg_sim_to_cent = (1 - cluster[:, DIST_METRIC_INDEX].astype("float32")).mean()
        std_sim_to_cent = (1 - cluster[:, DIST_METRIC_INDEX].astype("float32")).std()

        ## -- We need upper tringular matrix because (1)we don't need to look at self sim (always=1) (2)we need the compinations not permutations
        triu_sim_mat = torch.triu(pair_w_sim_matrix, diagonal=1)
        del pair_w_sim_matrix
        # pair_w_sim_matrix[lower_tri_ids] = 0

        ## -- if the max sim between one example and any other example is > 1-eps, remove this example
        M = torch.max(triu_sim_mat, dim=0)[0]
        print(f"Step time: {time.time()-start_time}(s)")

        return (
            M,
            avg_sim_to_others_list,
            max_pair_w_sim_list,
            min_pair_w_sim_list,
            std_pair_w_sim,
            avg_sim_to_cent,
            std_sim_to_cent,
        )

    def _process_shard(self):
        print("SemDeDup params: ", self.args)
        start_time = time.time()

        start, end = 0, self.args.ncentroids
        print(f"This task will process cluster {start} to cluster {end}")

        embs = init_memmap_embs(
            self.args.emb_memory_loc,
            self.args.dataset_size,
            self.args.emb_size,
            dtype="float32",
        )
        statistics_df = pd.DataFrame(
            columns=[
                "cluster_size",
                "cluster_id",
                "avg_sim_to_cent",
                "std_sim_to_cent",
                "std_pair_w_sim",
                "avg_sim_to_others_list",
                "max_pair_w_sim_list",
                "min_pair_w_sim_list",
            ]
        )

        eps_df_dicts = {
            eps: pd.DataFrame(
                columns=["duplicates_ratio", "num_duplicates", "cluster_id"]
            )
            for eps in self.args.eps_list
        }

        eps_dict_file_loc = os.path.join(
            self.args.save_folder, f"statistics/dicts/shard_{start}.pt"
        )
        statistics_df_file_loc = os.path.join(
            self.args.save_folder, f"statistics/dataframes/shard_{start}.pkl"
        )
        Path(eps_dict_file_loc).parent.mkdir(parents=True, exist_ok=True)
        Path(statistics_df_file_loc).parent.mkdir(parents=True, exist_ok=True)

        step_time = []

        for cluster_id in tqdm(range(start, end)):
            step_start_time = time.time()

            # dict_file_loc = os.path.join(self.args.save_folder, f"dicts/cluster_{cluster_id}.pt")
            df_file_loc = os.path.join(
                self.args.save_folder, f"dataframes/cluster_{cluster_id}.pkl"
            )

            if os.path.exists(df_file_loc):  # and os.path.exists(dict_file_loc):
                print(f"{df_file_loc} exists, moving on")
                continue
            else:
                print(f"{df_file_loc} does not exist, creating")
                Path(df_file_loc).parent.mkdir(parents=True, exist_ok=True)

            ## -- load cluster i representations
            cluster_i = np.load(
                os.path.join(
                    self.args.sorted_clusters_file_loc, f"cluster_{cluster_id}.npy"
                )
            )
            # 1) store cluster size
            cluster_size = cluster_i.shape[0]
            print("cluster_size: ", cluster_size)

            if cluster_size == 1:
                points_to_remove_df = pd.DataFrame()
                points_to_remove_df["indices"] = [0]
                for eps in self.args.eps_list:
                    ## We need to remove a point from the dataset when its pairwise similarity to other point is > 1-ebs
                    points_to_remove_df[f"eps={eps}"] = [False]
                if self.args.save_folder != "":
                    ## --save df
                    with open(df_file_loc, "wb") as file:
                        pickle.dump(points_to_remove_df, file)
                print("DONE cluster_id ", cluster_id)
                continue

            ## -- By default, we keep hard examples from groups
            clutser_items_indices = list(range(cluster_size))
            ## -- OR: shuffle cluster to keep random example from each group
            if self.args.which_to_keep.lower() == "random":
                random.shuffle(clutser_items_indices)
                cluster_i = cluster_i[clutser_items_indices]
            ## -- OR: reverse cluster to keep easy examples
            if self.args.which_to_keep.lower() == "easy":
                clutser_items_indices = clutser_items_indices[::-1]
                cluster_i = cluster_i[clutser_items_indices]

            ## -- indices for cluster items in the dataset
            cluster_ids = cluster_i[:, 1].astype("int32")
            cluster_reps = embs[cluster_ids]
            cluster_reps = torch.tensor(cluster_reps)

            avg_sim_to_others_list = torch.tensor([])
            max_pair_w_sim_list = torch.tensor([])
            min_pair_w_sim_list = torch.tensor([])
            std_pair_w_sim = 0
            avg_sim_to_cent = 0
            std_sim_to_cent = 0
            M = torch.tensor([])

            # half_size = [0, cluster_size//4, cluster_size//2, cluster_size*3//4, cluster_size]
            num_small_clusters = (
                math.ceil(cluster_size / self.args.largest_cluster_size_to_process) + 1
            )
            cluster_part_ids = np.linspace(
                0, cluster_size, num_small_clusters, dtype="int64"
            )
            for i in range(len(cluster_part_ids) - 1):

                (
                    tem_M,
                    tem_avg_sim_to_others_list,
                    tem_max_pair_w_sim_list,
                    tem_min_pair_w_sim_list,
                    tem_std_pair_w_sim,
                    tem_avg_sim_to_cent,
                    tem_std_sim_to_cent,
                ) = self.semdedup(
                    cluster_i,
                    cluster_reps[cluster_part_ids[i] : cluster_part_ids[i + 1]],
                )

                avg_sim_to_others_list = torch.cat(
                    (avg_sim_to_others_list, tem_avg_sim_to_others_list)
                )
                max_pair_w_sim_list = torch.cat(
                    (max_pair_w_sim_list, tem_max_pair_w_sim_list)
                )
                min_pair_w_sim_list = torch.cat(
                    (min_pair_w_sim_list, tem_min_pair_w_sim_list)
                )
                std_pair_w_sim += tem_std_pair_w_sim
                avg_sim_to_cent = (
                    tem_avg_sim_to_cent  #  (1-cluster_i[:, 2].astype('float32')).mean()
                )
                std_sim_to_cent = (
                    tem_std_sim_to_cent  #  (1-cluster_i[:, 2].astype('float32')).std()
                )
                M = torch.cat((M, tem_M))

            std_pair_w_sim /= len(cluster_part_ids)
            points_to_remove_df = pd.DataFrame()
            points_to_remove_df["indices"] = clutser_items_indices

            ## -- load files when they exist
            # with open(df_file_loc, 'rb') as file:
            #     points_to_remove_df = pickle.load(file)

            # num_duplicates_in_cluster_i = torch.load(dict_file_loc)

            # values_for_eps = {}

            for eps in self.args.eps_list:
                ## -- 5) We need to remove a point from the dataset when its pairwise similarity to other point is > 1-ebs
                eps_points_to_remove = M > 1 - eps
                points_to_remove_df[f"eps={eps}"] = eps_points_to_remove

                ## -- 6) num duplicates in this cluster
                eps_num_duplicates = sum(eps_points_to_remove).item()

                ## -- 7) duplicates ratio %
                eps_duplicates_ratio = 100 * eps_num_duplicates / cluster_size
                ## -- 8) number of similar points to each point (group size, including the item)
                # eps_num_duplicates_for_each_point_list = 1 + torch.sum(pair_w_sim_matrix>1-eps, dim=0) # -- array of shape (cluster_size x 1)
                ## -- store all the value computed for this eps
                eps_df_dicts[eps] = pd.concat(
                    [
                        eps_df_dicts[eps],
                        pd.DataFrame(
                            {
                                "duplicates_ratio": eps_duplicates_ratio,
                                "num_duplicates": eps_num_duplicates,
                                "cluster_id": cluster_id,
                            },
                            index=range(cluster_size),
                        ),
                    ]
                )

            # num_duplicates_in_cluster_i = {
            #                                 'values_for_eps': values_for_eps,
            #                                 'cluster_size': cluster_size,
            #                                 'cluster_id': cluster_id,
            #                                 'avg_sim_to_cent': avg_sim_to_cent,
            #                                 'std_sim_to_cent': std_sim_to_cent,
            #                                 'std_pair_w_sim': std_pair_w_sim,
            #                                 'avg_sim_to_others_list': avg_sim_to_others_list,
            #                                 'max_pair_w_sim_list': max_pair_w_sim_list,
            #                                 'min_pair_w_sim_list': min_pair_w_sim_list
            #                                 }
            statistics_df = pd.concat(
                [
                    statistics_df,
                    pd.DataFrame(
                        {
                            "cluster_size": cluster_size,
                            "cluster_id": cluster_id,
                            "avg_sim_to_cent": avg_sim_to_cent,
                            "std_sim_to_cent": std_sim_to_cent,
                            "std_pair_w_sim": std_pair_w_sim,
                            "avg_sim_to_others_list": [avg_sim_to_others_list],
                            "max_pair_w_sim_list": [max_pair_w_sim_list],
                            "min_pair_w_sim_list": [min_pair_w_sim_list],
                        }
                    ),
                ]
            )

            if self.args.save_folder != "":
                ## -- save dict
                # torch.save(captions_dict, dict_file_loc)
                ## --save df
                with open(df_file_loc, "wb") as file:
                    pickle.dump(points_to_remove_df, file)

            step_time.append(time.time() - step_start_time)
            print("step_time", step_time)
            print("DONE cluster: ", cluster_id)

        if self.args.save_folder != "":
            torch.save(eps_df_dicts, eps_dict_file_loc)
            with open(statistics_df_file_loc, "wb") as file:
                pickle.dump(statistics_df, file)

        print("DONE step_time", step_time)
        if len(step_time) != 0:
            print(
                f"DONE in {((time.time()-start_time)/60):.2f} minutes, Average Step time {(sum(step_time)/len(step_time)):.2f}(s)"
            )
        return

    def __call__(self):
        pp = pprint.PrettyPrinter(indent=4)
        pp.pprint(vars(self.args))
        self._process_shard()


if __name__ == "__main__":
    # Configure command line arguments
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--sim_metric",
        type=str,
        default="cosine",
        choices=["cosine", "l2"],
        help="similarity metric",
    )
    parser.add_argument(
        "--keep_hard", type=bool, default=True, help="keep hard examples"
    )
    parser.add_argument(
        "--Kmeans_with_cos_dist",
        type=bool,
        default=True,
        help="use cosine similarity for kmeans clustering",
    )
    parser.add_argument(
        "--emb_memory_loc", type=str, required=True, help="embedding memory location"
    )
    parser.add_argument(
        "--text_emb_memory_loc",
        type=str,
        default=None,
        help="text embedding memory location",
    )
    parser.add_argument(
        "--paths_memory_loc", type=str, default=None, help="paths memory location"
    )
    parser.add_argument(
        "--sorted_clusters_file_loc",
        type=str,
        default="<your path>/sorted_clusters",
        help="sorted clusters file location",
    )
    parser.add_argument(
        "--save_folder", type=str, default="<your path>", help="save folder"
    )
    parser.add_argument(
        "--path_str_dtype", type=str, default="U10000", help="path string data type"
    )
    parser.add_argument(
        "--ncentroids", type=int, default=50000, help="number of centroids"
    )
    parser.add_argument("--dataset_size", type=int, default=None, help="dataset size")
    parser.add_argument("--emb_size", type=int, default=None, help="embedding size")
    parser.add_argument("--niter", type=int, default=100, help="number of iterations")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument(
        "--largest_cluster_size_to_process",
        type=int,
        default=10000000,
        help="largest cluster size to process",
    )
    parser.add_argument(
        "--eps_list",
        nargs="+",
        type=float,
        default=[0.00095, 0.03, 0.07],
        help="list of epsilons",
    )
    parser.add_argument(
        "--which_to_keep",
        type=str,
        default="hard",
        choices=["hard", "random", "easy"],
        help="which examples to keep in a duplicated group",
    )
    args = parser.parse_args()

    semdedup = SemDeDup(args)
    semdedup()

    print("Done!")
