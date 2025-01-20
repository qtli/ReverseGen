

import os
import yaml
import pprint
import logging
import argparse
import numpy as np
from pathlib import Path
from clustering.utils import get_logger
from clustering.clustering import compute_centroids


def main(args):
    # Initialize logger
    log_file = os.path.join(args.save_folder, "compute_centroids.log")
    logger = get_logger(
        file_name=log_file,
        level=logging.INFO,
        stdout=True,
    )

    with open(Path(args.save_folder, "clustering_params.txt"), "w") as fout:
        pprint.pprint(args, fout)

    ## -- Load embeddings
    emb_memory = np.memmap(
        args.emb_memory_loc,
        dtype="float32",  # todo: float32 or float16
        mode="r",
        shape=(args.dataset_size, args.emb_size),
    )

    ## -- Compute centroids
    compute_centroids(
        data=emb_memory,
        ncentroids=args.ncentroids,
        niter=args.niter,
        nredo=args.nredo,
        seed=args.seed,
        Kmeans_with_cos_dist=args.Kmeans_with_cos_dist,
        save_folder=args.save_folder,
        logger=logger,
        verbose=True,
    )


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
    parser.add_argument(
        "--nredo",
        type=int,
        default=1,
        help="repeat clustering for nredo times, before selecting the best",
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    args = parser.parse_args()

    main(args)
