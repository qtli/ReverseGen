
import random
import argparse
import os
from tqdm import tqdm
import pickle
import numpy as np
from constants import IMAGE_NAME_INDEX
import json
import re

def extract_pruned_data(
    sorted_clusters_path,
    semdedup_pruning_tables_path,
    eps,
    num_clusters,
    output_txt_path,
    retreive_kept_samples=True,
):

    ## -- list of paths to the examples we want to keep/remove.
    example_paths = []

    for cluster_id in tqdm(range(0, num_clusters)):

        cluster_i = np.load(
            os.path.join(sorted_clusters_path, f"cluster_{cluster_id}.npy")
        )
        with open(
            f"{semdedup_pruning_tables_path}/cluster_{cluster_id}.pkl", "rb"
        ) as file:
            semdedup_pruning_tables = pickle.load(file)
        ## -- See which examples to keep/remove from this cluster.
        ## -- Use retreive_kept_samples=True when kept dataset size <= 50%. This will return a smaller output text file,
        ## -- semdedup_pruning_tables contain True values for the examples to be removed.
        images_to_keep_or_remove = semdedup_pruning_tables[f"eps={eps}"][
            semdedup_pruning_tables[f"eps={eps}"] == (not retreive_kept_samples)
        ].index.to_numpy()
        if "indices" in semdedup_pruning_tables.columns:
            cluster_i = cluster_i[semdedup_pruning_tables["indices"]]
        ## -- retrieve only the examples we want and add to the list.
        dedup_cluster = cluster_i[images_to_keep_or_remove]
        example_paths += dedup_cluster[:, IMAGE_NAME_INDEX].astype("<U32").tolist()

    with open(output_txt_path, "w") as fp:
        fp.write("\n".join(example_paths))

    print(f"DONE saving {len(example_paths)} image paths")

    return


def reformulate_pruned_data(
    eps,
    unfiltered_file,
    filtered_file,
):
    raw_data = json.load(open(unfiltered_file))
    print("raw data size: ", len(raw_data))


    item_dedup = []
    for line in open(filtered_file, encoding="utf-8").readlines():
        regex = r"index (\d+):"
        match = re.search(regex, line)
        if match:
            number = match.group(1)
        else:
            print("No match found.")
        item = raw_data[int(number)]
        item_dedup.append(item)

    random.shuffle(item_dedup)
    print("after filtering: ", len(item_dedup))
    formulated_filtered_file = ".".join(unfiltered_file.split(".")[:-1]) + f"_filtered_by_eps{eps}.json"
    json.dump(item_dedup, open(formulated_filtered_file, "w"), indent=2)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Parameters for extracting dedup data."
    )

    parser.add_argument(
        "--raw_file_path",
        type=str,
        help="Path to the unfiltered raw file.",
    )
    parser.add_argument(
        "--formulated_output_file_path",
        type=str,
        help="Path to the output text file.",
    )
    parser.add_argument(
        "--output_txt_path",
        type=str,
        help="Path to the output text file.",
    )
    parser.add_argument(
        "--semdedup_pruning_tables_path",
        type=str,
        help="Path to the semdedup pruning tables (dataframe).",
    )
    parser.add_argument(
        "--sorted_clusters_path",
        type=str,
        help="Path to the sorted clusters.",
    )
    parser.add_argument("--eps", type=float, help="Epsilon value for clustering.")
    parser.add_argument("--num_clusters", type=int, help="Number of clusters.")
    args = parser.parse_args()

    extract_pruned_data(
        sorted_clusters_path=args.sorted_clusters_path,
        semdedup_pruning_tables_path=args.semdedup_pruning_tables_path,
        eps=args.eps,
        num_clusters=args.num_clusters,
        output_txt_path=args.output_txt_path,
        retreive_kept_samples=True,
    )

    reformulate_pruned_data(
        eps=args.eps,
        unfiltered_file=args.raw_file_path,
        filtered_file=args.output_txt_path
    )
