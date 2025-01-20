import json
import argparse
import os
import pdb
import re
import random
import numpy as np
from tqdm import tqdm

import sys
from collections import defaultdict
import binascii

def getTriangleIndex(i, j, numDocs=1000):
    # If i == j that's an error.
    if i == j:
        sys.stderr.write("Can't access triangle matrix with i == j")
        sys.exit(1)
    # If j < i just swap the values.
    if j < i:
        temp = i
        i = j
        j = temp

    # Calculate the index within the triangular array.
    # This fancy indexing scheme is taken from pg. 211 of:
    # http://infolab.stanford.edu/~ullman/mmds/ch6.pdf
    # But I adapted it for a 0-based index.
    # Note: The division by two should not truncate, it needs to be a float.
    k = int(i * (numDocs - (i + 1) / 2.0) + j - i) - 1
    return k


def pickRandomCoeffs(k, maxShingleID=2 ** 32 - 1):
    # Create a list of 'k' random values.
    randList = []
    while k > 0:
        # Get a random shingle ID.
        randIndex = random.randint(0, maxShingleID)
        # Ensure that each random number is unique.
        while randIndex in randList:
            randIndex = random.randint(0, maxShingleID)
        # Add the random number to the list.
        randList.append(randIndex)
        k = k - 1
    return randList


def deduplicate_minhash(queries, numHashes=20, threshold=0.9, ngram=2, similar_pair_file="minhash_pair_files_sft_04.txt"):
    print("Shingling articles...")
    docsAsShingleSets = {}  # a dictionary of the articles, mapping the article identifier (e.g., "t8470") to the list of shingle IDs that appear in the document.
    docsAsStrings = {}
    totalShingles = 0
    docNames = []


    numDocs = len(queries)
    print("size of the queries: ", numDocs)

    for i in range(0, numDocs):
        words = queries[i].strip().split(" ")
        docID = "D"+str(i)
        docNames.append(docID)

        # 'shinglesInDoc' will hold all of the unique shingle IDs present in the current document.
        shinglesInDoc = set()
        for index in range(0, len(words) - (ngram - 1)):
            # Construct the shingle text by combining n-gram words together
            shingle = words[index]
            for ng in range(1, ngram):
                shingle += " " + words[index + ng]

            # Hash the shingle to a 32-bit integer.
            crc = binascii.crc32(str.encode(shingle)) & 0xffffffff
            shinglesInDoc.add(crc)

        docsAsShingleSets[docID] = shinglesInDoc
        docsAsStrings[docID] = queries[i].strip()
        # Count the number of shingles across all documents.
        totalShingles = totalShingles + (len(words) - (ngram - 1))
    print('Shingling ' + str(numDocs))
    print('Average shingles per doc: %.2f' % (totalShingles / numDocs))

    # Calculate the number of elements needed in our triangle matrix
    numElems = int(numDocs * (numDocs - 1) / 2)
    # Initialize the empty list to store the similarity values.
    # 'estJSim' will be for the estimated Jaccard Similarities found by comparing the MinHash signatures.
    estJSim = [0 for _ in range(numElems)]

    # Record the maximum shingle ID that we assigned.
    maxShingleID = 2 ** 32 - 1
    # We need the next largest prime number above 'maxShingleID'.
    # I looked this value up here:  http://compoasso.free.fr/primelistweb/page/prime/liste_online_en.php
    nextPrime = 4294967311

    coeffA = pickRandomCoeffs(numHashes, maxShingleID=maxShingleID)
    coeffB = pickRandomCoeffs(numHashes, maxShingleID=maxShingleID)

    # List of documents represented as signature vectors
    signatures = []
    for docID in docNames:
        shingleIDSet = docsAsShingleSets[docID]
        signature = []

        for i in range(0, numHashes):
            minHashCode = nextPrime + 1

            for shingleID in shingleIDSet:
                hashCode = (coeffA[i] * shingleID + coeffB[i]) % nextPrime
                if hashCode < minHashCode:
                    minHashCode = hashCode

            signature.append(minHashCode)
        signatures.append(signature)

    for i in tqdm(range(0, numDocs)):
        # Get the MinHash signature for document i.
        signature1 = signatures[i]

        for j in range(i + 1, numDocs):
            signature2 = signatures[j]

            count = 0
            # Count the number of positions in the minhash signature which are equal.
            for k in range(0, numHashes):
                count = count + (signature1[k] == signature2[k])

            # Record the percentage of positions which matched.
            estJSim[getTriangleIndex(i, j, numDocs=numDocs)] = (count / numHashes)

    print("List of Document Pairs with J(d1,d2) more than", threshold)
    # print("Values shown are the estimated Jaccard similarity and the actual")
    # print("Jaccard similarity.\n")
    # print("                   Est. J   Act. J")

    wf_dict = []
    docs2neighbours = defaultdict(list)
    docs2distances = np.zeros((numDocs, numDocs))

    for i in tqdm(range(0, numDocs)):
        docs2distances[i, i] = 0.5 # 后续有上三角和下三角相加
        docs2neighbours[docsAsStrings[docNames[i]]] = []
        for j in range(i + 1, numDocs):  # 上三角的形状
            estJ = estJSim[getTriangleIndex(i, j, numDocs=numDocs)]
            docs2distances[i, j] = estJ
            if estJ >= threshold:
                wf_dict.append([estJ, docsAsStrings[docNames[i]], docsAsStrings[docNames[j]]])

    wf_dict = sorted(wf_dict, reverse=True, key=lambda k: k[0])

    if similar_pair_file != "":
        wf = open(similar_pair_file, "w")
        for item in wf_dict:
            wf.write(f"{item[0]}\t | {item[1]}\t | {item[2]}\n")

    print("size of docs2neighbours: ", len(docs2neighbours))
    return wf_dict



parser = argparse.ArgumentParser()
parser.add_argument("--dedup_mode", required=True, default="minhash", help="semdedup or minhash or combine")
parser.add_argument("--raw_file", required=True, help="Input raw file")
parser.add_argument("--output_dir", required=True, help="Output dir")
parser.add_argument("--filter_data_dir", required=True, help="Output dir")
parser.add_argument("--selected_eps", help="selected eps for semdedup")
parser.add_argument("--minhash_threshold", type=float, default=0.9, help="selected eps for semdedup")
parser.add_argument("--start_idx", type=int, help="")
parser.add_argument("--end_idx", type=int, help="")
parser.add_argument("--sample_times", type=int, help="")

args = parser.parse_args()


# selected_emb_filtered_file = os.path.join(args.filter_data_dir, f"data_eps{args.selected_eps}_questions.json")
# selected_emb_filtered_data = json.load(open(selected_emb_filtered_file))
# print("size of data after filtering with semdedup: ", len(selected_emb_filtered_data))
raw_data = []
with open(args.eval_file) as f:
    for item in json.load(f):
        if isinstance(item["predict_instruction"], list):
            raw_data.extend(item["predict_instruction"])
        else:
            raw_data.append(item["predict_instruction"])

wf_dict = deduplicate_minhash(
    queries=raw_data,
    numHashes=128,
    threshold=args.minhash_threshold,
    ngram=1,
    similar_pair_file=""
)

for item in wf_dict:
    estj = item[0]
    x = item[1]
    y = item[2]
    print(f"{item[0]}\t | {item[1]}\t | {item[2]}\n")

    if y in raw_data:
        raw_data.remove(y)
    else:
        print("not found")

print(f"size of data after filtering with MinHash: ", len(raw_data))
save_dir = os.path.split(args.eval_file)[:-1]
final_file_name = os.path.join(save_dir, f"minhash_{args.minhash_threshold}.json")
json.dump(raw_data, open(final_file_name, "w"), indent=2)


