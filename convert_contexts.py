import argparse
import json

import numpy as np

from tqdm import tqdm

from utils.convert_utils import encode_context
# from utils.preprocessing import normalize_resp

parser = argparse.ArgumentParser()
parser.add_argument(
    "--context_file_path",
    help="Path to the json context file",
    required=True)
parser.add_argument(
    "--store_file_path",
    help="Store to a file of npy format",
    required=True)

args = parser.parse_args()

with open(args.context_file_path, 'r', encoding="utf8") as f:
    contexts = json.load(f)

contexts = [[" ".join(ut) for ut in utts] for utts in contexts]

with open(args.store_file_path, 'wb') as f:
    for context in tqdm(contexts):
        context_encoding = encode_context(context)
        np.save(f, context_encoding)

print('Context is encoded')
