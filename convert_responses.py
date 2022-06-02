import argparse
import json

from numpy import vstack
from numpy import save as np_save
from tqdm import tqdm

from utils.convert_utils import encode_responses


def main(input_path, output_path):

    with open(input_path, 'r', encoding="utf8") as f:
        responses = json.load(f)

    encoded_responses = list()

    num_resp = len(responses)

    for pos in tqdm(range(0, num_resp, 1000)):
        offsets = [pos, pos + 1000]
        if offsets[1] > num_resp:
            offsets[1] = num_resp

        encoded_responses.append(
            encode_responses(responses[offsets[0]:offsets[1]])
        )

    encoded_responses = vstack(encoded_responses)

    assert encoded_responses.shape[0] == num_resp

    with open(output_path, 'wb') as f:
        np_save(f, encoded_responses)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--responses_file_path",
        help="Path to the json responses file",
        required=True)
    parser.add_argument(
        "--store_file_path",
        help="Store to a file of npy format",
        required=True)

    args = parser.parse_args()

    main(args.responses_file_path, args.store_file_path)
