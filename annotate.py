import argparse
import json

from utils.annotation import Midas, EntityDetection

def main(dataset_path: str, midas_url: str, ner_url: str, output_path: str):

    with open(dataset_path, 'r', encoding="utf8") as f:
        dataset = json.load(f)

    midas_model = Midas(url=midas_url, first_phrase="")
    ner_model = EntityDetection(url=ner_url)

    midas_model.annotate(dataset)
    ner_model.annotate(dataset)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)

    print(f'Success! The annotated dataset is saved to {output_path}')


if __name__ == '__main__':
    """add midas labels to each sentence and detect entities

    midas_classification and entity_detection are Dream annotators
    https://github.com/deepmipt/dream/tree/main/annotators
    """
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg("--dataset", "-d", help="the path to a dataset to annotate.", required=True)
    arg("--midas", "-m", help="url to midas batch_model", required=True)
    arg("--ner", "-n", help="url to entity detection model", required=True)
    arg("--output_path", "-o", help="the annotated dataset will be saved to this path", required=True)

    args = parser.parse_args()

    main(args.dataset, args.midas, args.ner, args.output_path)
