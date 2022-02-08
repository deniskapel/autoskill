import argparse
import json

from utils.annotation import Midas

def main(dataset_path: str, midas_url: str, output_path: str):
    
    with open(dataset_path, 'r', encoding="utf8") as f:
        dataset = json.load(f)
        
    midas_model = Midas(url=midas_url)
    
    midas_model.annotate(dataset)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)
    
    

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg("--dataset", "-d", help="the path to a dataset to annotate.", required=True)
    arg("--midas", "-m", help="url to midas batch_model", required=True)
    arg("--output_path", "-o", help="the annotated dataset will be saved to this path", required=True)
    
    args = parser.parse_args()

    main(args.dataset, args.midas, args.output_path)