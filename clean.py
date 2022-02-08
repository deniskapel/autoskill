import argparse
import json

from utils.preprocessing import Daily2Clean, Topical2Clean


def main(topical_path: str, daily_path):
    
    with open(topical_path, 'r', encoding="utf8") as file:
        topical = json.load(file)
        
    with open(daily_path, 'r', encoding="utf8") as file:
        daily = file.readlines()
        
    tc = Topical2Clean(topical, 'data/topical_clean.json')
    with open(tc.output_path, 'w', encoding='utf-8') as f:
        json.dump(tc.clean(), f, ensure_ascii=False, indent=2)
        
    # remove duplicates from the daily dialogue dataset
    daily = list(set(daily))
    dd = Daily2Clean(daily, 'data/daily_clean.json')
    with open(dd.output_path, 'w', encoding='utf-8') as f:
        json.dump(dd.clean(), f, ensure_ascii=False, indent=2)
        
    print('Success')
    print(f'The preprocessed version of the Topical Chat dataset is saved to {tc.output_path}')
    print(f'The preprocessed version of the Daily dialogue dataset is saved to {dd.output_path}')
    
    

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg("--topical", "-t", help="path to raw topical chat dataset", required=True)
    arg("--daily", "-d", help="path to raw daily dialogue dataset", required=True)

    args = parser.parse_args()

    main(args.topical, args.daily)