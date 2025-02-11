import argparse
import json
from pprint import pprint
from utils.eval import calc_rouge
from utils.load_data import load_config, load_splits_data, load_summ_data

def load_jsonl(path):
    data = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line != "":
                data.append(json.loads(line)['response'])
    return data

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("file_path")
    parser.add_argument("-d", "--dataset_name", default="amazon", type=str)
    parser.add_argument("-o", "--output_file", required=False)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args() 
    config = load_config("config.yml") 

    # Load data
    if args.dataset_name == "amazon":
        dev_data, test_data = load_summ_data(config['data']['amazon_summ_data_path'],
                                             config['data']['amazon_summ_splits_path'])
    else:
        raise ValueError(f"dataset_name = {args.dataset_name}")
    labels = [entity['labels'] for entity in test_data] 

    predictions = load_jsonl(args.file_path)
    result = calc_rouge(predictions, labels)
    pprint(result)
    if args.output_file:
        import os
        from pathlib import Path
        output_path = Path(config.output_file)
        os.makedirs(output_path.parent, exist_ok=True) 
        with open(output_path, 'w') as f:
            json.dump(result, f)
            f.write('\n')
        print(f"Saved result to {output_path}")

