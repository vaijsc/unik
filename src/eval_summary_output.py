import argparse
from src.utils import read_jsonl, setup_logger
from src.data_utils import read_data, ASPECTS_SPACE, ASPECTS_AMASUM
from src.eval import calc_rouge

logger = setup_logger()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_path', type=str)
    parser.add_argument('-g', '--gold_data_path', type=str)
    parser.add_argument('-d', '--dataset', type=str, default='amasum')
    args = parser.parse_args()

    data = read_data(args.gold_data_path, args.dataset)
    aspects = ASPECTS_SPACE if args.dataset == 'space' else ASPECTS_AMASUM
    summary_output = read_jsonl(args.input_path)

    for aspect in aspects:
        gold_summaries = {d["entity_id"]: d['summaries'][aspect] for d in data}
        predictions = {d["entity_id"]: d['summaries'][0] for d in summary_output if d['aspect'] == aspect}
        entity_ids = list(gold_summaries.keys())
        
        predictions = [predictions[entity] for entity in entity_ids]
        references = [gold_summaries[entity] for entity in entity_ids]

        result = calc_rouge(predictions, references)

        output = "\n".join([f"{k} {v}" for k, v in result.items()])
        logger.info(f"{aspect}\n{output}")

    

if __name__ == "__main__":
    main()
