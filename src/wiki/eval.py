import argparse
from src.utils import read_jsonl, setup_logger
from src.data_utils import read_data, ASPECTS_SPACE, ASPECTS_AMASUM
from src.eval import calc_rouge

from transformers import AutoTokenizer

logger = setup_logger(mode='basic')

def cut(text, aspect, tokenizer):
    cutoff = {
        "general": 80,
        "rooms": 35,
        "location": 60,
        "service": 20,
        "cleanliness": 35,
        "building": 15,
        "food": 20,
    }
    return tokenizer.decode(tokenizer.encode(text, add_special_tokens=False)[:cutoff[aspect]])
    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_path', type=str)
    parser.add_argument('-g', '--gold_data_path', type=str)
    parser.add_argument('-d', '--dataset', type=str, default='space')
    args = parser.parse_args()

    data = read_data(args.gold_data_path, args.dataset)
    aspects = ASPECTS_SPACE if args.dataset == 'space' else ASPECTS_AMASUM
    summary_output = read_jsonl(args.input_path)

    model_path = '/lustre/scratch/client/vinai/users/hoangnv49/hf/mistralai/Mistral-7B-Instruct-v0.2'
    tokenizer = AutoTokenizer.from_pretrained(model_path)


    for aspect in aspects:
        gold_summaries = {d["entity_id"]: d['summaries'][aspect] for d in data}
        predictions = {d["entity_id"]: d['summaries'][0] for d in summary_output if d['aspect'] == aspect}
        entity_ids = list(gold_summaries.keys())
        
        # breakpoint()
        new_predictions = []
        new_references = []
        for entity in entity_ids:
            if predictions[entity].strip() != "":
                new_predictions.append(cut(predictions[entity], aspect, tokenizer))
                new_references.append(gold_summaries[entity])
        predictions = new_predictions
        references = new_references

        result = calc_rouge(predictions, references)

        output = "\n".join([f"{k} {v}" for k, v in result.items()])
        logger.info(f"{aspect}\n{output}")

    

if __name__ == "__main__":
    main()
