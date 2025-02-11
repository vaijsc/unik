import os
import json
import argparse
from tqdm import tqdm
from src.config import cfg
from src.prompt_template import PROMPT_COT, PROMPT_ZERO_SHOT, PROMPT_FEW_SHOT, N_SHOT_EXAMPLES
from src.utils import setup_logger, read_jsonl, write_jsonl
from src.eval import calc_rouge
from src.call_llm.vllm_model import VLLMModel
from src.call_llm.gpt import GPT
from src.data_utils import read_data, ASPECTS_AMASUM, ASPECTS_SPACE


class FewShot:
    def __init__(self, llm, mode) -> None:
        self.llm = llm
        self.mode = mode

    def build_messages(self, text, dataset_name, aspect):
        """Build messages for a sample"""

        if self.mode == "zeroshot":
            prompt = PROMPT_ZERO_SHOT.format(aspect=aspect, text=text).strip()
        elif self.mode == "fewshot":
            example_summary = "\n\n".join(N_SHOT_EXAMPLES[dataset_name][aspect])
            prompt = PROMPT_FEW_SHOT.format(aspect=aspect, example_summary=example_summary, text=text).strip()
        elif self.mode == "cot":
            prompt = PROMPT_COT.format(aspect=aspect, text=text).strip()
        else:
            raise ValueError()

        return [{"role": "user", "content": prompt}]

    def get_review_texts(self, entity):
        return [review["review_text"] for review in entity["reviews"]]

    def split_reviews(self, reviews, K):
        return [reviews[i : i + K] for i in range(0, len(reviews), K)]

    def summarize(self, review_text, dataset_name: str, aspect: str, sampling_params: dict):
        messages = self.build_messages(review_text, dataset_name, aspect)
        response = self.llm.generate(messages, sampling_params)
        return response

    def iterative_summarize(self, review_texts, group_size, dataset_name, aspect, sampling_params):
        # if there is only one review, summarize it
        if len(review_texts) == 1:
            return self.summarize(review_texts[0], dataset_name, aspect, sampling_params)

        # if there are two or more reviews, summarize them
        while len(review_texts) > 1:
            # Split reviews into groups based on group_size
            review_groups = self.split_reviews(review_texts, group_size)

            # Summarize each group of reviews
            summaries = [
                self.summarize(" ".join(review_group), dataset_name, aspect, sampling_params)
                for review_group in review_groups
            ]

            # Update reviews with the new summaries
            review_texts = summaries

        return review_texts[0]

    # Get procesed data from file
    def get_processed_data(self, output_path):
        data = []
        if os.path.isfile(output_path):
            data = read_jsonl(output_path)

        logger.info(f"Processed data: {len(data)} lines")
        return data

    def evaluate(self, predictions, references, aspect, eval_output_path=None):
        entity_ids = [d["entity_id"] for d in predictions]
        predictions = {d["entity_id"]: d["summaries"][0] for d in predictions}
        references = {d["entity_id"]: d["summaries"] for d in references}

        predictions = [predictions[entity_id] for entity_id in entity_ids]
        references = [references[entity_id] for entity_id in entity_ids]

        result = calc_rouge(predictions, references)
        output = "\n".join([f"{k} {v}" for k, v in result.items()])
        logger.info(f"Result-{aspect}:\n{output}")

        if eval_output_path:
            with open(eval_output_path, "a") as f:
                f.write(f"{aspect}\n{output}")

    def process(
        self,
        data,
        dataset_name,
        eval_output_path,
        output_path,
        sampling_params,
        group_size=1,
        save_every=1,
        iterative_summarize=False,
    ):
        # data config
        aspects = ASPECTS_AMASUM if dataset_name == "amasum" else ASPECTS_SPACE

        # Get processed data
        processed_data = self.get_processed_data(output_path)

        for aspect in aspects:
            logger.info(f"Processing aspect: {aspect}")

            # Initialize reference and generated summaries
            processed_predictions = [d for d in processed_data if d["aspect"] == aspect]
            processed_entity_ids = [d["entity_id"] for d in processed_predictions]
            gold_summaries = [
                {"entity_id": entity["entity_id"], "aspect": aspect, "summaries": entity["summaries"][aspect]}
                for entity in data
                if entity["entity_id"] in processed_entity_ids
            ]

            # Process each data item to get reference and generated summary
            predictions = []
            cnt = 0
            prediction_container = []
            for entity in tqdm(
                data,
                total=len(data),
                desc=f"Summarize aspect: {aspect.upper()}",
                ncols=0,
            ):
                if entity["entity_id"] in processed_entity_ids or aspect not in entity["summaries"]:
                    continue

                # append directly to gold summaries because we dont savee them to disk
                gold_summaries.append(
                    {"entity_id": entity["entity_id"], "aspect": aspect, "summaries": entity["summaries"][aspect]}
                )

                review_texts = self.get_review_texts(entity)
                if iterative_summarize:
                    # iterative summarization
                    prediction = self.iterative_summarize(
                        review_texts, group_size, dataset_name, aspect, sampling_params
                    )
                else:
                    # zeroshot, fewshot or chain of thought summarization
                    prediction = self.summarize(" ".join(review_texts), dataset_name, aspect, sampling_params)

                prediction_container.append(
                    {"entity_id": entity["entity_id"], "aspect": aspect, "summaries": [prediction]}
                )
                cnt += 1
                if cnt >= save_every:
                    # Save the generated summary to the file
                    write_jsonl(prediction_container, output_path, mode="a", verbose=False)
                    # Reset counter
                    predictions.extend(prediction_container)
                    prediction_container = []
                    cnt = 0

            if len(prediction_container) > 0:
                write_jsonl(prediction_container, output_path, mode="a", verbose=False)
                predictions.extend(prediction_container)
                prediction_container = []

            # Merge processed predictions and predictions
            predictions = processed_predictions + predictions
            # Evaluate the generated summaries against the reference summaries
            self.evaluate(predictions, gold_summaries, aspect, eval_output_path)


def main():
    parser = argparse.ArgumentParser(description="Updating Config settings.")
    parser.add_argument("--log_level", type=str, default="INFO")
    parser.add_argument("--log_file", type=str, default=None)
    parser.add_argument("--model_name", type=str, default="mistral")
    parser.add_argument("--mode", type=str, default="zeroshot")
    parser.add_argument("--output_path", type=str, default=None)
    parser.add_argument("--eval_output_path", type=str, default=None)
    parser.add_argument("--dataset", type=str, default="amasum")
    parser.add_argument("--save_every", type=int, default=1)
    parser.add_argument("--group_size", type=int, default=1)
    parser.add_argument("--iterative_summarize", action="store_true", default=False)

    args = parser.parse_args()

    cfg.update(args)

    # setup logger
    global logger
    logger = setup_logger(file=cfg.CONF["log_file"])
    logger.info(cfg)

    dataset_name = cfg.CONF["dataset"]
    data_path = cfg.DATA_CONF[dataset_name]["test_path"]

    model_name = cfg.CONF["model_name"]
    if "gpt" in model_name:
        logger.info("Generate by GPT")
        model_config = cfg.OPENAI_CONF[model_name]
        llm = GPT(**model_config)
        sampling_params = {
            "max_tokens": cfg.CONF["summarization"]["max_tokens"],
            "temperature": cfg.CONF["summarization"]["temperature"],
            "top_p": cfg.CONF["summarization"]["top_p"],
            # "top_k": cfg.conf["top_k"],
        }
    else:
        # use mistral
        from vllm import SamplingParams

        logger.info("Generate by vLLM")

        sampling_params = {
            "max_tokens": cfg.CONF["summarization"]["max_tokens"],
            "temperature": cfg.CONF["summarization"]["temperature"],
            "top_p": cfg.CONF["summarization"]["top_p"],
            "top_k": cfg.CONF["summarization"]["top_k"],
        }
        sampling_params = SamplingParams(**sampling_params)
        model_config = cfg.HF_CONF[model_name]
        llm = VLLMModel(
            model_name=model_config["model_path"],
            swap_space=4,
            dtype=model_config["dtype"],
            seed=42,
            gpu_memory_utilization=0.9,
        )

    data = read_data(path=data_path, dataset_name=dataset_name)

    fs = FewShot(llm, mode=cfg.CONF["mode"])
    fs.process(
        data=data,
        dataset_name=cfg.CONF["dataset"],
        output_path=cfg.CONF["output_path"],
        eval_output_path=cfg.CONF["eval_output_path"],
        group_size=cfg.CONF["group_size"],
        sampling_params=sampling_params,
        save_every=cfg.CONF["save_every"],
        iterative_summarize=cfg.CONF["iterative_summarize"],
    )


if __name__ == "__main__":
    main()
