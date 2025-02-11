import argparse
import json
import os
from src import prompt_template
from src.config import cfg
from tqdm import tqdm
from src.utils import setup_logger, write_jsonl, prepare_file
from src.data_utils import read_reviews

logger = None


def get_messages(reviews: list[str], extraction_prompt: str) -> list[list[dict[str, str]]]:
    prompts = [extraction_prompt.format(reviews=r) for r in reviews]
    messages_list = [[{"role": "user", "content": prompt}] for prompt in prompts]
    return messages_list


def check_processed_data(path):
    cnt = 0
    if os.path.isfile(path):
        with open(path) as fin:
            for _ in fin:
                cnt += 1
    logger.info(f"Processed {cnt} reviews")
    return cnt


def write_output(reviews, responses, output_path):
    output = []
    for review, response in zip(reviews, responses):
        obj = {"entity_id": review["entity_id"], "review_id": review["review_id"], "response": response}
        output.append(obj)
    write_jsonl(output, output_path, mode="a", verbose=False)
    logger.info(f"Write {len(reviews)} lines to `{output_path}`")


def main():
    parser = argparse.ArgumentParser(description="Updating Config settings.")
    parser.add_argument(
        "--log_level", type=str, default="INFO", help="Set the logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)"
    )
    parser.add_argument("--log_file", type=str, default=None)
    parser.add_argument("--model_name", type=str, default="mistral")
    parser.add_argument("--exp_name", type=str, help="Name of the experiment", default="test")
    parser.add_argument("--output_path", type=str, default=None)
    args = parser.parse_args()

    cfg.update(args)

    # setup logger
    global logger
    logger = setup_logger(file=cfg.CONF["log_file"])
    logger.info(cfg)

    dataset_name = cfg.CONF["dataset"]
    data_path = cfg.DATA_CONF[dataset_name]["test_path"]

    reviews = read_reviews(path=data_path, dataset_name=dataset_name)

    # Check processed examples
    output_path = cfg.CONF["output_path"]
    n_processed = check_processed_data(output_path)
    reviews = reviews[n_processed:]

    extraction_prompt = prompt_template.PROMPT_EXTRACTION[dataset_name].strip()
    messages_list = get_messages(reviews, extraction_prompt)

    p_bar = tqdm(total=len(reviews), desc="Reviews", ncols=0)
    if "gpt" in cfg.CONF["model_name"]:
        logger.info("Generate by GPT")
        from src.call_llm.gpt import GPT

        model_config = cfg.OPENAI_CONF[cfg.CONF["model_name"]]
        model = GPT(**model_config)
        sampling_params = {
            "max_tokens": cfg.CONF['extraction']["max_tokens"],
            "temperature": cfg.CONF['extraction']["temperature"],
            "top_p": cfg.CONF['extraction']["top_p"],
            # "top_k": cfg.conf["top_k"],
        }
        prepare_file(output_path)
        fout = open(output_path)
        for review, messages in zip(reviews, messages_list):
            response = model.generate(messages, **sampling_params)
            obj = {
                "entity_id": review["entity_id"],
                "review_id": review["review_id"],
                "response": response,
            }
            json.dump(obj, fout)
            fout.write("\n")
            p_bar.update(1)
        logger.info(f"Saved reponses to `{output_path}`")
    else:
        # use mistral
        logger.info("Generate by vLLM")
        from vllm import LLM, SamplingParams

        sampling_params = {
            "max_tokens": cfg.CONF['extraction']["max_tokens"],
            "temperature": cfg.CONF['extraction']["temperature"],
            "top_p": cfg.CONF['extraction']["top_p"],
            "top_k": cfg.CONF['extraction']["top_k"],
        }
        sampling_params = SamplingParams(**sampling_params)
        model_config = cfg.HF_CONF["mistral"]
        model = LLM(
            model_config["model_path"], swap_space=4, dtype=model_config["dtype"], seed=42, gpu_memory_utilization=0.9
        )
        tokenizer = model.get_tokenizer()
        prompts = [
            tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            for messages in messages_list
        ]

        # Generate
        L = len(prompts)
        batch_size = cfg.CONF["vllm_batch_size"]
        for start_idx in range(0, L, batch_size):
            end_idx = min(L, start_idx + batch_size)
            responses = model.generate(prompts[start_idx:end_idx], sampling_params=sampling_params)
            responses = [r.outputs[0].text.strip() for r in responses]
            write_output(reviews[start_idx:end_idx], responses, output_path)
            p_bar.update(end_idx-start_idx)
    p_bar.close()

if __name__ == "__main__":
    main()
