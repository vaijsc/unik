from typing import List, Union, Text
import os
import json
from pathlib import Path
from tqdm import tqdm

from src.prompt_template import PROMPT_COT, PROMPT_ZERO_SHOT, PROMPT_FEW_SHOT, N_SHOT_EXAMPLES
from src.utils.load_data import load_summ_data
from src.utils.eval import calc_rouge

class FewShot():
    def __init__(self, llm, exp_type, is_opensource=False) -> None:
        self.llm = llm
        self.exp_type = exp_type
        self.is_opensource = is_opensource

    def get_messages(self, prompt):
        messages = []
        messages.append(
                {"role": "user", "content": prompt}
            )
        return messages

    def build_prompt(self, text, aspect, data_name): 
        """Build prompt for a sample"""
        
        if self.exp_type == 'zeroshot': 
            prompt = PROMPT_ZERO_SHOT.format(aspect=aspect, text=text).strip()
        elif self.exp_type == 'fewshot':
            example_summary = '\n\n'.join(N_SHOT_EXAMPLES[data_name][aspect])
            prompt = PROMPT_FEW_SHOT.format(aspect=aspect, example_summary=example_summary, text=text).strip()
        elif self.exp_type == 'cot':
            prompt = PROMPT_COT.format(aspect=aspect, text=text).strip()
        else:
            raise ValueError()

        return prompt
    
    def get_reviews(self, entity):
        reviews = entity["reviews"]
        reviews_combined = [" ".join(review) for review in reviews]
        return reviews_combined

    def split_reviews(self, reviews, K):
        return [reviews[i:i + K] for i in range(0, len(reviews), K)]

    def summarize_reviews(self, reviews_group, text_summarizer, dataset_name: str, aspect_name: str):
        combined_reviews_text = " ".join(reviews_group)
        prompt = self.build_prompt(combined_reviews_text, aspect_name, dataset_name)
        if not self.is_opensource:
            messages = self.get_messages(prompt)
        else:
            messages = prompt
        # summary_text = text_summarizer.generate(messages, aspect_name=aspect_name)
        summary_text = text_summarizer.generate(messages)
        summary_lines = summary_text.split("\n\n")
        final_summary = summary_lines[0]
        for line in summary_lines:
            if "Output:" in line:
                final_summary = line.split(":")[1]
        return final_summary
    
    def iterative_summarization(self, text_summarizer, reviews, group_size, dataset_name, aspect_name):
        combined_summary = ""
        while len(reviews) > 1:
            # Split reviews into groups based on group_size
            reviews_groups = self.split_reviews(reviews, group_size)
            
            # Summarize each group of reviews
            summaries = [self.summarize_reviews(group, text_summarizer, dataset_name, aspect_name) for group in reviews_groups]
            
            # Update reviews with the new summaries
            reviews = summaries
            
            # Combine summaries if only two reviews are left
            if len(reviews) == 2:
                combined_summary = " ".join(reviews)
        
        # Return the combined summary and the final summary
        return combined_summary, reviews[0]
    
    def setup_summaries_storage(self, summaries):
        summaries_storage = Path(summaries)
        os.makedirs(summaries_storage.parent, exist_ok=True)
        return open(summaries_storage, "a")

    def process_entity(self, text_summarizer, entity, aspect_name, dataset_name, K):
        references = entity["labels"]
        reviews = self.get_reviews(entity)[:40]
        # reviews = self.get_reviews(entity)[:200]
        _, response = self.iterative_summarization(text_summarizer, reviews, K, dataset_name, aspect_name)
        return references, response

    def save_output(self, writer, idx, aspect_name, response):
        output = {
            "idx": idx,
            "aspect": aspect_name,
            "summary": response,
        }
        json.dump(output, writer)
        writer.write("\n")

    def evaluate_responses(self, responses, references, aspect_name, eval_output):
        result = calc_rouge(responses, references)
        print(f"Result-{aspect_name}: ")
        print(result)
        with open(eval_output, "a") as f:
            f.write(f"{aspect_name}: ")
            json.dump(result, f)
            f.write("\n")
            
    def process(self, data, eval_output_path, summaries_file_path, group_size=99999):
        # data config
        dataset_name = data.name
        aspect_names = data.aspects
        
        # Get the text summarizer object
        text_summarizer = self.llm
        
        # Create and open the file for storing summaries
        summaries_storage = self.setup_summaries_storage(summaries_file_path)

        for aspect_name in aspect_names:
            # Load reviews and golden summaries labels
            _, test_data = load_summ_data(
                data.raw.summ_data_path,
                data.raw.summ_splits_path,
                aspect_name=aspect_name,
            )
            print("....data loaded")
            
            # Lists to store reference summaries and generated summaries
            reference_summaries, generated_summaries = [], []

            for idx, entity in tqdm(enumerate(test_data[:50]), desc=f"{aspect_name.upper()}-Eval Examples", ncols=100):
                # Process each data item to get reference and generated summary
                reference, generated_summary = self.process_entity(text_summarizer, entity, aspect_name, dataset_name, group_size)
                reference_summaries.append(reference)
                generated_summaries.append(generated_summary)
                
                # Save the generated summary to the file
                self.save_output(summaries_storage, idx+10, aspect_name, generated_summary)

            # Evaluate the generated summaries against the reference summaries
            self.evaluate_responses(generated_summaries, reference_summaries, aspect_name, eval_output_path)

        # Close the file after processing is complete
        summaries_storage.close()



if __name__ == "__main__":
    import hydra
    from omegaconf import DictConfig, OmegaConf
    from hydra.core.global_hydra import GlobalHydra
    from hydra import initialize, compose
    @hydra.main(config_path="../../conf", config_name="config")
    def main(cfg: DictConfig):
        # Initialize the Hydra context to manually load additional config files
        GlobalHydra.instance().clear()
        initialize(config_path="../../conf/secret")

        # Load the secret config file
        secret_cfg = compose(config_name="openai_keys")

        # Merge secret config with the main config
        cfg = OmegaConf.merge(OmegaConf.create({"secret": secret_cfg}), OmegaConf.create(cfg))
        from src.call_llm.openai_generator import OpenAIGenerator
        model_name = cfg.llm_config.model_name
        omini = OpenAIGenerator(cfg.llm_init.model[model_name], cfg.llm_init.api_key[model_name], api_version=cfg.llm_init.api_version, endpoint=cfg.llm_init.endpoint[model_name])
        fs = FewShot(omini, cfg.experiments.exp_type)
        fs.process(data=cfg.data, eval_output_path='output.json', summaries_file_path='summaries.json', group_size=cfg.llm_config.group_size)
        print("zeroshot done")
    
    main()