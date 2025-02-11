from typing import List, Text
from src.prompt_template import PROMPT_EXTRACTION

class GraphExtractor:
    def __init__(self, llm):
        self.llm = llm
        self.is_opensource = True
    
    def get_messages(self, reviews: List[Text]):
        content = "\n".join(reviews)
        content = PROMPT_EXTRACTION.format(reviews=content).strip()
        if self.is_opensource:
            return content
        else:
            messages = [
                {"role": "user", "content": content},
            ]
            return messages

    def generate(self, data, output_path):
        # check processed examples
        prepare_path(output_path)
        n_processed_examples = get_processed_examples(output_path) 
        
        # output file
        writer = open(output_path, 'a')
        prompts = self.prepare_prompt(test_data)
        L = len(prompts)
        for i in tqdm(range(n_processed_examples, len(prompts), batch_size), desc='review batches', ncols=0):
            input_texts = [p['prompt'] for p in prompts[i: min(L, i + batch_size)]]
            responses = self.model.generate(input_texts)            
            for idx, response in zip(prompts[i: min(L, i + batch_size)], responses):
                output = {
                    **idx,                
                    "response": response, 
                } 
                json.dump(output, writer)
                writer.write('\n')
        writer.close()

if __name__ == "__main__":
    config = load_config('config.yml')
    args = get_args() 
    print_args(args)

    _, test_data = load_summ_data(config['data']['space_summ_data_path'],
                                         config['data']['space_summ_splits_path'],
                                         aspect_name='general')
    if args.model == "gpt4":
        gen_by_gpt4(test_data, args=args, output_path=args.output_path)
    if args.model == "llama3":
        model_path = config['LLama3']['model_path']
        gen_by_llama3(test_data, model_path, args.batch_size, args.output_path)
