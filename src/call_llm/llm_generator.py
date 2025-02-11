from typing import List, Dict, Text
import gc
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

class LlmGenerator(object):
    def __init__(self, model_path, dtype=None):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side='left')
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype= dtype if dtype else "auto",
            device_map="cuda",
            use_cache=True
            # attn_implementation="flash_attention_2",
        )
        self.terminators = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]

    def generate(self, input_texts: List[Text], sampling_params): 
        encoded_input = self.tokenizer(input_texts, padding=True, return_tensors='pt')
        input_ids = encoded_input['input_ids'].to(self.model.device)
        attention_mask = encoded_input['attention_mask'].to(self.model.device)
        self.model.eval()    
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pad_token_id=self.tokenizer.pad_token_id, 
                eos_token_id=self.terminators,
                do_sample=True,
                **sampling_params,
            )
        responses = self.tokenizer.batch_decode(outputs[:, input_ids.shape[-1]:], skip_special_tokens=True)
        responses = [response.strip() for response in responses]
        del input_ids
        del attention_mask
        del outputs
        gc.collect()
        torch.cuda.empty_cache()
        return responses

if __name__ == "__main__":
    model_path = '/home/ubuntu/hf/Mistral-7B-Instruct-v0.2'  # Example model, replace with your model path
    generator = LlmGenerator(model_path=model_path)

    input_texts = ["Once upon a time"]
    sampling_params = {
        'max_length': 128,
        'temperature': 0.7,
        'top_k': 50
    }
    
    responses = generator.generate(input_texts, sampling_params)
    for i, response in enumerate(responses):
        print(f"Input: {input_texts[i]}")
        print(f"Response: {response}\n")
