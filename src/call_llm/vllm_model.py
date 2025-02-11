from vllm import LLM, SamplingParams


class VLLMModel:
    def __init__(self, model_name, **kwargs):
        self.model_name = model_name
        self.model = LLM(model_name, **kwargs)

    def generate(self, messages: list[str], sampling_params: SamplingParams, max_length=32000) -> list[str]:
        tokenizer = self.model.get_tokenizer()
        # check if content is too long
        inputs = tokenizer.encode(messages[-1]['content'], truncation=True, max_length=max_length, add_special_tokens=False)
        messages[-1]['content'] = tokenizer.decode(inputs)
        inputs = tokenizer.apply_chat_template( messages, add_generation_prompt=True, tokenize=True)
        generation_outputs = self.model.generate(
            prompt_token_ids=inputs, sampling_params=sampling_params, use_tqdm=False
        )
        response = generation_outputs[0].outputs[0].text.strip()
        return response

    def batch_generate(
        self,
        messages_list: list[list[str]] | list[str],
        sampling_params: SamplingParams,
    ) -> list[str]:
        # if receive single messages, convert to list
        if isinstance(messages_list[0], str):
            messages_list = [messages_list]  # type: ignore
        tokenizer = self.model.get_tokenizer()
        prompts = [
            tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            for messages in messages_list
        ]
        generation_outputs = self.model.generate(prompts, sampling_params)
        responses = [o.outputs[0].text for o in generation_outputs]
        return responses
