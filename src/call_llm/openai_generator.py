from typing import List, Dict, Text
from openai import OpenAI, AzureOpenAI
import time

class OpenAIGenerator:
    def __init__(self, model_name, api_key, api_version=None, endpoint = None):
        self.model_name = model_name
        if api_version:
            self.client = AzureOpenAI(
                api_key = api_key,
                azure_endpoint = endpoint,
                api_version = api_version
            )
        else:
            self.client = OpenAI(
                api_key=api_key,
            )

    def generate(self, messages: List[Dict[Text, Text]], max_tokens=128, time_sleep=1, **kwargs) -> str:
        """
        :param messages: List of messages to send
        :param max_tokens: The maximum number of tokens to generate.
        :return: The generated text.
        """
        response = ''
        while not response:
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    max_tokens=max_tokens,
                    **kwargs
                )
            except Exception as e:
                print(f"Error: {e}")
                print("Retrying...")
                time.sleep(time_sleep)
        return response.choices[0].message.content.strip()