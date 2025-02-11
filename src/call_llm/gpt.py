import logging
import time
import yaml
from openai import AzureOpenAI, OpenAI

logger = logging.getLogger(__name__)


class GPT:
    def __init__(
        self,
        model_name="",
        api_type="",
        wait_time: float = 0.0,
        max_retry: int = 5,
        **kwargs,
    ) -> None:
        if api_type == "openai":
            self.client = OpenAI(api_key=kwargs["api_key"])
        elif api_type == "azure":
            self.client = AzureOpenAI(
                api_version=kwargs["api_version"],
                api_key=kwargs["api_key"],
                azure_endpoint=kwargs["endpoint"],
            )
        elif api_type == "local":
            self.client = OpenAI(
                base_url=kwargs["endpoint"], api_key=kwargs['api_key']
            )
        self.model_name = model_name
        self.wait_time = wait_time
        self.max_retry = max_retry

    def generate(
        self,
        messages: list[dict[str, str]],
        sampling_params: dict,
    ):
        response = ""
        cnt = 0
        while response == "":
            try:
                res = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    **sampling_params,
                )
                if res.choices:
                    response = res.choices[0].message.content.strip()
                    # logger.info(f"Retrieved response from model:{response}")
                    return response
            except Exception as err:
                logger.error(f"Unexpected error: {err}")
                cnt += 1
                if cnt == self.max_retry:
                    logger.info("Maximum retry exceeded")
                    break
                if self.wait_time > 0:
                    time.sleep(self.wait_time)

        if response == "":
            logger.info("Failed to generate text after retries.")
        return response


if __name__ == "__main__":
    config = yaml.safe_load(open("secret/openai_config.yml"))
    model_config = config['gpt35-turbo']
    model = GPT(**model_config)
    sampling_params = {"max_tokens": 32, "temperature": 1.0, "top_p": 1.0,}
    messages = [{"role": "user", "content": "Hello, who are you?"}]
    res = model.generate(messages, **sampling_params)
    print(res)
