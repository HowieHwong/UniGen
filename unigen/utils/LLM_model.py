from tenacity import retry, wait_random_exponential, stop_after_attempt
from openai import OpenAI, AzureOpenAI
# from anthropic import Anthropic
import traceback

def singleton(cls):
    instances = {}

    def get_instance(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]

    return get_instance

@singleton
class ModelAPI:
    def __init__(self,config, model_type='gpt', temperature=0.8):
        self.config=config
        self.model_type = config['generation_settings']['model_type'].lower()
        self.temperature = config['generation_settings']['temperature']
        if self.model_type not in ['gpt', 'claude', 'llama3']:
            raise ValueError(f"Unsupported model type: {model_type}")

    @retry(wait=wait_random_exponential(min=1, max=10), stop=stop_after_attempt(10))
    def get_res(self, text, model=None, message=None, azure=True, json_format=False, ):
        temperature = self.temperature
        try:
            if self.model_type == 'gpt':
                return self.api_send_gpt4(text, model, message, azure, json_format, temperature)
            elif self.model_type == 'claude':
                self.api_key = self.config["api_settings"]["claude_api_key"]
                return self.api_send_claude(text, model, message, json_format, temperature)
            elif self.model_type == 'llama3':
                self.api_key = self.config["api_settings"]["llama3_api_key"]
                return self.api_send_llama3(text, model, message, json_format, temperature)
            else:
                raise ValueError(f"Unsupported model type: {self.model_type}")
        except Exception as e:
            print(traceback.format_exc())

    def api_send_llama3(self, string, model=None, message=None, json_format=False, temperature=0.8):
        client = OpenAI(api_key=self.api_key,
                        base_url="https://api.deepinfra.com/v1/openai",
                        )
        top_p = 1 if temperature <= 1e-5 else 0.9
        temperature = 0.01 if temperature <= 1e-5 else temperature
        model = 'meta-llama/Meta-Llama-3-70B-Instruct'
        print(f"Sending API request...{model},temperature:{temperature}")
        chat_completion = client.chat.completions.create(
            model='meta-llama/Meta-Llama-3-70B-Instruct',
            messages=[{"role": "user", "content": string}],
            temperature=temperature,
            top_p=top_p,
            max_tokens=4096,
        )
        if not chat_completion.choices[0].message.content:
            raise ValueError("The response from the API is NULL or an empty string!")
        return chat_completion.choices[0].message.content

    def api_send_claude(self, string, model="claude-3-opus-20240229", message=None, json_format=False, temperature=0.8):
        client = Anthropic(
            api_key=self.api_key,
        )
        model = "claude-3-opus-20240229",
        print(f"Sending API request...{model}")
        message = client.messages.create(
            temperature=temperature,
            max_tokens=4096,
            messages=[
                {
                    "role": "user",
                    "content": string,
                }
            ],
            model="claude-3-opus-20240229",
        )
        if not message.choices[0].message.content:
            raise ValueError("The response from the API is NULL or an empty string!")
        full_response = message.content[0].text
        return full_response

    def api_send_gpt4(self, string, model, message=None, azure=True, json_format=False, temperature=0.8):
        azure = self.config["api_settings"]["use_azure"]
        if message is None:
            message = [{"role": "user", "content": string}]
        response_format = {"type": "json_object"} if json_format else None
        if azure:
            azure_endpoint = self.config["api_settings"]["azure_base_url"]
            api_key = self.config['api_settings']['azure_api_key']
            api_version = self.config["api_settings"]["azure_version"]
            model = self.config["api_settings"]["azure_model"]
            print(f"Sending API request...{model},temperature:{temperature}")
            client = AzureOpenAI(
                azure_endpoint=azure_endpoint,
                api_key=api_key,
                api_version=api_version,
            )
            chat_completion = client.chat.completions.create(
                model=model,
                messages=message,
                temperature=temperature,
                response_format=response_format,
            )
        else:
            model = self.config["api_settings"]["openai_chat_model"]
            base_url = self.config["api_settings"].get("base_url")
            api_key = self.config['api_settings']['openai_api_key']

            # Correct the client initialization
            if base_url:
                client = OpenAI(api_key=api_key, base_url=base_url)
            else:
                client = OpenAI(api_key=api_key)
            model = "gpt-4-0125-preview"
            print(f"Sending API request...{model},temperature:{temperature}")
            chat_completion = client.chat.completions.create(
                model=model,
                messages=message,
                temperature=temperature,
                response_format=response_format,
            )
        dict(chat_completion.usage)  #{'completion_tokens': 28, 'prompt_tokens': 12, 'total_tokens': 40}
        if not chat_completion.choices[0].message.content:
            raise ValueError("The response from the API is NULL or an empty string!")
        full_response = chat_completion.choices[0].message.content
        return full_response
