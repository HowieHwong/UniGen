import time
import torch
from fastchat.model import load_model, get_conversation_template
from .utils.generation_utils import *
from .utils.file_process import *
from dotenv import load_dotenv
import os
import json
import threading
from tqdm import tqdm
import urllib3
import traceback

load_dotenv()
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


class LLMGeneration:
    def __init__(self, config):
        self.config = config
        self.model_path = config.get("model_path", None)
        self.test_type = config.get("test_type", None)
        self.data_path = config.get("data_path", None)
        
        self.online_model = config.get("online_model", None)
        self.repetition_penalty = config.get("repetition_penalty", None)
        self.num_gpus = config.get("num_gpus", None)
        self.max_new_tokens = config.get("max_new_tokens", None)
        self.debug = config.get("debug", None)
        self.online_model_list = get_models()[1]
        self.model_mapping = get_models()[0]
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.use_replicate = config.get("use_replicate", None)
        self.use_deepinfra = config.get("use_deepinfra", None)
        self.model_name = self.model_mapping.get(self.model_path, "") if self.model_mapping else ""
        self.max_retries = config.get("max_retries", None)
        self.retry_interval = config.get("retry_interval", None)
        self.key_name=config.get("key_name", 'prompt')

    def _generation_hf(self, prompt, tokenizer, model, temperature):
        """
            Generates a response using a Hugging Face model.

            :param prompt: The input text prompt for the model.
            :param tokenizer: The tokenizer associated with the model.
            :param model: The Hugging Face model used for text generation.
            :param temperature: The temperature setting for text generation.
            :return: The generated text as a string.
            """

        prompt = prompt2conversation(prompt, self.model_path)
        inputs = tokenizer([prompt])
        inputs = {k: torch.tensor(v).to(self.device) for k, v in inputs.items()}
        output_ids = model.generate(
            **inputs,
            do_sample=True if temperature > 1e-5 else False,
            temperature=temperature,
            repetition_penalty=self.repetition_penalty,
            max_new_tokens=self.max_new_tokens,
        )
        output_ids = output_ids[0][len(inputs["input_ids"][0]):]
        outputs = tokenizer.decode(
            output_ids, skip_special_tokens=True, spaces_between_special_tokens=False
        )
        return outputs

    def generation(self, model_name, prompt, tokenizer, model, temperature=None):
        """
            Generates a response using either an online or a local model.

            :param model_name: The name of the model.
            :param prompt: The input text prompt for the model.
            :param tokenizer: The tokenizer for the model.
            :param model: The model used for text generation.
            :param temperature: The temperature setting for text generation. Default is None.
            :return: The generated text as a string.
            """

        try:
            if (model_name in self.online_model_list) or (
                (self.online_model and self.use_replicate) or (self.online_model and self.use_deepinfra)):
                ans = gen_online(model_name, prompt, temperature, replicate=self.use_replicate,deepinfra=self.use_deepinfra)
            else:
                ans = self._generation_hf(prompt, tokenizer, model, temperature)
            if not ans:
                raise ValueError("The response is NULL or an empty string!")
            return ans
        except Exception as e:
            tb = traceback.format_exc()
            print(tb)

    def process_element(self, el, model, model_name, tokenizer, index, temperature, key_name='prompt'):
        """
            Processes a single element (data point) using the specified model.

            :param el: A dictionary containing the data to be processed.
            :param model: The model to use for processing.
            :param model_name: The name of the model.
            :param tokenizer: The tokenizer for the model.
            :param index: The index of the element in the dataset.
            :param temperature: The temperature setting for generation.
            :param key_name: The key in the dictionary where the prompt is located.
            """

        try:
            # If 'res' key doesn't exist or its value is empty, generate a new response
            if "res" not in el or not el['res']:
                res = self.generation(model_name=model_name, prompt=el[key_name], tokenizer=tokenizer, model=model,
                                      temperature=temperature)
                el['res'] = res
        except Exception as e:
            # Print error message if there's an issue during processing
            print(f"Error processing element at index {index}: {e}")

    def process_file(self, data_path, save_path, model_name, tokenizer, model, file_config, key_name='prompt'):
        """
            Processes a file containing multiple data points for text generation.

            :param data_path: Path to the input data file.
            :param save_path: Path where the processed data will be saved.
            :param model_name: The name of the model used for processing.
            :param tokenizer: The tokenizer for the model.
            :param model: The model to use for processing.
            :param file_config: Configuration settings for file processing.
            :param key_name: The key in the dictionary where the prompt is located.
            """
        # if os.path.basename(data_path) not in file_config:
        #     print(f"{os.path.basename(data_path)} not in file_config")
        #     return

        with open(data_path) as f:
            original_data = json.load(f)

        if os.path.exists(save_path):
            with open(save_path, 'r') as f:
                saved_data = json.load(f)
        else:
            saved_data = original_data

        GROUP_SIZE = 8 if self.online_model else 1
        for i in tqdm(range(0, len(saved_data), GROUP_SIZE), desc=f"Processing {data_path}", leave=False):
            group_data = saved_data[i:i + GROUP_SIZE]
            threads = []
            for idx, el in enumerate(group_data):
                #temperature = file_config.get(os.path.basename(data_path), 0.0)
                temperature=0.0
                t = threading.Thread(target=self.process_element,
                                     args=(el, model, model_name, tokenizer, idx, temperature, key_name))
                t.start()
                threads.append(t)
            save_json(saved_data, f"{save_path}")

            # Wait for all threads to complete
            for t in threads:
                t.join()
        save_json(saved_data, f"{save_path}")

    def _run_task(self, model_name, model, tokenizer, base_dir, file_config, key_name='prompt'):
        """
            Runs a specific evaluation task based on provided parameters.

            :param model_name: The name of the model.
            :param model: The model used for processing.
            :param tokenizer: The tokenizer for the model.
            :param base_dir: Base directory containing test data files.
            :param file_config: Configuration settings for file processing.
            :param key_name: The key in the dictionary where the prompt is located.
            """

        test_res_dir = os.path.join(base_dir, 'test_res', model_name)
        if not os.path.exists(test_res_dir):
            os.makedirs(test_res_dir)
        section = base_dir.split('/')[-1]

        #os.makedirs(os.path.join('generation_results', model_name, section), exist_ok=True)

        file_list = os.listdir(base_dir)
        file_list = [file for file in file_list if file.endswith('.json')]
        for file in tqdm(file_list, desc="Processing files"):
            data_path = os.path.join(base_dir, 'test_data',file)
            save_path = os.path.join(base_dir, 'test_res',model_name,file) 
            self.process_file(data_path, save_path, model_name, tokenizer, model, file_config, key_name)


    def run_data(self, model_name, model, tokenizer):
        #file_config =  self.config.get('task_files',[])
        file_config=None
        self._run_task(model_name, model, tokenizer, self.data_path, file_config,self.key_name)

    def _run_single_test(self):
        """
            Executes a single test based on specified parameters.

            :param args: Contains parameters like test type, model name, and other configurations.
            :return: "OK" if successful, None otherwise.
            """
        model_name = self.model_name
        # print(f"Beginning generation with {self.test_type} evaluation at temperature {self.temperature}.")
        print(f"Evaluation target model: {model_name}")
        print(model_name in self.online_model_list)
        print(self.online_model_list)
        if (model_name in self.online_model_list) or (
                (self.online_model and self.use_replicate) or (self.online_model and self.use_deepinfra)):
            model, tokenizer = (None, None)
        else:
            model, tokenizer = load_model(
                self.model_path,
                num_gpus=self.num_gpus,
                device=self.device,
                debug=self.debug,
            )

        self.run_data(model_name=model_name, model=model, tokenizer=tokenizer)

    def generation_results(self, max_retries=10, retry_interval=3):
        """
            Main function to orchestrate the test runs with retries.

            :param args: Command-line arguments for the test run.
            :param max_retries: Maximum attempts to run the test.
            :param retry_interval: Time interval between retries in seconds.
            :return: Final state of the test run.
            """
        if not os.path.exists(self.data_path):
            print(f"Dataset path {self.data_path} does not exist.")
            return None

        for attempt in range(max_retries):
            try:
                state = self._run_single_test()
                if state:
                    print(f"Test function successful on attempt {attempt + 1}")
                    return state
            except Exception as e:
                print(f"Test function failed on attempt {attempt + 1}: {e}")
                print(f"Retrying in {retry_interval} seconds...")
                time.sleep(retry_interval)

        print("Test failed after maximum retries.")
        return None
