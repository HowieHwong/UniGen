import os.path
import json
import random
import tiktoken
import copy
from tqdm import tqdm
from pprint import pprint
from .utils import attribute,embedding,data_format, RAG_eval, math_eval, self_reflection
from unigen.utils.IO import print, input
from unigen.utils.LLM_model import ModelAPI
from unigen.utils.embedding import EmbeddingProcessor
from unigen.utils.file_process import save_json,load_json,check_and_rename_file
from joblib import Parallel, delayed
from threading import Thread
from queue import Queue
import warnings
import traceback
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Thread
from datetime import datetime
from tenacity import retry, wait_random_exponential, stop_after_attempt
from .utils.prompt import prompt_template

warnings.filterwarnings("ignore")


class UniGen:
    def __init__(self,
                 config,
                 **kwargs):

        self.config=config
        self.efficiency_configuration=config['efficiency_configuration']

        generation_config=config['generation_settings']
        self.batch_size = generation_config['batch_size']
        self.random_example = generation_config['random_example']
        self.few_shot_num = generation_config['few_shot_num']
        self.generation_number =generation_config['generation_number']
        self.max_worker = generation_config['max_worker']
        self.temperature = generation_config['temperature']
        
        self.dataset_config = config['generation_hint']
        self.dataset_name = self.dataset_config["dataset_name"]
        self.dataset_description = self.dataset_config['dataset_description']
        self.constraints=self.dataset_config["dataset_constraint"]
        
        self.with_label = self.dataset_config['with_label']
        self.with_attribute = self.dataset_config['with_attribute']
        self.add_attribute = self.dataset_config['add_attribute']
        self.attribute_key = self.dataset_config['attribute_key'] if self.with_attribute or self.add_attribute else None
        self.extra_info_keys = self.dataset_config.get('extra_info_keys', [])
        
        self.Embedder=EmbeddingProcessor(self.config)
        self.LLM_model=ModelAPI(self.config)

    # def check_original_dataset(self, file_path):
    #     data=load_json(file_path)
    #     assert isinstance(data, list)
    #     if self.with_label:
    #         assert 'text' in list(data[0].keys()) and 'label' in list(data[0].keys())
    #     else:
    #         assert 'text' in list(data[0].keys())
    
    
    def _get_dataset_keys(self):
        keys = ['text', 'label'] + self.extra_info_keys
        if self.attribute_key:
            keys.append(self.attribute_key)
        return keys
    
    
    def example_selection(self,random_=False):

        data = self.Embedder.preprocess_original_dataset()  
        keys=self._get_dataset_keys()
        filtered_data = [{k: item[k] for k in keys if k in item} for item in data]
        
        if random_:
            random.shuffle(data)
            examples = data[:self.few_shot_num]
            filtered_examples = [{k: item[k] for k in keys if k in item} for item in examples]
        else:
            examples = self.Embedder.cluster_embeddings(data, num_clusters=self.few_shot_num)
            filtered_examples = [{k: item[k] for k in keys if k in item} for item in examples]
            
        random.shuffle(filtered_examples)

        return filtered_examples

    def few_shot_description(self, examples):
        random.shuffle(examples)
        json_output = json.dumps(examples, indent=4)
        return json_output

    def add_constraints(self, constraints):
        constraints_text = prompt_template["constraints_prefix"]
        for i, constraint in enumerate(constraints, 1):
            constraints_text += f"{i}. {constraint}\n"
        constraints_text += prompt_template["constraints_suffix"]
        return constraints_text

    def learn_from_human_feedback(self, examples):
        for example in examples:
            self._collect_user_feedback(example)
        feedback_string = ""
        for index, item in enumerate(examples):
            if 'label' in item:
                feedback_string += "Example: " + item['text'] + "\n" + "Label: " + item[
                    'label'] + '\n' + "Human Feedback: " + item['feedback'] + '\n\n'
            else:
                feedback_string += "Example: " + item['text'] + "\n" + "Human Feedback: " + item['feedback'] + '\n\n'
        return feedback_string

    def _collect_user_feedback(self, example):
        if 1:
            feedback = 'good'
            example['feedback'] = feedback
            return
        print("---------------------Please input your feedback-------------------------", "GREEN")
        if 'label' in example:
            print(f"Example: Text: {example['text']},    Label: {example['label']}")
            print("-------------------------------------------------------------------------", "GREEN")
            feedback = input("Please provide your feedback: ")
        else:
            print(f"Example: Text: {example['text']}")
            print("-------------------------------------------------------------------------", "green")
            feedback = input("Please provide your feedback: ", "red")
        example['feedback'] = feedback

    def count_tokens(self, text):
        enc = tiktoken.encoding_for_model(self.model)
        num_tokens = len(enc.encode(text))
        return num_tokens


    def run(self,):
        assert self.generation_number % self.batch_size == 0, "generation_number must be divisible by batch_size"
        self.Embedder = embedding.EmbeddingProcessor(config=self.config)
        self.Embedder.preprocess_original_dataset()  
                
        save_path=self.dataset_config['save_path']
        data_path=os.path.join(save_path, f"{self.dataset_name}_generated.json")
        generated_data_file_path = check_and_rename_file(data_path)
        print(generated_data_file_path)
        
        @retry(wait=wait_random_exponential(min=5, max=20), stop=stop_after_attempt(3))
        def batch_generation(batch_id, queue):
            try:
                batch_data = []
                if self.few_shot_num > 0:
                    examples = self.example_selection(self.random_example)
                    few_shot_des = self.few_shot_description(examples)
                else:
                     constraint_des = ""
                if self.constraints != []:
                    constraint_des = self.add_constraints(self.constraints)
                else:
                    constraint_des = ""
                description_prompt = prompt_template["description_prompt"].format(
                    description_for_dataset=self.dataset_description,
                )
                initial_prompt = prompt_template["initial_prompt"].format(batch_size=self.batch_size,
                                                                               dataset_constraint=constraint_des,
                                                                               few_shot_examples=few_shot_des)
                epoch_prompt = description_prompt + initial_prompt

                if self.add_attribute and not self.with_attribute:
                    examples = attribute.get_attribute(examples, dataset_description=self.dataset_description)
                    self.with_attribute = True
                if self.with_attribute:
                    epoch_prompt += attribute.add_attributes(examples=examples, attribute_key=self.attribute_key, attr=None)
                epoch_prompt += data_format.create_data_entries(num_elements=self.batch_size, with_label=self.with_label,attribute_key=self.attribute_key)
                
                res_data = data_format.get_res_data(epoch_prompt)
                epoch_data_item = data_format.extract_data_item(res_data)

                if self.efficiency_configuration["self_reflection"]:
                    reflection_res = self_reflection.reflection(epoch_data_item, self.dataset_description,
                                                                few_shot_des, constraint_des)
                    for index, reflect_res in enumerate(reflection_res):
                        if reflect_res['text']:
                            batch_data.append(reflect_res)
                        else:
                            batch_data.append(epoch_data_item[index])
                else:
                    batch_data += epoch_data_item
                if self.efficiency_configuration["math_eval"]:
                    for item in batch_data:
                        batch_data[batch_data.index(item)] = math_eval.math_eval(item)

                if self.efficiency_configuration["truthfulness_eval"]:
                    if batch_data:
                        for item in batch_data:                            
                            truthfulness_eval_res = RAG_eval.wiki_check(item)
                            if truthfulness_eval_res:
                                batch_data[batch_data.index(item)] = truthfulness_eval_res
                            else:
                                batch_data[batch_data.index(item)] = item
                queue.put(batch_data)
                return batch_data
            except Exception as e:
                print(traceback.format_exc())
                return None

        total_batches = int(self.generation_number / self.batch_size)
        print(f'total_batches:{total_batches}', color='GREEN', )

        def save_dataset(generated_dataset):
            """
            Save the dataset to a JSON file.
            """
            try:
                current_time = datetime.now()
                human_readable_time = current_time.strftime("%Y-%m-%d %H:%M:%S")
                config=copy.deepcopy(self.config)
                del config['api_settings']
                config['data_entry_num'] = len(generated_dataset)
                filtered_dataset = list(filter(lambda item: item['isgood'], generated_dataset))
                config['filtered_data_entry_num'] = len(filtered_dataset)
                genset = {
                    'update_time': human_readable_time,
                    "config": config,
                    "dataset": generated_dataset
                }
                    
                base_dir = os.path.dirname(generated_data_file_path)
                if not os.path.exists(base_dir):
                    os.makedirs(base_dir)
                    
                save_json(genset,generated_data_file_path)
                print(f"Data save path:{generated_data_file_path}\n\n")
                print("Dataset saved successfully.", color='BLUE', )
            except Exception as e:
                print(f"Failed to save dataset: {e}")

        all_data = []

        def save_data_to_file(queue):
            while True:
                data = queue.get() 
                if data == "DONE":
                    break
                all_data.extend(data)
                save_dataset(all_data)
                queue.task_done()

        data_queue = Queue()
        save_thread = Thread(target=save_data_to_file, args=(data_queue,))
        save_thread.start()
        with ThreadPoolExecutor(max_workers=self.max_worker) as executor:
            futures = [executor.submit(batch_generation, batch_id=i, queue=data_queue) for i in range(total_batches)]
            for _ in tqdm(as_completed(futures), total=total_batches, desc="Processing Batches"):
                pass
        data_queue.put("DONE")
        save_thread.join()
        
        
def unigen_generation(config):
    generator = UniGen(config)
    generator.run()


def eval_generation(config):
    dataset_name = config['dataset_name']
    generation_number = config['generation_number']
    data_file_path = config['data_file_path']
    generated_data_file_path = config['generated_file']

    data_file_path = f"test_dataset/{dataset_name}/{dataset_name}.json"
    generator = UniGen(config)
    generator.run(data_file_path, generated_data_file_path)
    
def eval_generation(config):
    dataset_name = config['dataset_name']
    generation_number = config['generation_number']
    data_file_path = config['data_file_path']
    generated_data_file_path = config['generated_file']

    data_file_path = f"test_dataset/{dataset_name}/{dataset_name}.json"
    generator = UniGen(config)
    generator.run(data_file_path, generated_data_file_path)