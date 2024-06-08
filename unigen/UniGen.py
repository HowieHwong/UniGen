import os.path
import openai
import sys
import json
import typing
import random
import tiktoken
from tqdm import tqdm
from pprint import pprint
from utils import attribute,embedding, data_format, RAG_eval, math_eval, self_reflection
from utils.configuration import ConfigManager
from utils.IO import print, input
from utils.file_process import save_json,load_json,check_and_rename_file
from joblib import Parallel, delayed
from threading import Thread
from queue import Queue
import warnings
import traceback
from dataclasses import dataclass, field
from simple_parsing import ArgumentParser
from concurrent.futures import ThreadPoolExecutor
from threading import Thread
from datetime import datetime
from tenacity import retry, wait_random_exponential, stop_after_attempt

warnings.filterwarnings("ignore")

DEBUG = True


class UniGen:
    def __init__(self,
                 config,
                 **kwargs):

        dataset_config = config['dataset_configuration']
        self.efficiency_configuration=config['efficiency_configuration']
        pprint(dataset_config)
        dataset_description = dataset_config['dataset_configuration']['dataset_description']
        self.batch_size = config['generation_settings']['batch_size']
        self.few_shot_num = config['generation_settings']['few_shot_num']
        self.generation_number =config['generation_settings']['generation_number']
        self.dataset_description = dataset_description
        self.constraints=dataset_config["dataset_configuration"]["dataset_constraint"]
        self.dataset_name = config["dataset_configuration"]["dataset_name"]
        self.temperature = config['generation_settings']['temperature']
        self.random_example = dataset_config['dataset_configuration']['with_label']
        self.with_label = dataset_config['dataset_configuration']['with_label']
        self.with_attribute = dataset_config['dataset_configuration']['with_attribute']
        self.add_attribute = dataset_config['dataset_configuration']['add_attribute']
        self.extra_info_keys = dataset_config['dataset_configuration'].get('extra_info_keys', [])
        self.max_worker = config['generation_settings']['max_worker']
        self.prompt_template = config["prompt"]
        self.attr_key = dataset_config['dataset_configuration']['attribute_key'] if self.with_attribute or self.add_attribute else None

    def initialize_prompt(self):
        initial_prompt = self.prompt_template["initial_prompt"]
        initial_prompt = initial_prompt.replace("[[BATCH_SIZE]]", self.batch_size)
        prompt = initial_prompt + self.dataset_description + self.dataset_constraint
        return prompt

    def preprocess_input(self, file_path):
        data=load_json(file_path)
        assert isinstance(data, list)
        if self.with_label:
            assert 'text' in list(data[0].keys()) and 'label' in list(data[0].keys())
        return data

    def example_selection(self, data, random_=False):
        keys = ['text', 'label'] + self.extra_info_keys
        filtered_data = []
        if self.attribute_key:
            keys = keys + [self.attribute_key, ]
        for item in data:
            filtered_item = {k: item[k] for k in keys if k in item}
            filtered_data.append(filtered_item)
        data = filtered_data
        assert isinstance(data, list)
        if random_:
            random.shuffle(data)
            examples = data[:self.few_shot_num]
        else:
            embedding_path = 'embedding/{}_dataset_embedding.json'
            Embedder = embedding.EmbeddingProcessor(config=self.efficiency_configuration)
            if not os.path.exists(embedding_path.format(self.dataset_name)):
                embeddings = Embedder.generate_dataset_embedding(data)
                save_json(embeddings, embedding_path.format(self.dataset_name))
            else:
                embeddings = load_json(embedding_path.format(self.dataset_name))
            examples = Embedder.cluster_embeddings(data, embeddings, num_clusters=self.few_shot_num)
        random.shuffle(examples)
        filtered_example = []
        for item in examples:
            filtered_item = {k: item[k] for k in keys if k in item}
            filtered_example.append(filtered_item)

        print(filtered_example[1].keys())
        return filtered_example

    def few_shot_description(self, examples):
        random.shuffle(examples)
        json_output = json.dumps(examples, indent=4)
        return json_output

    def add_constraints(self, constraints):
        constraints_text = self.prompt_template["constraints_prefix"]

        for i, constraint in enumerate(constraints, 1):
            constraints_text += f"{i}. {constraint}\n"
        constraints_text += self.prompt_template["constraints_suffix"]
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
        if DEBUG:
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

    def label_constrain(self, raw_data, label_ratio):
        assert isinstance(raw_data, list)
        assert isinstance(label_ratio, dict)
        return_data = []
        for k, v in label_ratio.items():
            k_label_data = [el for el in raw_data if el['label'] == k]
            if v > len(k_label_data):
                raise ValueError(f"label {k} needs {v} examples, but only {len(k_label_data)} examples in dataset")
            k_label_data = random.sample(k_label_data, v)
            return_data.extend(k_label_data)
        return return_data

    def diversity_setting(self):
        pass

    def run(self, dataset_path, generated_data_file_path):
        assert self.generation_number % self.batch_size == 0, "generation_number must be divisible by batch_size"
        base_data = self.preprocess_input(dataset_path)

        @retry(wait=wait_random_exponential(min=5, max=20), stop=stop_after_attempt(3))
        def batch_generation(batch_id, queue):
            try:
                batch_data = []
                if self.few_shot_num > 0:
                    examples = self.example_selection(base_data, self.random_example)
                    few_shot_des = self.few_shot_description(examples)
                else:
                     constraint_des = ""
                if self.constraints != []:
                    constraint_des = self.add_constraints(self.constraints)
                else:
                    constraint_des = ""
                description_prompt = self.prompt_template["description_prompt"].format(
                    description_for_dataset=self.dataset_description,
                )
                initial_prompt = self.prompt_template["initial_prompt"].format(batch_size=self.batch_size,
                                                                               dataset_constraint=constraint_des,
                                                                               few_shot_examples=few_shot_des)
                epoch_prompt = description_prompt + initial_prompt

                if self.add_attribute:
                    examples = attribute.get_attribute(examples, dataset_description=self.dataset_description)
                    self.with_attribute = True
                if self.with_attribute:
                    epoch_prompt += attribute.add_attributes(examples=examples, attribute_key=self.attribute_key, attr=None)
                epoch_prompt += data_format.data_entry_format(el_num=self.batch_size, with_label=self.with_label,
                                                                attribute_key=self.attribute_key)
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
                        print("math_eval", item["text"])
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
                config = ConfigManager.get_config_dict()
                config['data_entry_num'] = len(generated_dataset)
                filtered_dataset = list(filter(lambda item: item['isgood'], generated_dataset))
                config['filtered_data_entry_num'] = len(filtered_dataset)
                genset = {
                    'update_time': human_readable_time,
                    "dataset_config": self.config,
                    "gen_config": config,
                    "dataset": generated_dataset
                }
                with open(generated_data_file_path, 'w') as f:
                    json.dump(genset, f, ensure_ascii=False, indent=4)
                    print("Dataset saved successfully.", color='BLUE', )
            except Exception as e:
                print(f"Failed to save dataset: {e}")

        all_data = []

        def save_data_to_file(queue):
            print(f"Data save path:{generated_data_file_path}\n\n\n")
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
        data_queue.put("DONE")
        save_thread.join()


def generation(config):
    dataset_name = config['dataset_name']
    generation_number = config['generation_number']
    data_file_path = config['data_file_path']
    generated_data_file_path = config['generated_file']

    data_file_path = f"test_dataset/{dataset_name}/{dataset_name}.json"
    generator = UniGen(config)
    generator.run(data_file_path, generated_data_file_path)



