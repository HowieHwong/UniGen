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

# import debugpy
# try:
#     # 5678 is the default attach port in the VS Code debug configurations. Unless a host and port are specified, host defaults to 127.0.0.1
#     debugpy.listen(("localhost", 9501))
#     print("Waiting for debugger attach")
#     debugpy.wait_for_client()
# except Exception as e:
#     pass


warnings.filterwarnings("ignore")

DEBUG = True
config = dict()


class DyGenset:
    def __init__(self,
                 model=None,
                 generation_number=None,
                 batch_size=None,
                 dataset_description=None,
                 dataset_constraint=None,
                 dataset_name="",
                 temperature=None,
                 few_shot_num=None,
                 max_tokens=1000,
                 random_example=False,
                 with_label=True,
                 max_worker=None,
                 embedding_model="text-embedding-ada-002",
                 label_ratio=None,
                 **kwargs):

        config = ConfigManager.get_config_dict()
        dataset_config = config['dataset_configuration']
        pprint(dataset_config)
        dataset_description = dataset_config['dataset_configuration']['dataset_description']

        batch_size = config['generation_settings']['batch_size']
        few_shot_num = config['generation_settings']['few_shot_num']
        self.model = model
        self.dataset_description = dataset_description
        self.dataset_constraint = dataset_constraint
        self.dataset_name = dataset_name
        self.temperature = config['generation_settings']['temperature']
        self.random_example = random_example
        self.max_tokens = max_tokens
        self.with_label = dataset_config['dataset_configuration']['with_label']
        self.with_attr = dataset_config['dataset_configuration']['with_attr']
        self.add_attribute = dataset_config['dataset_configuration']['add_attribute']
        self.extra_info_keys = dataset_config['dataset_configuration'].get('extra_info_keys', [])
        self.max_worker = config['generation_settings']['max_worker']
        self.generation_number = generation_number
        self.label_ratio = label_ratio
        self.batch_size = batch_size
        self.prompt_template = config["prompt"]
        self.few_shot_num = few_shot_num
        self.attr_key = dataset_config['dataset_configuration']['attr_key'] if self.with_attr or self.add_attribute else None

    def initialize_prompt(self):
        initial_prompt = self.prompt_template["initial_prompt"]
        initial_prompt = initial_prompt.replace("[[BATCH_SIZE]]", self.batch_size)
        prompt = initial_prompt + self.dataset_description + self.dataset_constraint
        return prompt

    def preprocess_input(self, file_path):
        # ensure the format of base dataset: {"text": "xxx", "label": "xxx"} or ["text_1", "text_2", ...]
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        assert isinstance(data, list)
        if self.with_label:
            assert 'text' in list(data[0].keys()) and 'label' in list(data[0].keys())
        return data

    def example_selection(self, data, random_=False):
        keys = ['text', 'label'] + self.extra_info_keys
        filtered_data = []
        if self.attr_key:
            keys = keys + [self.attr_key, ]

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
            if not os.path.exists(embedding_path.format(self.dataset_name)):
                embeddings = embedding.generate_dataset_embedding(data, self.embedding_model)

                save_json(embeddings, embedding_path.format(self.dataset_name))

            else:
                embeddings = load_json(embedding_path.format(self.dataset_name))

            examples = embedding.cluster_embeddings(data, embeddings, num_clusters=self.few_shot_num)

        random.shuffle(examples)
        # examples = data[:self.few_shot_num]
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

    def save_few_shot(self, examples, batch):

        # 将JSON数据保存到文件
        json_data = examples
        file_path = f'{self.dataset_name}_{batch}.json'  # 指定文件名和路径
        save_json(examples, file_path)
        print("JSON data has been saved to", file_path)

    def load_few_shot(self, batch):

        # 将JSON数据保存到文件
        # json_data=examples
        file_path = f'{self.dataset_name}_{batch}.json'  # 指定文件名和路径
        examples = load_json(file_path)
        print(f'{self.dataset_name}_{batch}.json')
        return examples

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
        dataset_config = load_json(f"test_dataset/{self.dataset_name}/config.json")
        generated_dataset = list()
        base_data = self.preprocess_input(dataset_path)

        total_feedback = ""
        if dataset_config["efficiency_configuration"]["learn_from_human_feedback"]:
            if config["generation_settings"]["few_shot_num"] > 0:
                examples = self.example_selection(base_data, self.random_example)
                few_shot_des = self.few_shot_description(examples)
            else:
                few_shot_des = "None"

            if not dataset_config["dataset_configuration"]["dataset_constraint"]:
                constraint_des = self.add_constraints(dataset_config["dataset_configuration"]["dataset_constraint"])
            else:
                constraint_des = ""
            description_prompt = self.prompt_template["description_prompt"].format(
                description_for_dataset=self.dataset_description, )


            initial_prompt = self.prompt_template["initial_prompt"].format(batch_size=self.batch_size,
                                                                           dataset_constraint=constraint_des,
                                                                           few_shot_examples=few_shot_des)
            initial_prompt = description_prompt + initial_prompt

            if self.with_attr:
                initial_prompt += attribute.add_attributes(examples=examples, attr_key=self.attr_key, attr=None)
            initial_prompt += data_format.data_entry_format(el_num=self.batch_size, with_label=self.with_label,
                                                            attr_key=self.attr_key)
            assert dataset_config["efficiency_configuration"]["feedback_iteration"] > 0
            for iter in range(0, dataset_config["efficiency_configuration"]["feedback_iteration"]):
                res_data = data_format.get_res_data(initial_prompt)
                data_item = data_format.extract_data_item(res_data)
                feedback = self.learn_from_human_feedback(data_item)
                if iter == 0:
                    feedback = self.prompt_template["feedback_prefix"] + feedback
                total_feedback += feedback

            print("-------------------------------Human Feedback-------------------------------", "GREEN")
            print(total_feedback)
            print("----------------------------------------------------------------------------", "GREEN")

        @retry(wait=wait_random_exponential(min=5, max=20), stop=stop_after_attempt(3))
        def batch_generation(batch_id, queue):
            try:
                batch_data = []

                if config["generation_settings"]["few_shot_num"] > 0:
                    examples = self.example_selection(base_data, self.random_example)
                    # examples=self.load_few_shot(batch_id)
                    few_shot_des = self.few_shot_description(examples)
                    # self.save_few_shot(examples,batch_id)
                    # return

                if dataset_config["dataset_configuration"]["dataset_constraint"] != []:
                    constraint_des = self.add_constraints(dataset_config["dataset_configuration"]["dataset_constraint"])
                else:
                    constraint_des = ""

                description_prompt = self.prompt_template["description_prompt"].format(
                    description_for_dataset=self.dataset_description,
                )
                initial_prompt = self.prompt_template["initial_prompt"].format(batch_size=self.batch_size,
                                                                               dataset_constraint=constraint_des,
                                                                               few_shot_examples=few_shot_des)
                epoch_prompt = description_prompt + initial_prompt

                if total_feedback:
                    epoch_prompt += total_feedback
                if self.add_attribute:
                    examples = attribute.get_attribute(examples, dataset_description=self.dataset_description)
                    self.with_attr = True
                if self.with_attr:
                    epoch_prompt += attribute.add_attributes(examples=examples, attr_key=self.attr_key, attr=None)
                epoch_prompt += data_format.data_entry_format(el_num=self.batch_size, with_label=self.with_label,
                                                              attr_key=self.attr_key)
                res_data = data_format.get_res_data(epoch_prompt)
                epoch_data_item = data_format.extract_data_item(res_data)

                with open('prompt.json', 'a') as f:
                    json.dump(epoch_prompt, f)

                if dataset_config["efficiency_configuration"]["self_reflection"]:
                    reflection_res = self_reflection.self_reflection(epoch_data_item, self.dataset_description,
                                                                     few_shot_des, constraint_des)
                    for index, reflect_res in enumerate(reflection_res):
                        reflect_res['text'] = reflect_res['text']
                        if reflect_res['text']:
                            batch_data.append(reflect_res)
                        else:
                            batch_data.append(epoch_data_item[index])

                else:
                    batch_data += epoch_data_item

                if dataset_config["efficiency_configuration"]["math_eval"]:

                    for item in batch_data:
                        print("math_eval", item["text"])
                        batch_data[batch_data.index(item)] = math_eval.math_eval(item)

                if dataset_config["efficiency_configuration"]["truthfulness_eval"]:
                    if batch_data:
                        for item in batch_data:
                            truthfulness_eval_res = RAG_eval.wiki_check(item)
                            if truthfulness_eval_res == "NONE":
                                batch_data[batch_data.index(item)] = item
                            else:
                                batch_data[batch_data.index(item)] = truthfulness_eval_res

                save_json(batch_data,
                                       f'/media/ssd/wtm/DyGenset/test_dataset/TruthfulQA/temper_case/{batch_id}_{self.temperature}.json')
                queue.put(batch_data)
                return batch_data
            except Exception as e:
                print(traceback.format_exc())
                # print(f"Error in running DyGenset: Epoch{_}")
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
                    "dataset_config": dataset_config,
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
        # 启动数据保存线程
        save_thread = Thread(target=save_data_to_file, args=(data_queue,))
        save_thread.start()
        # 并行生成数据
        with ThreadPoolExecutor(max_workers=self.max_worker) as executor:
            futures = [executor.submit(batch_generation, batch_id=i, queue=data_queue) for i in range(total_batches)]
        data_queue.put("DONE")
        save_thread.join()


def generation(config):
    dataset_name = config['dataset_name']
    generation_number = config['generation_number']
    few_shot_num = config['few_shot_num']
    model_type = config['model_type']
    temperature = config['temperature']
    
    data_file_path = f"test_dataset/{dataset_name}/{dataset_name}.json"
    generated_data_file_path = check_and_rename_file(
        f"test_dataset/{dataset_name}/{dataset_name}_{model_type}_generated.json")

    print(f'generation_number: {generation_number}')
    generator = DyGenset(dataset_name=dataset_name,
                         generation_number=generation_number,
                         few_shot_num=few_shot_num,
                         random_example=False)
    generator.run(data_file_path, generated_data_file_path)



