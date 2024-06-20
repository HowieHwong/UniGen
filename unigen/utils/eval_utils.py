

import os
import json
import re
import random
from typing import Dict, Any
from tqdm import tqdm

import matplotlib.pyplot as plt
import concurrent.futures
import numpy as np
import concurrent.futures as futures

from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
import logging
import os

from .file_process import save_json,load_json
from .prompt import eval_prompt_dict
from .data_format import clean_json_string


def find_uppercase_ABCD(text):
    """
    Finds all occurrences of uppercase A, B, C, D that are not part of longer words
    and returns them as a list.

    Args:
    - text (str): Text to search within.

    Returns:
    - list: A list of all occurrences of uppercase A, B, C, D not followed by another letter.
    """
    # This pattern matches A, B, C, D that are not immediately followed by another alphabetic character.
    matches = re.findall(r'\b([ABCDEFG])(?!\w)', text)
    return matches


def find_initial_capital(text):
    """
    Finds the first capital letter at the beginning of the string.
    If there's no capital letter at the beginning, it considers the match unsuccessful.

    Args:
    - text (str): Text to search within.

    Returns:
    - str: The first capital letter found at the beginning of the string, or an indication that the match was unsuccessful.
    """
    match = re.match(r'^\s*([A-Z])', text)
    if match:
        return match.group(1)
    else:
        return 0



def extract_options(self, text):
    """
    Extracts multiple choice options from a given text.

    Args:
    - text (str): Text containing multiple choice options.

    Returns:
    - dict: A dictionary mapping option numbers to option text.
    """
    matches = re.findall(r'\((\d+)\)\s+([A-Za-z\s]+)', text)
    return {match[0]: match[1].strip() for match in matches}

def multiple_choice_eval(self, data):
    """
    Evaluates emotional awareness in given data.

    Args:
    - data (list): List of data items to be evaluated.

    Returns:
    - float: The proportion of correctly identified emotions.
    """
    assert isinstance(data, list), "Data must be a list."

    total_length = len(data)
    total_correct = 0

    for el in data:
        golden_answer = self.extract_options(el['prompt'])
        golden_word = golden_answer[el['label']]
        all_words = list(golden_answer.values())
        flag = 0

        if golden_word.lower() in el['res'].lower():
            flag = 0
            for word in all_words:
                if word != golden_word and word.lower() in el['res'].lower():
                    flag = 1
                    break
        if flag == 0 and golden_word.lower() in el['res'].lower():
            total_correct += 1
        elif el['label'] in el['res']:
            total_correct += 1

    return total_correct / total_length if total_length > 0 else 0


def extract_label(el):
    option=['A','B','C','D',"E",'F','G']

    if isinstance(el['label'], str):
        if el['label'].isalpha() and el['label'].upper() in option:
            return el['label'].upper()
        else:
            initial_capital=find_initial_capital(el['label'])
            if(initial_capital):
                uppercase_ABCD=find_uppercase_ABCD(el['label'])
                if len(list(set(uppercase_ABCD)))==1 and initial_capital in uppercase_ABCD:
                    return el['label']
    return None
    
    pass
def extract_res(el):
    if not el.get('res'):
        return None
    initial_capital=find_initial_capital(el['res'])
    if(initial_capital):
        uppercase_ABCD=find_uppercase_ABCD(el['res'])
        if len(list(set(uppercase_ABCD)))==1 and initial_capital in uppercase_ABCD:
            return initial_capital
    return None


def str_to_bool(s):
    return s.lower() in ["true", "1", "yes"]



def evaluate_accuracy(filepath, eval_prompt_dict,task):

    dirname, filename = os.path.split(filepath)
    new_filename =  "evaluated_"+filename
    new_filepath = os.path.join(dirname, new_filename)

    ans_data = load_json(filepath)

    if os.path.exists(new_filepath):
        data = load_json(new_filepath)

        for i, (orig_dict, new_dict) in enumerate(zip(ans_data,data)):
            orig_text = orig_dict.get('text')
            new_text = new_dict.get('text')
            if orig_text == new_text:
                orig_res = orig_dict.get('res')
                new_res = new_dict.get('res', None)  # 使用 None 作为默认值
                
                # 如果 'res' 键值不同（包括新数据中 'res' 不存在的情况）
                if orig_res != new_res:
                    # 更新原始数据中的 'res' 值
                    data[i]['res'] = orig_res
                    #print(orig_res)
    else:
        data=ans_data

    if task in ['MMLU-constrain']:
        print(task*10)
        for el in data:
            el['gpt_eval']=True
        evaluator = AutoEvaluator()
        all_eval_data = evaluator.evaluate(data, task=task, eval_prompt_dict=eval_prompt_dict,save_filepath=new_filepath)
        for el in all_eval_data:
            eval_res = el.get('eval_res')
            if eval_res == 'YES':
                el['correct']=True 
            elif eval_res == 'NO':
                el['correct']=False 
        
    
    
    elif task not in ['gsm8k','MCQ',]:
        for el in data:
            label = extract_label(el)
            res = extract_res(el)
            if label and res:
                if label.lower() == res.lower():
                    el['correct']=True
                else:
                    el['correct']=False  
                el['gpt_eval']=False 
            else:
                el['gpt_eval']=True
        evaluator = AutoEvaluator()
        all_eval_data = evaluator.evaluate(data, task=task, eval_prompt_dict=eval_prompt_dict,save_filepath=new_filepath)
        for el in all_eval_data:
            eval_res = el.get('eval_res')
            if eval_res == 'YES':
                el['correct']=True 
            elif eval_res == 'NO':
                el['correct']=False 
    else:
        for el in data:
            if 'res' in el:
                el['gpt_eval']=True 
            else:
                print('no _res!')
        print(len(data))
        evaluator = AutoEvaluator()
        all_eval_data = evaluator.evaluate(data, task=task, eval_prompt_dict=eval_prompt_dict,save_filepath=new_filepath)
        for el in all_eval_data:
            try:
                eval_res = el.get('eval_res')
                #print('1')
                if eval_res is None:
                    continue  
                #print(eval_res,'\n\n\n\n')
                eval_res=clean_json_string(eval_res)
                is_same=json.loads(eval_res).get('is_same')
                #el['correct']=str_to_bool(is_same)
                #print(type(is_same))
                if isinstance(is_same, bool):
                    el['correct'] = is_same
                    #print('bool!!!')
                elif isinstance(is_same, str):
                    el['correct'] = str_to_bool(is_same)
            except Exception as e:
                print(eval_res is None )
                if 'correct' in el:
                    del el['correct']
                if 'eval_res' in el:
                    del el['eval_res']
                import traceback; traceback.print_exc();

    # Save the evaluated data
    save_json(all_eval_data, new_filepath)
    right_num=sum(el.get('correct')==True for el in all_eval_data)
    wrong_num=sum(el.get('correct')==False for el in all_eval_data)
    total_length=right_num+wrong_num
    accuracy = right_num / total_length  if  total_length> 0 else 0
    
    return accuracy,total_length





def eval_single(model,task,filename,base_dir):
    try:
        file_path=os.path.join(base_dir,'test_res',model, filename)
        print(file_path)
        if os.path.exists(file_path):
            accuracy,total_length=evaluate_accuracy(file_path,eval_prompt_dict,task)
            return (model,task,filename,accuracy,total_length)
        return  (model,task,filename,0.0,0)
    except Exception as e:
        print(e)
        return  (model,task,filename,0.0,0)
    

# def evaluate_models(models, tasks_files,base_dir):
#     tasks_results = {
#         task: {model: [] for model in models} for task in tasks_files
#     }
    
#     with futures.ThreadPoolExecutor(max_workers=8) as executor:
#         # 创建所有评估任务
#         future_to_params = {}
#         for task, files in tasks_files.items():
#             for model in models:
#                 for file in files:
#                     future = executor.submit(eval_single, model, task, file,base_dir)
#                     future_to_params[future] = (model, task, file)
        
#         # 收集结果
#         for future in futures.as_completed(future_to_params):
#             model, task, file = future_to_params[future]
#             model,task,filename,accuracy,total_length= future.result()
#             tasks_results[task][model].append((filename, accuracy,total_length))
#             #accuracy, total_length = future.result()  # 假设eval_single返回两个值: accuracy 和 total_length
#             #tasks_results[task][model].append((file, accuracy, total_length))
#             print(f"Completed evaluation: {model} on {task} with file {file}, Accuracy: {accuracy:.2f}")
        
#         # 确保每个任务的结果与原始文件列表顺序一致
#         for task, models_results in tasks_results.items():
#             for model, results in models_results.items():
#                 results.sort(key=lambda x: tasks_files[task].index(x[0]))

#     return tasks_results


# import pandas as pd

# def save_csv(tasks_results,file_name):
#     data=tasks_results
#     columns = ['Model', 'Filename', 'S', 'C']

#     # 将数据转换为DataFrame格式
#     rows = []
#     for category, models in data.items():
#         for model, results in models.items():
#             for result in results:
#                 rows.append([model, result[0], result[1], result[2]])

#     df = pd.DataFrame(rows, columns=columns)

#     # 将数据转换为图片中的格式
#     df_pivot = df.pivot_table(index='Model', columns='Filename', values=['S', 'C'])

#     # 将多级列索引转换为单级列索引
#     df_pivot.columns = ['_'.join(col).strip() for col in df_pivot.columns.values]

#     # 重置索引
#     df_pivot.reset_index(inplace=True)

#     # 保存为CSV文件
#     df_pivot.to_csv(file_name, index=False)







class AutoEvaluator:
    """
    A class for automating the evaluation of text using the OpenAI API.
    """

    def __init__(self, save_dir='saved_evaluations'):
        """
        Initialize the AutoEvaluator class.

        Args:
            save_dir (str): Directory for saving evaluation results.
        """
        self.save_dir = save_dir
        self.max_worker=8
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)


    def save_progress(self, data,file_path):
        """
        Save evaluation progress to a JSON file.

        Args:
            data: Data to be saved.
            filename (str): Name of the file for saving the data.
        """
        save_path = file_path
        save_json(data, save_path)
        logging.info("Progress saved to %s", save_path)

    def evaluate(self, data, task, resume=False, save_filepath='eval_progress.json',eval_prompt_dict=None, concat=False):
        """
        Evaluate a given dataset using a specified task.

        Args:
            data: Data to be evaluated.
            task (str): The task identifier for the evaluation.
            resume (bool): Flag to resume from saved progress. Default is False.
            progress_filename (str): The filename for saving or resuming progress.
            concat (bool): Flag to concatenate responses. Default is True.

        Returns:
            The evaluated data.
        """
        progress_filename=''
        
        for el in data:
            if 'res' in el:
                el['gpt_eval']=True 
            else:
                print('no _re@!!!')

        def save_progress_callback(future):
            if future.exception() is not None:
                logging.error("An error occurred: %s", str(future.exception()))
                # Save progress in case of an error
                print(save_filepath)
                self.save_progress(data, filename=save_filepath)

        def process_item(item, el):
            try:
                if 'res' in el:
                    if el['gpt_eval']==True and (not ('eval_res' in el) or el.get('eval_res') is None):
                        eval_res = get_res(item)
                        el['eval_res'] = eval_res
                        logging.info("Evaluated item: %s", item)
                        logging.info("Evaluated result: %s", eval_res)
                else:
                    print('no_res')
                    print(el.keys())
            except Exception as e:
                import traceback
                traceback.print_exc()
                print(e)
                #logging.error("Error processing item %s: %s", item, str(e))
                # self.save_progress(data, filename=progress_filename)
                raise

        eval_prompt_dict = eval_prompt_dict
        prompt_data = []

        if not concat:
            replace_dict = eval_prompt_dict.get(task, {}).get('mapping', {})
            prompt = eval_prompt_dict.get(task, {}).get('prompt', '')
            print(len(data),'aaa')
            for el in data:
                if el.get('res'):
                    single_prompt = prompt
                    for k, v in replace_dict.items():
                        single_prompt = single_prompt.replace(k, str(el[v]))
                    prompt_data.append(single_prompt)
        else:
            prompt = eval_prompt_dict.get(task, {}).get('prompt', '')
            prompt_data = [prompt + item['res'] for item in data if item.get('res')]

        if resume:
            load_path=save_filepath
            try:
                data = load_json(load_path)
                logging.info("Resuming evaluation from saved progress.")
            except FileNotFoundError:
                logging.warning("No saved progress file found at %s. Starting a new evaluation.", load_path)

        assert isinstance(data, list), "Data must be a list."
        assert task is not None, "Task must be specified for evaluation."

        print('Total data number: {}'.format(len(data)))
        print('eval',len(data),len(prompt_data))
        
        for el in data:
            if 'res' in el:
                el['gpt_eval']=True 
            else:
                print('no _res!')
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_worker) as executor:
            futures = [executor.submit(process_item, item, el) for item, el in zip(prompt_data, data)]
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
                future.add_done_callback(save_progress_callback)
            concurrent.futures.wait(futures)
        return data

