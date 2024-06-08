#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from . import data_format
from .data_format import prompt_format
from .IO import print, input
import json
import copy
from queue import Queue
from concurrent.futures import ThreadPoolExecutor
from tenacity import retry, wait_random_exponential, stop_after_attempt
from .prompt import prompt_template
from concurrent.futures import ThreadPoolExecutor
import traceback



@retry(wait=wait_random_exponential(min=2, max=3), stop=stop_after_attempt(3))
def reflection_improve_example(example, dataset_description, few_shot_des, constraint):
    reflection_epoch_trajectory = []
    safe_example = copy.deepcopy(example)
    reflection_epoch_trajectory.append(safe_example)

    few_shot = prompt_format(prompt_template['few_shot_examples'],
                             few_shot_des=few_shot_des)
    prompt = prompt_format(prompt_template['reflection_generation'],
                           description=dataset_description,
                           example=example,
                           constraint=constraint,
                           few_shot_examples=few_shot, )
    reflection_data = data_format.get_res_data(prompt)
    isgood = reflection_data.get("isgood", "Unknown") == "yes"
    reflection_epoch_trajectory.append(reflection_data)

    if (isgood):
        print("*" * 15 + 'No need to improve' + "*" * 15,
              "GREEN")
    else:
        print("*" * 15 + 'Example Need to be Improved' + "*" * 15,
              "GREEN")
        print(example["text"])
        print("*" * 40,
              "GREEN")
        ## improve procedure

        few_shot = prompt_format(prompt_template['few_shot_examples'],
                                 few_shot_des=few_shot_des)
        prompt = prompt_format(prompt_template['improve_examples_with_reflection'],
                               description=dataset_description,
                               reflection=reflection_data['reflection'], original_example=example,
                               constraint=constraint,
                               few_shot_examples=few_shot, )
        with_label = "label" in example.keys()
        data_format_str = data_format.data_entry_format(with_label=with_label, el_num=1)
        prompt += data_format_str

        improved_example = data_format.get_res_data(prompt)
        improved_example = improved_example[0]
        reflection_epoch_trajectory.append(improved_example)

        example['text'] = improved_example["text"]
        if with_label:
            example['label'] = improved_example["label"]

        print("*" * 15 + 'Example After Improvement' + "*" * 15, "GREEN")
        print(example["text"])
        print("*" * 40, "GREEN")
    return example, isgood, reflection_epoch_trajectory


def self_reflection(examples, dataset_description, few_shot_des, constraint, max_reflection=5,):
    with ThreadPoolExecutor(max_workers=len(examples)) as executor:
        futures = [executor.submit(reflect_single_example, example, dataset_description, few_shot_des, constraint,
                                   max_reflection) for example in examples]
        results = []
        for future in futures:
            result = future.result()
            if result is None:
                return None
            results.append(result)
    return results


def reflect_single_example(example, dataset_description, few_shot_des, constraint, max_reflection):
    try:
        reflection_list = []
        for epoch in range(max_reflection):
            print("*" * 15 + f'Self Reflection(Epoch{epoch})' + "*" * 15, "GREEN")
            example, isgood, reflection_ = reflection_improve_example(example, dataset_description, few_shot_des,constraint)
            reflection_list.append({
                'epoch': epoch,
                'process': reflection_
            })
            if (isgood):
                example["isgood"] = isgood
                example["reflection_epoch"] = epoch
                example['reflection_trajectory'] = reflection_list
                break
        if not isgood:
            example["isgood"] = isgood
            example["reflection_epoch"] = max_reflection
            example['reflection_trajectory'] = reflection_list
            print(
                "*" * 15 + 'The Example is not good after the last epoch' + "*" * 15,
                "GREEN")
        return example
    except Exception as e:
        traceback.print_exc()
        print(f'Error during reflection: {e}', "RED")
