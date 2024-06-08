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

ans_format = {
    "reflection": """(If isgood is "yes", include reasons here. If "no", include a detailed analysis here.)""",
    "isgood": "yes/no",
}

few_shot_examples = '''Few Shot Examples:
[[few_shot_des]]
'''

reflection_generation = '''You are a professional dataset generation assistant. Your task is to assess the quality of the provided Data Entry based on dataset description,few shot examples and criteria such as quality, format, relevance, accuracy, and challenge level. 
DATASET DESCRIPTION:[[description]]

[[few_shot_examples]]

[[constraint]]

Provide your evaluation in string format, formatted as JSON. For each question in the dataset, provide a detailed analysis in the 'reflection' field discussing the question's merits and shortcomings first. Identify its strengths, point out any weaknesses, suggest potential improvements, and evaluate the complexity of the question to ensure it stick to the purpose in dataset description and meets the challenge level. After reflecting, indicate in the 'isgood' field whether the question satisfies the expected standards. Use 'yes' ONLY if both conditions are met comprehensively. If the question falls short in any aspect, mark 'no'.

Data Entry for Evaluation:
[[example]]

Your assessment and reflection must be formatted as following JSON:
[[ans_format]]
Directly output your improved example as the following JSON format:'''

reflection_generation = reflection_generation.replace("[[ans_format]]", json.dumps(ans_format))

improve_examples_with_reflection = '''You are a professional dataset generation assistant.
Your task is to create improved versions of the original example based on the reflection and ,few shot examples.

DATASET DESCRIPTION:[[description]].

[[few_shot_examples]]

[[constraint]]

Ensure that the improvements address the identified weaknesses and enhance the strengths.


Original Data Entry:
[[original_example]]

Reflection:[[reflection]]


The structure and form of the improved example should be SAME with the original example; DO NOT make significant changes to the existing example. Directly output your improved Data Entry as the following JSON format:'''

prompt_template['reflection_generation'] = reflection_generation
prompt_template['improve_examples_with_reflection'] = improve_examples_with_reflection
prompt_template['few_shot_examples'] = few_shot_examples


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


def self_reflection(examples, dataset_description, few_shot_des, constraint, max_reflection=5, ):
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
            example, isgood, reflection_ = reflection_improve_example(example, dataset_description, few_shot_des,
                                                                      constraint)
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
        #return None
