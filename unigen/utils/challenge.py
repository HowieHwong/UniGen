import json
from tenacity import retry, stop_after_attempt, wait_fixed

from .file_process import save_json,load_json
from .prompt import eval_prompt_dict
from .data_format import clean_json_string,get_res_data

from .configuration import ConfigManager
ConfigManager.load_config()
config = ConfigManager.get_config_dict()

principles={"Paraphrasing Question": 'Modify the wording to present the same concept in a more complex manner.',
'Adding Extra Context into Question': "Incorporate additional, somewhat relevant context or details that do not directly aid in answering but increase the question's complexity.",
"Paraphrasing The Choices":"Each choice should be paraphrased to reflect the same concept or idea as the original, The essence and meaning must be preserved. If a choice cannot be paraphrased without changing its meaning, it should be kept unchanged.",
"Adding A New Choice": "Add a plausible but incorrect option to the existing choices to introduce ambiguity and deepen the required understanding."
}

output_format = {
  "initial_analysis": "[Analysis of the current question and its options]",
  "applied_principles": {
    'principles':[],
    'enhancement_applied':{
       "[Detailed]"
    }
    
  }
}

prompt_template=f'''
You are a professional question designer. Please assist me in analyzing and enhancing a question to make it more challenging while ensuring the answer remains unchanged.

Begin by evaluating the current data entry and its structure to determine their complexity and the depth of knowledge they test. 

Objective of the Question: [D]

Current Data Entry: [Q]

Based on this analysis, select and apply only those principles that are applicable to the specific type of question

Principles to Consider:
{principles}

Task:

1.Deep Analysis: Evaluate the current data entry. Describe their strengths and weaknesses and suggest potential areas for enhancement.
2.Choose Enhancements: Based on the initial analysis, explain which principle(s) you chose to apply and how they have been implemented to make the question more difficult.

Output Format:
Please return in a structured JSON format:
{output_format}
'''

output_improved_format={'improved_data_entry':'[maintain exactly same JSON format with the original entry]'}
prompt_improved_template=f'''You are a professional data analyst. Your task is to review and refine data entries to ensure their accuracy and coherence. You will be provided with an original data entry, Analysis and Enhancements comparing the two. Based on this information, you are expected to produce a final revised data entry that maintains the original format but incorporates necessary corrections and improvements.

Inputs:

Original Data Entry: [Original Data Entry]
Analysis and Enhancements of Data Entry: [Enhancements]

Task:

Review the Inputs: Examine the original data entry, the modified version, and the analysis to fully understand the changes and the rationale behind them.
Refine the Data Entry: Apply your professional judgment to adjust the modified data entry. Ensure that all changes are accurate, relevant, and improve upon the original entry while maintaining the same structural format.
Finalize the Revision: Produce a revised data entry that resolves any issues identified in the analysis and enhances the data quality.

Output Format:
Please return the modified content in a structured format:

Please return in a structured JSON format:
{json.dumps(output_improved_format)}
'''



@retry(stop=stop_after_attempt(5), wait=wait_fixed(2))
def improve_el(el,description):
    
    if el['applied_principles'] is not None:
        return el
    
    original_data_entry = str({k: el['original'][k] for k in ['label', 'text'] if k in el['original']})
    
    prompt = prompt_template.replace('[D]', description).replace('[Q]', original_data_entry)
    ###
    el_data=get_res_data(prompt)
    ###
    applied_principles=el_data['applied_principles']
    print(applied_principles['principles'])

    prompt = prompt_improved_template.replace('[Original Data Entry]', str(original_data_entry)).replace('[Enhancements]', str(applied_principles))
    ###
    el_data=get_res_data(prompt)
    ###
    improved=el_data['improved_data_entry']
    
    
    el['applied_principles'] = applied_principles
    el['modified'] = improved
    return el


from concurrent.futures import ThreadPoolExecutor, as_completed
import traceback
from tqdm import tqdm

def apply_improvements(data, description):
    with ThreadPoolExecutor(max_workers=8) as executor:
        # 提交所有任务到线程池
        future_to_el = {executor.submit(improve_el, el, description): el for el in data}
        
        # 使用tqdm包装as_completed以显示进度条
        for future in tqdm(as_completed(future_to_el), total=len(future_to_el), desc=description):
            try:
                el = future.result()  # 获取结果会自动更新data列表中对应的元素
            except Exception as exc:
                traceback.print_exc()
                print(f"Element generated an exception: {exc}")


def process_dataset(dataset_name, data_path):

    ConfigManager.load_description()
    description = ConfigManager.get_description()
    
    data = load_json(data_path)
    
    new_data = []
    for el in data:
        new_el = dict()
        new_el['original'] = el
        new_el['applied_principles'] = None
        new_el['modified'] = None
        new_data.append(new_el)
        
    apply_improvements(new_data[:],description[dataset_name])
    
    return new_data

# Example usage
dataset_name = 'HellaSwag'
config_path = '/media/ssd/wtm/DyGenset/test_dataset'
data_path = '/media/ssd/wtm/DyGenset/eval_set/dygenset_2/test_data/HellaSwag_gen_vallina1.json'

processed_data = process_dataset(dataset_name, config_path, data_path)
print(processed_data)