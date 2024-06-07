import json,re
from .LLM_model import ModelAPI
from utils.IO import print,input
from tenacity import retry, wait_random_exponential, stop_after_attempt
from .config import ConfigManager
from .json_parser import clean_json      
import json

config=ConfigManager.get_config_dict()
prompt_template=config["prompt"]

@retry(wait=wait_random_exponential(min=1, max=10), stop=stop_after_attempt(3))
def extract_data_item(data):
    # data = json.loads(response)
    for index, item in enumerate(data):
        item['number'] = index + 1
    return data

@retry(wait=wait_random_exponential(min=2, max=8), stop=stop_after_attempt(8))
def get_res_data(prompt):
    response=get_res_str(prompt)
    clear_response = clean_json_string(response)
    try:
        data=clean_json(clear_response)
        #data = json.loads(clear_response)
        return data
    except Exception as e:
        print(clear_response,color="GREEN")
        print(e)

@retry(wait=wait_random_exponential(min=2, max=8), stop=stop_after_attempt(8))
def get_res_str(prompt):
    config=ConfigManager.get_config_dict()
    model_type,temperature=config["model_type"],config['temperature']
    print(model_type,temperature)
    LLM_model=ModelAPI(model_type=model_type,
                       temperature=temperature)
    response = LLM_model.get_res(prompt,)
    # response = clean_json_string(response)
    # data = json.loads(response)
    return response
    

def clean_json_string(json_string):
    #print(json_string,'\n\n\n\n\n')
    if "```json" in json_string:
        json_data_match = re.search(r'```json\n([\s\S]*?)\n```', json_string)
        if json_data_match:
            json_data_content = json_data_match.group(1)
            json_string=json_data_content
        else:
            print("No JSON data found!!!:") 
            print(json_string,color='RED')
            print("No JSON data found")
            # json_data_content = "No JSON data found."
    elif "```" in json_string:
        json_data_match = re.search(r'```\n([\s\S]*?)\n```', json_string)
        if json_data_match:
            json_data_content = json_data_match.group(1)
            json_string=json_data_content
        else:
            print("No JSON data found!!!:") 
            print(json_string,color='RED')
            print("No JSON data found")
            # json_data_content = "No JSON data found."
    elif(json_string.startswith('[') or json_string.startswith('{')):
        pass
        # json_string=json_data_content
    else:   
        # 使用正则表达式匹配包含嵌套结构的 JSON
        pattern = r'(\[.*\]|\{.*\})'
        match = re.search(pattern, json_string, re.DOTALL)
        if match:
            json_data_content = match.group(0)
            json_string=json_data_content
        else:
            print("No JSON data found!!!:")  
            print(json_string,color='BLUE')
            
        #json_data_content=json_data_content
    return json_string


def data_entry_format(with_label, el_num, attr_key=None, extra_info_keys=None):
    dict_list = []
    for i in range(el_num):
        el = {'id': i, 'text': ''}
        if with_label:
            el['label'] = ''
        if attr_key is not None:
            el[attr_key] = ''
        if extra_info_keys:
            for key in extra_info_keys:
                el[key] = ''
        dict_list.append(el)
    json_output = json.dumps(dict_list, indent=4)  
    data_format_str = prompt_template['return_format_prompt'].format(batch_size=el_num, data_format=json_output)
    return data_format_str



def prompt_format(s, **replacements):
    for symbol, content in replacements.items():
        if isinstance(content, dict):
            content = json.dumps(content)
        original_s = s
        s = s.replace(f'[[{symbol}]]', content)
        if original_s == s:
            raise ValueError("No symbol Error！")
    return s
