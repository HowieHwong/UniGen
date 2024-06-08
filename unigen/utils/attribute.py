from . import data_format
from .IO import print,input

from .prompt import prompt_template




def get_attribute(example,dataset_description):
    prompt = prompt_template['attribute_prompt'].format(description=dataset_description, few_shot_examples=example)
    with_label = "label" in example[0].keys()
    res_data=data_format.get_res_data(prompt)
    added_example=data_format.extract_data_item(res_data)           
    print(added_example)
    return added_example
    
    
    
def add_attributes(attr_key='category',examples=None,attr=None):
    if attr is None:
        attr = [el[attr_key] for el in examples]
    attributes_text_list=[]
    attributes_text = f"Generate data should focus on these {attr_key}:{attr}"

    return attributes_text

    
    