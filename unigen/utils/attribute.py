from . import data_format
from .IO import print,input
from .config import ConfigManager
config=ConfigManager.get_config_dict()
prompt_template=config["prompt"]


attr_prompt='''My goal is to enhance the diversity of the dataset. I will provide an overall description of the dataset each time, along with a few examples from the original dataset. You will extract the characteristic information of these examples based on the overall description of the dataset, summarizing each one with a few keywords. Ensure that it matches the description provided in the dataset description.
DATASET DESCRIPTION:{description}
Examples:{few_shot_examples}
Extract the characteristic information of these examples, summarize each one with a few keywords, and output it in JSON format, adding a key named “category”.
'''

def get_attribute(example,dataset_description):
    prompt = attr_prompt.format(description=dataset_description, few_shot_examples=example)
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

    
    