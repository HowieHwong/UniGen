import os
import subprocess
import tempfile
from copy import deepcopy
import file_process,data_format
from .IO import print
from tenacity import retry, wait_random_exponential, stop_after_attempt
from .prompt import prompt_template


def execute_code(code):
    with tempfile.NamedTemporaryFile(delete=False, suffix='.py') as tmpfile:
        tmpfile.write(code.encode())
        tmpfile_path = tmpfile.name

    try:
        output = subprocess.check_output(['python', tmpfile_path], stderr=subprocess.STDOUT, text=True)
        os.remove(tmpfile_path)  # Clean up the temp file immediately after execution
        return output.strip(), True
    except subprocess.CalledProcessError as e:
        os.remove(tmpfile_path)
        return 'Error: ' + str(e.output), False


@retry(wait=wait_random_exponential(min=1, max=3), stop=stop_after_attempt(3))
def math_eval(example):
    example=deepcopy(example)
    cnt = 0
    success = False
    error_message=''
    eval_process = [] 
    
    while cnt < 5 and not success:
        formatted_prompt = prompt_template['math_eval'].format(expression=example['text'])
        if cnt != 0:  # not First attempt
            formatted_prompt += error_message
        response_json=data_format.get_res_data(formatted_prompt)
        print("*" * 15 + "Sample Information" + "*" * 15, "GREEN")
        print(example)
        print("*" * 15 + "Generated Code" + "*" * 15, "GREEN")
        print(response_json.get('Code'))
        print("*" * 15 + "Analysis" + "*" * 15, "GREEN")
        print(response_json.get('Analysis'))
        print("*" * 35, "GREEN")

        # Save the code attempt for potential inclusion in the next attempt's error message
        epoch_code = response_json.get('Code')
        feedback, program_success = execute_code(response_json.get('Code'))
        epoch_eval_process={   'epoch':cnt,
            'epoch_code':epoch_code,
            'LLM_answer':example['label'],
            'program_answer':feedback,
        }
        

        if not program_success:
            cnt += 1
        else:
            success = True
            print("Program answer: " + feedback, "BLUE")
            prompt= prompt_template['math_eval_compare'].format(response1=str(feedback), response2=str(example['label']))
            compare_result=data_format.get_res_str(prompt)
            epoch_eval_process['compare']=compare_result
            if 'true' in compare_result.lower():
                print("Math Evaluation is same to the label.", "Green")
                example['math_eval']="Same"
                eval_process.append(epoch_eval_process)
                break
                #return example
            else:
                cnt += 1  # If comparison is false, consider it as an attempt and increase the counter
        eval_process.append(epoch_eval_process)
                
    if program_success and not feedback=="":
        example['math_eval'] = "python"
        example['label'] = feedback
        print("Math Evaluation is different to the label.", "RED")
    else:
        example['math_eval'] = "original"
        print("Math Evaluation fail，original answer will be saved.", "RED")
    
    example['math_eval_process'] = eval_process  # Store the evaluation process in the example dictionary
    return example

