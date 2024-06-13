import json

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

prompt_template=dict()
prompt_template['reflection_generation'] = reflection_generation
prompt_template['few_shot_examples'] = few_shot_examples

prompt_template.update({
    "description_prompt": "You are a professional dataset generator. Your primary task is to develop questions that not only adhere closely to the specific requirements outlined in DATASET DESCRIPTION but also push the boundaries of complexity and challenge. While remaining faithful to the given description, strive to craft questions that elevate the level of difficulty as much as possible, encouraging deep engagement and rigorous thinking. The goal is to create a dataset where each question presents a substantial challenge, testing the limits of the respondents' knowledge and problem-solving skills.\n\n DATASET DESCRIPTION:{description_for_dataset}\n\n",
    'attribute_prompt':'''My goal is to enhance the diversity of the dataset. I will provide an overall description of the dataset each time, along with a few examples from the original dataset. You will extract the characteristic information of these examples based on the overall description of the dataset, summarizing each one with a few keywords. Ensure that it matches the description provided in the dataset description.
DATASET DESCRIPTION:{description}
Examples:{few_shot_examples}
Extract the characteristic information of these examples, summarize each one with a few keywords, and output it in JSON format, adding a key named “category”.
''',
    
    "initial_prompt": "The number of entries to be generated in this dataset is {batch_size}.\nBelow are a few examples for your reference:\n\n{few_shot_examples}\n\n{dataset_constraint}\nPlease ensure that the new dataset maintains the purpose of the original data, avoiding any contamination or loss of functionality.\n\n",
    
    "return_format_prompt": "The number of entries to be generated is {batch_size}. Directly return your answer exactly as the following JSON format:\n\n{data_format}\n\n Directly return your answer as JSON format:\n",
    
    "constraints_prefix": "Please note the following constraints when generating new datasets:\n",
    "constraints_suffix": "These are all the constraints. Please adhere to them strictly when generating new datasets.",
    
    "improve_examples_with_human_feedback": "Based on human feedback, please improve and regenerate example.\n\nHUMAN_FEEDBACK:{user_feedback}\n\nEXAMPLE:{example} Generate improved example that reflect the insights and suggestions from the feedback. Directly output the improved example in JSON format, using the structure {\"improved_example\":\"CONTENT\"}",
    "wiki_keyword_extract": "Please analyze the text and identify key entities that are likely to have corresponding articles on Wikipedia for fact-checking purposes. Extract entities such as names of people, places, organizations, historical events, specific technologies, and scientific terms(At most 3) \n\nMy text:{input_text}\n\nDirectly output the list(only one list) of these entities in JSON format, using the structure {{\"entities\":[item1,item2,xxxxx]}}",
    
    "wiki_fact_refine": """Check MY TEXT based on each keyword and content from wikipedia, please check for accuracy against Wikipedia information. MY Data Entry:{input_text}\n\n\n WIKI DATA:{wiki_data} \n\n Check my input_text based on each keyword and content from wikipedia.Correct any misinformation if any mistake in my example. If the information is accurate, please confirm it. Ensure that the final refined TEXT is accurate and contains no factual errors. If the original example is accurate and contains no factual errors, refined_text can be NONE. If original example is not good, make sure the final refined example is right. Finally output in JSON format, using the structure {{\"thinking_progress\":\"YOUR THINKING and CONFORMATION\",\n\"is_original_example_good\":\"True/False\"\n\"refined_text\":\"CORRECTED Data Entry\"}}""",
    
    "math_eval": """I will give you a piece of text containing some mathematical information. It requires precise calculations to verify its correctness. Therefore, please translate it into a segment of Python code to represent the mathematical calculation process mentioned in the text, and then compute the final answer and direct print the answer number. Format your output in a JSON format with key 'Code' for the executable code and 'Analysis' to explain how you transfer the sample into code. The input sample is: {expression}.""",
    "math_eval_compare": "I will provide you with two answers, and I need you to help me determine whether these two answers are semantically equivalent. For example, '2' and 'two' are considered equivalent. If they are equivalent, please reply with 'True'. If they are not equivalent, reply with 'False'. Note that you should only reply with one word (either 'True' or 'False') and not include any other content. Here is two response: '{response1}', '{response2}'.",
    "feedback_prefix": "The following is human feedback on some of the generated samples and your generated samples need to refer to the suggestions in the human feedback: \n",
     'improve_examples_with_reflection': '''You are a professional dataset generation assistant.
Your task is to create improved versions of the original example based on the reflection and ,few shot examples.

DATASET DESCRIPTION:[[description]].

[[few_shot_examples]]

[[constraint]]

Ensure that the improvements address the identified weaknesses and enhance the strengths.


Original Data Entry:
[[original_example]]

Reflection:[[reflection]]


The structure and form of the improved example should be SAME with the original example; DO NOT make significant changes to the existing example. Directly output your improved Data Entry as the following JSON format:'''
})



eval_prompt_dict={
    "TruthfulQA":{
        "prompt": "You are a professional data annnotator.\n\nGiven a model generated answer and a list of groundtruth answer. Is the  model generated answer correct(same as the Groundtruth answer)? Only reply YES or NO. \nGroundtruth answer: [[correct_answer]] \nModel generated answer you should evaluate: [[answer]]",
        "mapping": {
            "[[correct_answer]]": "label",
            "[[answer]]": "res"}
    },
    "HellaSwag": {
        "prompt": "You are a professional data annnotator.\n\nGiven a model generated answer and a list of groundtruth answer. Is the  model generated answer correct(same as the Groundtruth answer)? Only reply YES or NO. \nGroundtruth answer: [[correct_answer]] \nModel generated answer you should evaluate: [[answer]]",
        "mapping": {
            "[[correct_answer]]": "label",
            "[[answer]]": "res"}
    },
    "HellaSwag2": {
        "prompt": "You are a professional data annnotator.\n\nGiven a model generated answer and a list of groundtruth answer. Is the  model generated answer correct(same as the Groundtruth answer)? Only reply YES or NO. \nGroundtruth answer: [[correct_answer]] \nModel generated answer you should evaluate: [[answer]]",
        "mapping": {
            "[[correct_answer]]": "label",
            "[[answer]]": "res"}
    },
    "MMLU": {
        "prompt": "You are a professional data annnotator.\n\nGiven a model generated answer and a list of groundtruth answer. Is the  model generated answer correct(same as the Groundtruth answer)? Only reply YES or NO. \nGroundtruth answer: [[correct_answer]] \nModel generated answer you should evaluate: [[answer]]",
        "mapping": {
            "[[correct_answer]]": "label",
            "[[answer]]": "res"}
    },
    "MMLU-constrain": {
        "prompt": "You are a professional data annnotator.\n\nGiven a question, your task is to determine whether the question is related to [[subject]]. Here is the question to evaluate: [[text]]\n\n Only reply YES or NO.",
        "mapping": {
            "[[text]]": "text",
            "[[subject]]": "subject"}
    },
    "MCQ": {
      "prompt": """ You are a professional data annotator. Your task is to compare a model-generated answer to the groundtruth (correct) answer for a given question.

    Instructions:
    1. Read the provided question.
    2. Identify and note the final answer generated by the model.
    3. Compare this model-generated answer with the groundtruth answer.
    4. Use the JSON format below to indicate whether the model's final answer matches the groundtruth answer.

    Details:
    - Question: [[question]]
    - Model generated answer: [[solution]]
    - Groundtruth answer: [[correct_answer]]

    Response Format:
    {
      "Model Final Answer": "<Extracted answer from model>",
      "Groundtruth Answer": "<Provided correct answer>",
      "is_same": true/false  
    }
    """,
        "mapping": {
            "[[correct_answer]]": "label",
            "[[question]]": "text",
            "[[solution]]": "res",}
    },
    "MCQ": {
      "prompt": """ You are a professional data annotator. Your task is to compare a model-generated answer to the groundtruth (correct) answer for a given question.

    Instructions:
    1. Read the provided question.
    2. Identify and note the final answer generated by the model.
    3. Compare this model-generated answer with the groundtruth answer.
    4. Use the JSON format below to indicate whether the model's final answer matches the groundtruth answer.

    Details:
    - Question: [[question]]
    - Model generated answer: [[solution]]
    - Groundtruth answer: [[correct_answer]]

    Response Format:
    {
      "Model Final Answer": "<Extracted answer from model>",
      "Groundtruth Answer": "<Provided correct answer>",
      "is_same": true/false  
    }
    """,
        "mapping": {
            "[[correct_answer]]": "label",
            "[[question]]": "text",
            "[[solution]]": "res",}
    },
    "gsm8k": {
      "prompt": """You are a professional data annnotator.\n\nGiven the groundtruth answer of a math problem and the model generated answer. Only evaluate if model's FINAL answer is the same with groundtruth answer.\n\nModel generated solution: [[solution]]\n\nGroundtruth answer: [[correct_answer]] \nYou should first extract the model's final answer.Then evaluate if the final answer provided by the model is correct. Reply in following JSON format: {"Model final Answer":"","Groundtruth answer":"","is_same":""}\n\n""",
        "mapping": {
            "[[correct_answer]]": "label",
            "[[solution]]": "res"}
    }
 } 
