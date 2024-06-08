import wikipediaapi
from . import data_format
from .data_format import prompt_format
from .IO import print, input
import json
from .prompt import prompt_template
from utils import file_process,data_format
from utils.IO import print,input
import wikipedia
import concurrent.futures



def _extract_keywords(input_text):
    prompt = prompt_template['wiki_keyword_extract'].format(input_text=input_text)
    
    response_json=data_format.get_res_data(prompt)
    extracted_keywords = response_json.get("entities", "Unknown")
    assert isinstance(extracted_keywords, list)
    print("Extracted keywords: " + str(extracted_keywords),"BLUE")
    return extracted_keywords


def _refine_content(input_text, keyword_content_pairs):
    combined_content = "\n".join(
        [f"Keyword: {keyword}\nContent: {content}" for keyword, content in keyword_content_pairs])
    prompt = prompt_template['wiki_fact_refine'].format(input_text=input_text, wiki_data=combined_content)

    response_json=data_format.get_res_data(prompt)
    return response_json



def _get_wiki_content(keyword):
    """
    Retrieves the content of a Wikipedia page for a given keyword.

    :param keyword: The keyword to search on Wikipedia.
    :return: The content of the Wikipedia page.
    """
    try:
        wiki = wikipediaapi.Wikipedia('Bias/0.0 (tangjingyu0621@gmail.com)','en')
        page = wiki.page(keyword)
        return page.text
    except Exception as e:
        return str(e)
    

        
def wiki_check(input_text):
    keywords = _extract_keywords(input_text)
    keyword_content_pairs = []
    def fetch_content(keyword):
        content = _get_wiki_content(keyword)
        return (keyword, content)

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(fetch_content, keyword) for keyword in keywords]
        for future in concurrent.futures.as_completed(futures):
            keyword_content_pairs.append(future.result())

    refined_content = {
        'keyword_content_pairs': keyword_content_pairs,
        'refined_content': _refine_content(input_text, keyword_content_pairs)
    }
    return refined_content


