from utils import embedding
from .embedding import get_embedding
from .file_process import save_json,load_json
import concurrent.futures
import numpy as np
import pandas as pd
import json,random,os
import os,random
from tqdm import tqdm
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Any
from collections import defaultdict

from configuration import ConfigManager
ConfigManager.load_config()
config = ConfigManager.get_config_dict()




def get_embedding_matrix(data_path):
    Embedder = embedding.EmbeddingProcessor(config=config)
    get_single_item_embedding=Embedder.get_single_item_embedding
    # Check if the path ends with "_embedding.json", directly load if it exists.
    if data_path.endswith("_embedding.json"):
        embeddings = load_json(data_path)
    else:
        save_path = data_path.replace(".json", "_embedding.json")
        data_gen = load_json(data_path)
        if os.path.exists(save_path):
            cached_embeddings = load_json(save_path)
            cached_ids = {item['text'] for item in cached_embeddings}
            new_data = [item for item in data_gen if item['text'] not in cached_ids]
            print(f"Using {len(cached_embeddings)} cached embeddings, generating {len(new_data)} new embeddings.")
            
            if new_data:
                with ThreadPoolExecutor(max_workers=8) as executor:
                    new_embeddings = list(tqdm(executor.map(get_single_item_embedding, new_data), total=len(new_data)))
                embeddings = cached_embeddings + new_embeddings
                save_json(embeddings, save_path)
            else:
                embeddings = cached_embeddings
        else:
            with ThreadPoolExecutor(max_workers=8) as executor:
                embeddings = list(tqdm(executor.map(get_single_item_embedding, data_gen), total=len(data_gen)))
            save_json(embeddings, save_path)
    
    df = pd.DataFrame(embeddings)
    matrix = np.array(df['embedding'].tolist())
    return matrix, df


def concat_constrain_generated_datasets(datasets,gen_model,base_path):
    for dataset_name, num_gen in [(d["dataset_name"], d["num_gen"]) for d in datasets]:
        folder_path=os.path.join(base_path,dataset_name)
        file_list=[file for file in os.listdir(folder_path) if file.endswith('.json') and file.startswith(f"{dataset_name}_{gen_model}_generated")]

        def load_json_data(file_path):
            with open(file_path, 'r') as file:
                json_data = json.load(file)
            return json_data

        file_list_updated=[]
        
            
        for file in file_list:
            start = file.find("(") + 1
            end = file.find(")")
            if start != -1 and end != -1:
                file_num = int(file[start:end])
                if file_num in num_gen:
                    file_list_updated.append(os.path.join(folder_path,file))
            if 0 in num_gen:
                if file.endswith('_generated.json'):
                    file_list_updated.append(os.path.join(folder_path,file))
        data = []
        constrain_dict = defaultdict(list)
        
        for file_path in file_list_updated:
            d = load_json_data(file_path)
            constrain_list = d['dataset_config']['dataset_configuration']['dataset_constraint']
            if constrain_list:
                constrain_tuple = tuple(tuple(item.items()) for item in constrain_list)
                constrain_dict[constrain_tuple].extend(d["dataset"])


        merged_data = []
        for constrain, datasets in constrain_dict.items():
            merged_dataset = {
                "constraint": [dict(item) for item in constrain],  
                "datasets": datasets  
            }
            merged_data.append(merged_dataset)

        return merged_data




def concat_generated_datasets(datasets,gen_model,base_path):
    for dataset_name, num_gen in [(d["dataset_name"], d["num_gen"]) for d in datasets]:
        folder_path=os.path.join(base_path,dataset_name)
        file_list=[file for file in os.listdir(folder_path) if file.endswith('.json') and file.startswith(f"{dataset_name}_{gen_model}_generated")]

        def load_json_data(file_path):
            with open(file_path, 'r') as file:
                json_data = json.load(file)
            return json_data

        file_list_updated=[]
        
            
        for file in file_list:
            # 提取文件名中的数字
            start = file.find("(") + 1
            end = file.find(")")
            if start != -1 and end != -1:
                file_num = int(file[start:end])
                if file_num in num_gen:
                    file_list_updated.append(os.path.join(folder_path,file))
            if 0 in num_gen:
                if file.endswith('_generated.json'):
                    file_list_updated.append(os.path.join(folder_path,file))


        data=[]
        #print(file_list_updated)
        for file_path in file_list_updated:
            d=load_json_data(file_path)
            data.extend(d["dataset"])



        # Modify each element in the dataset
        for i, item in enumerate(data, start=1):
            item["unique_id"] = i  # Add unique_id
            item["batch_id"] = item.pop("id")  # Rename "id" to "batch_id"
            item.pop("number")

        print(dataset_name,len(data))

        # Saving the modified json data to a new file for review
        #save_folder_path=f'/media/ssd/wtm/DyGenset/{gen_model}_gen/'
        modified_file_path = os.path.join(folder_path,f'{dataset_name}_concated_{gen_model}.json')
        with open(modified_file_path, 'w', encoding='utf-8') as file:
            json.dump(data, file, ensure_ascii=False, indent=4)



import numpy as np

def remove_similar_points(data,threshold):
    n_points = data.shape[0]  # 数据点的数量
    keep = np.ones(n_points, dtype=bool)  # 标记需要保留的点
    similar_pairs = []  # 用于记录相似的数据点对

    for i in range(n_points):
        for j in range(i + 1, n_points):
            if keep[i] and keep[j]:  # 只有当两点都未被标记去除时，才计算它们的距离
                dist = np.linalg.norm(data[i] - data[j])  # 计算点i和点j之间的欧式距离
                if dist < threshold:
                    similar_pairs.append((i, j))  # 记录相似的数据点对
                    keep[j] = False  # 标记其中一个点去除

    # 返回被标记为保留的点和所有被认为相似的点对
    return keep,similar_pairs



def remove_similar_points_by_cosine(data, threshold):
    n_points = data.shape[0]  # 数据点的数量
    keep = np.ones(n_points, dtype=bool)  # 标记需要保留的点
    similar_pairs = []  # 用于记录相似的数据点对

    # 归一化数据点，以确保每个向量的长度为1，这样只需计算点积即可得到余弦相似度
    normalized_data = data / np.linalg.norm(data, axis=1, keepdims=True)

    for i in range(n_points):
        for j in range(i + 1, n_points):
            if keep[i] and keep[j]:  # 只有当两点都未被标记去除时，才计算它们的相似度
                cosine_similarity = np.dot(normalized_data[i], normalized_data[j])
                if cosine_similarity > threshold:
                    similar_pairs.append((i, j))  # 记录相似的数据点对
                    keep[j] = False  # 标记其中一个点去除

    # 返回被标记为保留的点和所有被认为相似的点对
    return keep, similar_pairs


def remove_similar_file(file_path,threshold=0.3):
    matrix,df=get_embedding_matrix(file_path)
    keep, similar_pairs = remove_similar_points(matrix, threshold)
    print("Original data:\n", len(matrix))
    print("Keep data:\n", len(matrix[keep]))
    print("Similar pairs:\n", similar_pairs)
    print("\n\n")

    save_path=file_path.replace(".json","_filtered.json")
    filtered_df=df[keep]
    columns_to_save = filtered_df.columns.difference(['embedding'])
    #columns_to_save = filtered_df.columns
    filtered_df_to_save = filtered_df.loc[:, columns_to_save]
    filtered_df_to_save.to_json(save_path,orient='records',indent=4)
    return filtered_df_to_save


def filter_good(data):
    new_data = []
    i_values = [] 
    for i, el in enumerate(data):
        try:
            if 'isgood' in el.keys():
                if el['isgood']:
                    #print(i)
                    new_data.append(el)
                    i_values.append(i)  
            elif el["reflection_trajectory"][-1]["process"][-1]['isgood'] == 'yes':
                new_data.append(el)
                i_values.append(i)  
        except Exception as e:
            print('Error', e)
    return new_data, i_values