import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics.pairwise import cosine_similarity
from itertools import combinations
from openai import OpenAI,AzureOpenAI
import random
from utils import file_process
import concurrent.futures


def get_embedding(string, config, azure=False):
    
    if azure:
        azure_endpoint = config["generation_settings"]["azure_base_url"]
        api_key = config['generation_settings']['openai_azure_api']
        api_version = config["generation_settings"]["azure_version"]
        client = AzureOpenAI(
            azure_endpoint=azure_endpoint,
            api_key=api_key,
            api_version=api_version,
        )
        response = client.embeddings.create(
            model=config["generation_settings"]["azure_embedding_engine"],
            input=string
        )
        return response.data[0].embedding
    else:
        embedding_model= config['generation_settings']['embedding_model']
        api_key = config['generation_settings']['openai_api']
        client = OpenAI(api_key=api_key,
                        #base_url=base_url
                        )
        response = client.embeddings.create(
            model=embedding_model,
            input=string
        )
        return response.data[0].embedding



def get_single_item_embedding(item,embedding_model):
    item["embedding"]=get_embedding(item["text"],embedding_model)
    return item

def select_embeddings_with_auto_min_similarity(embeddings, n, embedding_key='embedding', start_similarity=0.0,decrement=0.05):
    embeddings_array = np.array([el[embedding_key] for el in embeddings])
    similarity_matrix = cosine_similarity(embeddings_array)

    # 初始化min_similarity
    min_similarity = start_similarity

    while min_similarity <1:
        selected_indices = [np.random.choice(range(len(embeddings)))]  # 随机选择一个起始嵌入向量
        available_indices = set(range(len(embeddings))) - set(selected_indices)

        while len(selected_indices) < n and available_indices:
            # 计算未选择的嵌入向量与已选择嵌入向量组的最小相似度
            min_similarities = similarity_matrix[selected_indices, :][:, list(available_indices)].min(axis=0)

            # 找出满足当前最小相似度阈值的嵌入向量
            candidates = [i for i, sim in zip(available_indices, min_similarities) if sim <= min_similarity]

            if not candidates:
                break  # 如果没有满足当前相似度阈值的候选嵌入向量，跳出内循环

            # 从候选者中随机选择一个并更新索引列表
            new_selected = np.random.choice(candidates)
            selected_indices.append(new_selected)
            available_indices.remove(new_selected)

        if len(selected_indices) == n:
            # 如果已经找到足够数量的嵌入向量，返回结果
            return selected_indices
            return [embeddings[idx] for idx in selected_indices]

        # 降低min_similarity阈值，尝试更宽松的条件
        min_similarity += decrement

    # 如果循环结束仍未找到足够的嵌入向量，返回空列表或抛出错误
    raise ValueError("Unable to find enough embeddings with any minimum similarity threshold.")

def generate_dataset_embedding(data,embedding_model):
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        embeddings = list(filter(None, executor.map(get_single_item_embedding, data,embedding_model)))
    return embeddings


def cluster_embeddings(data,embeddings, num_clusters, method='cosine_similarity', embedding_key='embedding', text_key='text'):

    
    embeddings_array = np.array([el[embedding_key] for el in embeddings])
    assert method in ['kmeans', 'agglomerative', 'dbscan',"cosine_similarity"]
    clustering_model = None
    if method == 'kmeans':
        clustering_model = KMeans(n_clusters=num_clusters)
    elif method == 'agglomerative':
        clustering_model = AgglomerativeClustering(n_clusters=num_clusters)
    elif method == 'cosine_similarity':
        selected_indices = select_embeddings_with_auto_min_similarity(embeddings, n=num_clusters)
        #print(selected_indices)
        
        return_examples = [data[idx] for idx in selected_indices]
        return return_examples
    # elif method == 'dbscan':
    #     clustering_model = DBSCAN()

    labels = clustering_model.fit_predict(embeddings_array)
    cluster_embeddings = {cluster_label: [] for cluster_label in np.unique(labels)}

    for i, label in enumerate(labels):
        cluster_embeddings[label].append(embeddings[i])

    random_embeddings = {label: random.choice(cluster) for label, cluster in cluster_embeddings.items()}

    selected_indices = [embeddings.index(random_embeddings[label]) for label in random_embeddings]

    return_embeddings = [embeddings[idx][text_key] for idx in selected_indices]

    return return_embeddings
