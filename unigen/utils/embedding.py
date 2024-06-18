import os
import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity
import random
import concurrent.futures
from openai import OpenAI, AzureOpenAI
from .file_process import save_json,load_json

class EmbeddingProcessor:
    def __init__(self, config):
        self.config = config
        print(config)

    def preprocess_original_dataset(self,):
        original_dataset_path=self.config['generation_hint']['original_dataset_path']
        base_path = os.path.splitext(original_dataset_path)[0]
        embedding_path = f"{base_path}_dataset_embedding.json"
        if not os.path.exists(embedding_path):
            data=load_json(original_dataset_path)
            embeddings = self.generate_dataset_embedding(data)
            save_json(embeddings, embedding_path)
        else:
            embeddings = load_json(embedding_path)
        return embeddings
            
    
    
    def get_embedding(string, config):
        settings = config["api_settings"]
        azure = settings['use_azure']
        if azure:
            client = AzureOpenAI(
                azure_endpoint=settings["api_base"],
                api_key=settings['azure_api_key'],
                api_version=settings["azure_version"],
            )
            response = client.embeddings.create(
                model=settings["embedding_model"],
                input=string
            )
        else:
            client = OpenAI(api_key=settings['openai_api_key'])
            response = client.embeddings.create(
                model=settings['embedding_model'],
                input=string
            )
        return response.data[0].embedding

    def get_single_item_embedding(self, item):
        item["embedding"] = EmbeddingProcessor.get_embedding(item["text"],self.config)
        return item
    
    
    def generate_dataset_embedding(self, data):
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            embeddings = list(filter(None, executor.map(self.get_single_item_embedding, data)))
        return embeddings

    def select_embeddings_with_auto_min_similarity(self, embeddings, n, start_similarity=0.0, decrement=0.05):
        embeddings_array = np.array([el['embedding'] for el in embeddings])
        similarity_matrix = cosine_similarity(embeddings_array)
        min_similarity = start_similarity

        while min_similarity < 1:
            selected_indices = [np.random.choice(range(len(embeddings)))]
            available_indices = set(range(len(embeddings))) - set(selected_indices)

            while len(selected_indices) < n and available_indices:
                min_similarities = similarity_matrix[selected_indices, :][:, list(available_indices)].min(axis=0)
                candidates = [i for i, sim in zip(available_indices, min_similarities) if sim <= min_similarity]
                if not candidates:
                    break
                new_selected = np.random.choice(candidates)
                selected_indices.append(new_selected)
                available_indices.remove(new_selected)

            if len(selected_indices) == n:
                return selected_indices
            min_similarity += decrement

        raise ValueError("Unable to find enough embeddings with any minimum similarity threshold.")



    def cluster_embeddings(self, embeddings, num_clusters, method='cosine_similarity'):
        embeddings_array = np.array([el['embedding'] for el in embeddings])
        assert method in ['kmeans', 'agglomerative', 'cosine_similarity']
        clustering_model = None
        if method == 'kmeans':
            clustering_model = KMeans(n_clusters=num_clusters)
        elif method is 'agglomerative':
            clustering_model = AgglomerativeClustering(n_clusters=num_clusters)
        elif method == 'cosine_similarity':
            selected_indices = self.select_embeddings_with_auto_min_similarity(embeddings, num_clusters)
            #print(selected_indices)
            return [embeddings[idx] for idx in selected_indices]

        labels = clustering_model.fit_predict(embeddings_array)
        cluster_embeddings = {cluster_label: [] for cluster_label in np.unique(labels)}

        for i, label in enumerate(labels):
            cluster_embeddings[label].append(embeddings[i])

        random_embeddings = {label: random.choice(cluster) for label, cluster in cluster_embeddings.items()}
        selected_indices = [embeddings.index(random_embeddings[label]) for label in random_embeddings]

        return [embeddings[idx]['text'] for idx in selected_indices]
