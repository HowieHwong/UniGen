import os
import numpy as np
import pandas as pd
import concurrent.futures
from tqdm import tqdm
from sklearn.metrics.pairwise import euclidean_distances
import embedding
from file_process import load_json, save_json
from configuration import ConfigManager
ConfigManager.load_config()
config = ConfigManager.get_config_dict()


def get_single_item_embedding(item):
    item["embedding"] = embedding.EmbeddingProcessor.get_embedding(item['text'], config)
    return item


def get_embedding_matrix(data_path):
    if data_path.endswith("_embedding.json"):
        embeddings = load_json(data_path)
    else:
        save_path = data_path.replace(".json", "_embedding.json")
        if os.path.exists(save_path):
            embeddings = load_json(save_path)
        else:
            data_gen = load_json(data_path)
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                embeddings = list(tqdm(executor.map(get_single_item_embedding, data_gen), total=len(data_gen)))
            save_json(embeddings, save_path)

    df = pd.DataFrame(embeddings)
    matrix = np.array(df['embedding'].tolist())
    return matrix, df


def compute_remote_clique(matrix):
    pairwise_distances = euclidean_distances(matrix)
    n = matrix.shape[0]
    remote_clique_score = (1 / n**2) * pairwise_distances.sum()
    return remote_clique_score


def main():
    file_names = [
        "data/sample_1.json",
        "data/sample_2.json",
        "data/sample_3.json",
        "data/sample_4.json",
        "data/sample_5.json",
        "data/sample_6.json",
        "data/sample_7.json",
        "data/sample_8.json",
        "data/sample_9.json",
        "data/sample_10.json",
    ]

    for file_path in file_names:
        matrix, df = get_embedding_matrix(file_path)
        remote_clique_score = compute_remote_clique(matrix)
        print(f"Remote-Clique score for {os.path.basename(file_path)}: {remote_clique_score}")


if __name__ == "__main__":
    main()
