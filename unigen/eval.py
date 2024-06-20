import os
import os
import csv
from .utils.eval_utils import *


def judge(config):
    results =[]
    models = config['models']
    base_dir = config['data_path']
    tasks_files = config['tasks_files']

    for model in models:
        for task, filename in tasks_files.items():

            result = eval_single(model, task, filename, base_dir)
            results.append({
                'model': model,
                'task': task,
                'filename': filename,
                'result': result
            })
    csv_file = os.path.join(base_dir,'evaluation_results.csv')
    with open(csv_file, 'w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=['model', 'task', 'filename', 'result'])
        writer.writeheader()
        for row in results:
            writer.writerow(row)

    print(f"Results saved to {csv_file}")