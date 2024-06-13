    
    
import os
import os
import csv
from utils.eval_utils import eval_single
from utils.eval_utils import *
from utils.configuration import ConfigManager

def judge(config):
    
    # # 加载配置文件
    # ConfigManager.load_config()
    # config = ConfigManager.get_config_dict()
    results =[]
    # 从配置文件中读取信息
    models = config['models']
    base_dir = config['base_dir']
    tasks_files = config['tasks_files']

    # 运行评估函数
    for model in models:
        for task, filenames in tasks_files.items():
            for filename in filenames:
                result = eval_single(model, task, filename, base_dir)
                results.append({
                    'model': model,
                    'task': task,
                    'filename': filename,
                    'result': result
                })

    # 保存结果到 CSV 文件
    csv_file = os.path.join(base_dir, 'evaluation_results.csv')
    with open(csv_file, 'w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=['model', 'task', 'filename', 'result'])
        writer.writeheader()
        for row in results:
            writer.writerow(row)

    print(f"Results saved to {csv_file}")