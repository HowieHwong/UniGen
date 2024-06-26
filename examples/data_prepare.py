import json
import os
from datetime import datetime

def concat_json_files(input_directory, output_directory, filter_good=True):
    data_list = []

    # 遍历目录中的所有JSON文件
    for filename in os.listdir(input_directory):
        if filename.endswith(".json"):
            file_path = os.path.join(input_directory, filename)
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if "dataset" in data:
                    dataset = data["dataset"]

                    # 根据 filter_good 参数决定是否过滤 isgood 字段
                    if filter_good:
                        dataset = list(filter(lambda item: item.get('isgood', False), dataset))
                    
                    # 删除每个数据项中的指定键
                    for item in dataset:
                        item.pop("reflection_epoch", None)
                        item.pop("reflection_trajectory", None)

                    data_list.extend(dataset)

    # 生成文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_directory, f"combined_dataset_{timestamp}.json")

    # 将合并后的数据写入输出文件
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data_list, f, ensure_ascii=False, indent=4)
    
    print(f"Data combined and saved to {output_file}")

if __name__ == "__main__":
    input_directory = "/Users/admin/Documents/GitHub/UniGen/generated_data/"  
    output_directory = "/Users/admin/Documents/GitHub/UniGen/examples" 
    filter_good = True  
    concat_json_files(input_directory, output_directory, filter_good)