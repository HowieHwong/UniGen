import os
import yaml
import json

class ConfigManager:
    _config = None
    _description = None

    @staticmethod
    def load_config():
        if ConfigManager._config is None:
            current_file_path = os.path.abspath(__file__)
            current_dir_path = os.path.dirname(current_file_path)
            config_path = os.path.join(current_dir_path, "../config.yaml")

            with open(config_path, "r") as file:
                ConfigManager._config = yaml.safe_load(file)
                
    @staticmethod
    def load_description():
        if ConfigManager._description is None:
            current_file_path = os.path.abspath(__file__)
            current_dir_path = os.path.dirname(current_file_path)
            description_path = os.path.join(current_dir_path, "../dataset/description.json")

            with open(description_path, "r", encoding='utf-8') as file:
                ConfigManager._description = json.load(file)

    @classmethod
    def get_config_dict(cls):
        if cls._config is None:
            cls.load_config()
        return cls._config

    @classmethod
    def get_description(cls):
        if cls._description is None:
            cls.load_description()
        return cls._description
        
    @classmethod
    def set_config_dict(cls, config_dict):
        cls._config = config_dict

    @classmethod
    def set_description(cls, description_dict):
        cls._description = description_dict

# 示例使用
if __name__ == "__main__":
    ConfigManager.load_config()
    config = ConfigManager.get_config_dict()
    
    # 加载并打印 description.json 的内容
    ConfigManager.load_description()
    description = ConfigManager.get_description()
    
    # 打印配置和描述以验证读取是否成功
    print("Config:", config)
    print("Description:", description)
    
    