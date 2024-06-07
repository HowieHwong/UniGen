# config.py
import os
import json

class ConfigManager:
    # 初始时，配置为空
    _config = None

    @staticmethod
    def load_config():
        if ConfigManager._config is None:
            current_file_path = os.path.abspath(__file__)
            current_dir_path = os.path.dirname(current_file_path)
            config_path = os.path.join(current_dir_path, "../config.json")

            with open(config_path, "r") as file:
                ConfigManager._config = json.load(file)

    @classmethod
    def get_config_dict(cls, ):
        # 确保配置已加载
        if cls._config is None:
            cls.load_config()
        return cls._config
        
    @classmethod
    def set_config_dict(cls, dict):
        cls._config=dict
        # # 确保配置已加载
        # if cls._config is None:
        #     cls.load_config()
        # return cls._config






