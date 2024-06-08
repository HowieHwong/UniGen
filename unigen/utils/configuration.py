# configuration.py
import os
import yaml

class ConfigManager:
    _config = None

    @staticmethod
    def load_config():
        if ConfigManager._config is None:
            current_file_path = os.path.abspath(__file__)
            current_dir_path = os.path.dirname(current_file_path)
            config_path = os.path.join(current_dir_path, "../config.yaml")

            with open(config_path, "r") as file:
                ConfigManager._config = yaml.safe_load(file)

    @classmethod
    def get_config_dict(cls):
        if cls._config is None:
            cls.load_config()
        return cls._config
        
    @classmethod
    def set_config_dict(cls, dict):
        cls._config = dict

# 示例使用
if __name__ == "__main__":
    ConfigManager.load_config()
    config = ConfigManager.get_config_dict()
    
    # 打印配置以验证读取是否成功
    print(config)




