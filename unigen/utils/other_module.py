# other_module.py
from utils.config import ConfigManager


if __name__=='__main__':
    # 获取配置
    config = ConfigManager.get_config_dict()
    print(config,'\n\n')
    # 打印配置以验证读取是否成功
    print(config)