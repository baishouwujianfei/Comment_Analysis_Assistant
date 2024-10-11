import yaml  
from pathlib import Path

# 获取当前脚本所在的绝对路径
current_dir = Path(__file__).resolve().parent.parent

    
config_file = current_dir / 'config.yaml'

class GetConfig:  
    """  
    一个用于获取和管理配置的单例类。  
      
    该类负责从YAML配置文件中加载配置，并允许通过命令行参数覆盖配置文件中的顶级设置。  
    """  
    _instance = None  # 单例的实例变量  
  
    def __new__(cls, *args, **kwargs):  
        """  
        单例模式的实现，确保只有一个实例存在。  
        """  
        if cls._instance is None:  
            cls._instance = super(GetConfig, cls).__new__(cls)  
            cls._instance._config = {}  # 初始化为空字典  
        return cls._instance  
  
    def initialize(self, args = None):  
        """  
        初始化配置。  
          
        从配置文件中加载配置，并允许通过命令行参数覆盖顶级设置。  
        :param args: 包含命令行参数的argparse Namespace对象，应具有config_file属性。  
        """  
        with open(config_file, "r", encoding="utf-8") as f:  
            config = yaml.safe_load(f)  # 从配置文件中安全加载配置  
  
        # # 提取命令行参数中需要覆盖的配置项（仅限于顶级键）  
        # overridden_keys = {k for k in vars(args) if k in config and getattr(args, k) is not None}  
        # for key in overridden_keys:  
        #     config[key] = getattr(args, key)  
  
        # 存储配置  
        self._config = config  
  
    def __getattr__(self, name):  
        """  
        动态获取配置属性，支持多层级的字典访问。  
          
        使用点符号访问多层级的配置，如 config.server.host。  
        :param name: 使用点符号的路径，如 'server.host'。  
        :return: 属性值或引发 AttributeError。  
        :raises AttributeError: 如果属性路径不存在。  
        """  
        parts = name.split('.')  
        current = self._config  
        for part in parts:  
            if part not in current:  
                raise AttributeError(f"'{name}' not found in configuration")  
            current = current[part]  
        return current