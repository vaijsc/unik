import abc
import json
from abc import ABC

import yaml
from pydantic.dataclasses import dataclass


class ConfigReaderInterface(abc.ABC):
    def __init__(self):
        super().__init__()

    def read_config_from_file(self, config_filename: str):
        raise NotImplementedError()


class JsonConfigReader(ConfigReaderInterface, ABC):
    def __init__(self):
        super(JsonConfigReader, self).__init__()

    def read_config_from_file(self, config_path: str):
        with open(config_path) as file:
            config = json.load(file)
        # config_object = Struct(**config)
        return config


class YamlConfigReader(ConfigReaderInterface):
    def __init__(self):
        super(YamlConfigReader, self).__init__()

    def read_config_from_file(self, config_path: str):
        with open(config_path) as file:
            config = yaml.safe_load(file)
        # config_object = Struct(**config)
        return config


@dataclass
class ConfigReaderInstance:
    json = JsonConfigReader()
    yaml = YamlConfigReader()


class Config:
    """Returns a config instance depending on the ENV_STATE variable."""

    def __init__(self, args=None):
        tmp_openai_conf = ConfigReaderInstance.yaml.read_config_from_file("configs/openai_config.yml")
        self.OPENAI_CONF = ConfigReaderInstance.yaml.read_config_from_file(tmp_openai_conf["secret_path"])

        tmp_db_conf = ConfigReaderInstance.yaml.read_config_from_file("configs/db_config.yml")
        self.DB_CONF = ConfigReaderInstance.yaml.read_config_from_file(tmp_db_conf["secret_path"])

        self.HF_CONF = ConfigReaderInstance.yaml.read_config_from_file("configs/hf_config.yml")
        self.DATA_CONF = ConfigReaderInstance.yaml.read_config_from_file("configs/data_config.yml")
        self.CONF = ConfigReaderInstance.yaml.read_config_from_file("configs/config.yml")

        # update params from args
        if args:
            self.update(args)

    def update(self, args):
        self.CONF.update(vars(args))

    def __repr__(self):
        return f"Configs\n{yaml.dump(self.CONF, indent=2,sort_keys=True)}"


cfg = Config()


def update_config(args=None):
    global cfg
    cfg.update(args)
    # print(vars(settings))
