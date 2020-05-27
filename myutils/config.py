# -*- coding: utf-8 -*-

class Config(object):

    def __init__(self, config=None, **kwargs):
        if config is None:
            config = dict()
        assert isinstance(config, dict)
        for key, value in kwargs.items():
            config[key] = value

        self._config = config

    def has(self, key):
        return key in self._config

    def set(self, key, value):
        self._config[key] = value

    def get(self, key, default_value=None):
        if key in self._config:
            return self._config[key]

        return default_value

my_config = Config()