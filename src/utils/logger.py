# mypy: ignore-errors
"""
Logging utilities
"""
import functools
import logging
from time import gmtime, strftime, time

import rich.syntax
import rich.tree
from omegaconf import DictConfig, ListConfig, OmegaConf
from rich import print


class MyLogger:
    def __init__(self, name):
        self._logger = logging.getLogger(name)
        self._logger.setLevel(logging.INFO)
        self.stream_handler = logging.StreamHandler()
        formatter = logging.Formatter(
            fmt="%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s : %(message)s",
            datefmt="%Y-%m-%d %H:%M",
        )
        self.stream_handler.setFormatter(formatter)
        self._logger.addHandler(self.stream_handler)

    @property
    def logger(
        self,
    ):
        return self._logger


class Timer:
    def __init__(self, prefix=""):
        self.prefix = prefix
        # self.logger = logger
        # self.suffix = suffix

    def __call__(self, func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time()
            # self.logger.info(f"{self.prefix}")
            # self.logger.debug(f"Args: {kwargs}")
            try:
                rsp = func(*args, **kwargs)
                # self.logger.debug(f"Response: {rsp}")
                print(f"{self.prefix} Used " + strftime("%H:%M:%S", gmtime(time() - start_time)))
                return rsp
            except Exception as e:
                # self.logger.error(repr(e))
                raise e

        return wrapper
