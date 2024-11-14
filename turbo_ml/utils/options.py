import logging
from typing import Literal
import torch


class options:
    def __init__(self):
        self._print_logs: bool = True
        # TODO: change default to auto-adjusted based on terminal size before release
        self._text_size: int = 64
        self._validation: bool = True  # TODO: change default to False before release
        # TODO: change default to automatic detection
        self._device: Literal['cpu', 'cuda', 'mps', 'auto'] = 'auto'
        self._threads: int = -1
        self._dev_mode = True
        self._dev_mode_logging = logging.INFO
        self._user_mode_logging = logging.ERROR
        self._blacklist = ['CalibratedClassifierCV']
        self._hyperparameters_declaration_priority: Literal['sklearn',
                                                            'custom'] = 'custom'

    @property
    def print_logs(self):
        return self._print_logs

    @print_logs.setter
    def print_logs(self, value: bool):
        self._print_logs = value

    @property
    def text_size(self):
        return self._text_size

    @text_size.setter
    def text_size(self, value: int):
        self._text_size = value

    @property
    def validation(self):
        return self._validation

    @validation.setter
    def validation(self, value: bool):
        self._validation = value

    @property
    def device(self):
        if self._device == 'auto':
            if torch.cuda.is_available():
                self._device = 'cuda'
            elif torch.backends.mps.is_available():
                self._device = 'mps'
            else:
                self._device = 'cpu'
        return self._device

    @device.setter
    def device(self, value: Literal['cpu', 'cuda', 'mps', 'auto'] = 'auto'):
        self._device = value

    @property
    def threads(self):
        return self._threads

    @threads.setter
    def threads(self, value: int):
        self._threads = value

    @property
    def dev_mode(self):
        return self._dev_mode

    @dev_mode.setter
    def dev_mode(self, value: bool):
        self._dev_mode = value

    @property
    def dev_mode_logging(self):
        return self._dev_mode_logging

    @dev_mode_logging.setter
    def dev_mode_logging(self, value):
        self._dev_mode_logging = value

    @property
    def user_mode_logging(self):
        return self._user_mode_logging

    @user_mode_logging.setter
    def user_mode_logging(self, value):
        self._user_mode_logging = value

    @property
    def blacklist(self):
        return self._blacklist

    @blacklist.setter
    def blacklist(self, value: list):
        self._blacklist = value

    @property
    def hyperparameters_declaration_priority(self):
        return self._hyperparameters_declaration_priority

    @hyperparameters_declaration_priority.setter
    def hyperparameters_declaration_priority(self, value: Literal['sklearn', 'custom']):
        self._hyperparameters_declaration_priority = value


options = options()
