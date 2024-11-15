import logging
from typing import Literal
import torch


class options:
    def __init__(self):
        self.print_logs: bool = True
        # TODO: change default to auto-adjusted based on terminal size before release
        self.text_size: int = 64
        self.validation: bool = True  # TODO: change default to False before release
        self._device: Literal['cpu', 'cuda', 'mps', 'auto'] = 'auto'
        self.threads: int = -1
        self.dev_mode = True
        self.dev_mode_logging = logging.INFO
        self.user_mode_logging = logging.ERROR
        self.blacklist = ['CalibratedClassifierCV']
        self.hyperparameters_declaration_priority: Literal['sklearn',
                                                           'custom'] = 'custom'
        self.hpo_trials = 10

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
    def device(self, value: Literal['cpu', 'cuda', 'mps', 'auto']):
        self._device = value


options = options()
