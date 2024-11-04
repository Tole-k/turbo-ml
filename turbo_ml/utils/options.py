import logging
from typing import Literal
import torch


class options:
    print_logs: bool = True
    # TODO: change default to auto-adjustem based on terminal size before release
    text_size: int = 64
    validation: bool = True  # TODO: change default to False before release
    # TODO: change default to automatic detection
    device: Literal['cpu', 'cuda', 'mps', 'auto'] = 'auto'
    threads: int = -1
    dev_mode = True
    dev_mode_logging = logging.INFO
    user_mode_logging = logging.ERROR
    blacklist = ['CalibratedClassifierCV']
    hyperparameters_declaration_priority: Literal['sklearn',
                                                  'custom'] = 'custom'

    def get_device(self):
        if self.device == 'auto':
            if torch.cuda.is_available():
                self.device = 'cuda'
            elif torch.backends.mps.is_available():
                self.device = 'mps'
            else:
                self.device = 'cpu'
        return self.device

    def set_device(self, device: Literal['cpu', 'cuda', 'mps', 'auto'] = 'auto'):
        self.device = device

    def get_threads(self):
        return self.threads

    def set_threads(self, threads: int):
        self.threads = threads
