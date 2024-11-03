import logging
from typing import Literal


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
