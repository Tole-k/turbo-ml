import logging
from typing import Literal


class option:
    print_logs: bool = True  # TODO: change default to False before release
    # TODO: change default to auto-adjustem based on terminal size before release
    text_size: int = 64
    validation: bool = True  # TODO: change default to False before release
    log_level = logging.WARNING
    blacklist = ['CalibratedClassifierCV']
    hyperparameters_declaration_priority: Literal['sklearn',
                                                  'custom'] = 'custom'
