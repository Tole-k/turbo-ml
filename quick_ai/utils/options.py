import logging

class option:
    print_logs: bool = True  # TODO: change default to False before release
    text_size: int = 64 # TODO: change default to auto-adjustem based on terminal size before release
    validation: bool = True  # TODO: change default to False before release
    log_level = logging.WARNING
