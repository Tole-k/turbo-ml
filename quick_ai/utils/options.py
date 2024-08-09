import logging


class option:
    print_logs: bool = True  # TODO: change default to False before release
    text_size: int = 40
    validation: bool = True  # TODO: change default to False before release
    log_level = logging.INFO
    # TODO: change default to auto-adjustem based on terminal size before release
