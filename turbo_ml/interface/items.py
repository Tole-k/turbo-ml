"""
Module containing helpful pre-defined items for the TurboML interface.
"""
from turbo_ml.utils import options
import sys
import textwrap


class Box:
    def __init__(self, params: dict, topic: str = 'Training') -> None:
        self.params = params.copy()
        self.topic = topic

    def print(self, with_header: bool = True) -> None:
        mes = '\n'.join(f'{i}: {j}'for i, j in self.params.items())
        self.num_lines = print_in_box(
            mes, topic=self.topic, with_header=with_header)

    def update(self, params: dict, additional_backsteps: int = 0) -> None:
        self.params = params.copy()
        sys.stdout.write(
            "\033[F" * (self.num_lines + 1 + additional_backsteps))
        self.print(with_header=False)
        sys.stdout.write('\n'*additional_backsteps)


def print_in_box(message: str, topic: str = 'Training', color_id: int = 92, max_width=40, with_header: bool = True) -> int:
    max_width = options.text_size
    if with_header:
        print('+' + '-' * (max_width) + '+')
        # Formatter is driving me crazy
        print(f"""|\033[{color_id}m{' ' * ((max_width // 2) - len(topic)//2)}{topic +
                                                                              ''}{' ' * (max_width - (max_width//2) - len(topic)//2)}\033[0m|""")
        print('+' + '-' * (max_width) + '+')

    line_counter = 0
    for msg in message.split('\n'):
        wrapped_message = textwrap.wrap(
            msg, max_width-2, break_long_words=False, break_on_hyphens=False)

        for line in wrapped_message:
            print('| ' + line.ljust(max_width-2) + ' |')
            line_counter += 1

    print('+' + '-' * max_width + '+')
    return line_counter


class Bar:
    def __init__(self, max_tick=100, with_header=True):
        self._tick = 0
        self.max_tick = max_tick
        width = options.text_size
        if with_header:
            print('+' + '-' * width + '+')
        print('|[' + ' ' * (width-2) + ']|')
        print('+' + '-' * width + '+')

    def tick(self):
        tick = self._tick + 1
        if tick >= self.max_tick:
            self.update(self.max_tick, 'done')
        else:
            self.update(self._tick + 1, 'loading')

    def update(self, tick, message=None):
        width = options.text_size
        self._tick = tick
        sys.stdout.write("\033[F")
        sys.stdout.write("\033[F")

        bar = int(tick / self.max_tick * (width-2))

        match message:
            case 'error':
                sys.stdout.write('|[\033[91m' + '#' * bar +
                                 ' ' * (width - bar-2) + '\033[0m]|\n')
            case 'loading':
                sys.stdout.write('|[\033[93m' + '#' * bar +
                                 ' ' * (width - bar-2) + '\033[0m]|\n')
            case 'done':
                sys.stdout.write('|[\033[92m' + '#' * bar +
                                 ' ' * (width - bar-2) + '\033[0m]|\n')
            case _:
                sys.stdout.write('|[' + '#' * bar +
                                 ' ' * (width - bar-2) + ']|\n')
        print('+' + '-' * (width) + '+')

    def error(self):
        self.update(self._tick, 'error')

    def done(self):
        self.update(self.max_tick, 'done')
