from ..utils import option
import sys
import textwrap


class Box:
    def __init__(self, params: dict, topic: str = 'Training') -> None:
        self.params = params.copy()
        self.topic = topic

    def print(self, with_header: bool = True) -> None:
        mes = '\n'.join(f'{i}: {j}'for i, j in self.params.items())
        self.num_lines = print_in_box(mes, with_header=with_header)

    def update(self, params: dict, additional_backsteps: int = 0) -> None:
        self.params = params.copy()
        sys.stdout.write(
            "\033[F" * (self.num_lines + 1 + additional_backsteps))
        self.print(with_header=False)
        sys.stdout.write('\n'*additional_backsteps)


def print_in_box(message: str, topic: str = 'Training', color_id: int = 92, max_width=40, with_header: bool = True) -> int:
    max_width = option.text_size
    if with_header:
        print('+' + '-' * (max_width) + '+')
        print(f'|\033[{color_id}m{' ' * ((max_width // 2) - len(topic)//2)
                                  }{topic}{' ' * (max_width - (max_width//2) - len(topic)//2)}\033[0m|')
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
