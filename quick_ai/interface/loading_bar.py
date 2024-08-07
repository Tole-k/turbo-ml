from ..utils import option
import sys


class Bar:
    def __init__(self, max_tick=100, with_header=True):
        self._tick = 0
        self.max_tick = max_tick
        width = option.text_size
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
        width = option.text_size
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
