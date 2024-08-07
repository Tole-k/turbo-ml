import sys
from collections import defaultdict
from .box import print_in_box, Box
from .tutorial import *

LOGO = """
  ____       _     __    ___   ____
 / __ \\__ __(_)___/ /__ / _ | /  _/
/ /_/ / // / / __/  '_// __ |_/ /  
\\___\\_\\_,_/_/\\__/_/\\_\\/_/ |_/___/ 
"""

LOGO_COLOR = """\033[32m
  ____       _     __    \033[33m___   ____\033[32m
 / __ \\__ __(_)___/ /__ \033[33m/ _ | /  _/\033[32m
/ /_/ / // / / __/  '_/\033[33m/ __ |_/ /\033[32m
\\___\\_\\_,_/_/\\__/_/\\_\\\033[33m/_/ |_/___/\033[0m
"""

WELCOME_MESSAGE = """
Welcome in Quick AI, this piece of software was created in order to make machine learning simple.
Given some dataset algorithms from library should be able to find the best machine learning algorithm with optimal parameters to solve problem provided in dataset whether this is classification or regression task.

## Guide ##
- load your dataset
- call quick_ai.train(dataset) to train model
- call quick_ai.predict(data) to find predictions
"""

RESPONSES = defaultdict(lambda: None, {
    5: "Need a helping hand with ... writing single letter?",
    6: "Yep, looks like you need it :/",
    7: 'Hey! Stop it! Just choose one option',
    8: "Downloading multiple viruses, please wait",
    9: "Ehh, life, I don't have too many responses left so just choose some option",
    10: "Like I said, just choose some option!",
    12: "I though that most of people would lose on the last one",
    13: "Okey, okey, you won, I give up, that's all I had"
})

CREDITS = """
For now not sure who to credit, there are only two of us.
"""

__DATASET_PATH: str = ''


def _choose_response(counter: int, default: str):
    if counter == 14:
        quit(0)
    if RESPONSES[counter] is None:
        return default
    return RESPONSES[counter]


def choose_option():
    def ask(counter: int):
        response = _choose_response(
            counter, "Choose option: 'number', quit: 'q': ")
        choice = input(response)
        match choice.lower():
            case '1':
                sys.stdout.write('\033[F'*(box.num_lines+5))
                sys.stdout.write(f'\n{' '*66}'*(box.num_lines+5))
                sys.stdout.write('\033[F'*(box.num_lines+5))
                tutorial()
            case '2':
                sys.stdout.write('\033[F'*(box.num_lines+5))
                sys.stdout.write(f'\n{' '*66}'*(box.num_lines+5))
                sys.stdout.write('\033[F'*(box.num_lines+5))
                load_dataset()
            case '3':
                sys.stdout.write('\033[F'*(box.num_lines+5))
                sys.stdout.write(f'\n{' '*66}'*(box.num_lines+5))
                sys.stdout.write('\033[F'*(box.num_lines+5))
                show_credits()
            case 'q': quit(0)
            case _:
                sys.stdout.write('\033[F')
                print(' '*100)
                sys.stdout.write('\033[F')
                ask(counter+1)

    options: dict = {'tutorial': 1,
                     'load dataset': 2,
                     'credits': 3}
    box = Box(options, topic='Choose option ')
    box.print()
    ask(0)


def welcome():
    print(LOGO)

    def ask(counter: int = 0):
        response = _choose_response(
            counter, "To continue type 'c', to quit type 'q': ")
        choice = input(response)
        match choice.lower():
            case 'c':
                sys.stdout.write('\033[F'*(res+5))
                sys.stdout.write(f'\n{' '*66}'*(res+5))
                sys.stdout.write('\033[F'*(res+5))
                choose_option()
            case 'q': quit(0)
            case _:
                sys.stdout.write('\033[F')
                ask(counter+1)
    res = print_in_box(WELCOME_MESSAGE, topic='Welcome in Quick AI ')
    ask(0)
    print(f'Now there should be calculations for dataset in file {
          __DATASET_PATH}')  # TODO: Provide calculations for dataset in this file


def load_dataset():
    print_in_box('Provide path to dataset file', topic='Load dataset ')
    global __DATASET_PATH
    __DATASET_PATH = input('Path: ')
    print(__DATASET_PATH)
    return


def show_credits():
    print_in_box(CREDITS + '\n' + LOGO, topic='Credits ')
    quit(0)


def tutorial():
    def show_tutorial(tutorial_number: int):
        def ask(counter: int = 0):
            response = _choose_response(
                counter, "continue: 'c', menu: 'm', quit 'q': ")
            choice = input(response)

            match choice.lower():
                case 'q': quit(0)
                case 'm':
                    sys.stdout.write('\033[F'*(size+5))
                    sys.stdout.write(f'\n{' '*66}'*(size+5))
                    sys.stdout.write('\033[F'*(size+5))
                    tutorial()
                case 'c':
                    sys.stdout.write('\033[F'*(size+5))
                    sys.stdout.write(f'\n{' '*66}'*(size+5))
                    sys.stdout.write('\033[F'*(size+5))
                    if tutorial_number+1 in TUTORIALS:
                        show_tutorial(tutorial_number+1)
                    else:
                        tutorial()
                case _:
                    sys.stdout.write('\033[F')
                    print(' '*100)
                    sys.stdout.write('\033[F')
                    ask(counter+1)

        tutorial_name = TUTORIALS.get(tutorial_number)
        tutorial_text = globals().get(f'TUTORIAL_{tutorial_number}')

        size = print_in_box(tutorial_text, topic=tutorial_name)
        ask(0)

    def ask(counter: int = 0):
        response = _choose_response(
            counter, "tutorial: 'number', menu: 'm', quit 'q': ")
        choice = input(response)

        if choice.lower() == 'q':
            quit(0)

        if choice.lower() == 'm':
            sys.stdout.write('\033[F'*(box.num_lines+5))
            sys.stdout.write(f'\n{' '*66}'*(box.num_lines+5))
            sys.stdout.write('\033[F'*(box.num_lines+5))
            choose_option()

        if not choice.isdigit():
            sys.stdout.write('\033[F')
            print(' '*100)
            sys.stdout.write('\033[F')
            ask(counter+1)

        choice = int(choice)
        if choice in TUTORIALS:
            sys.stdout.write('\033[F'*(box.num_lines+5))
            sys.stdout.write(f'\n{' '*66}'*(box.num_lines+5))
            sys.stdout.write('\033[F'*(box.num_lines+5))
            show_tutorial(choice)

    box = Box(TUTORIALS, topic='Choose tutorial ')
    box.print()
    ask()


if __name__ == '__main__':
    welcome()
