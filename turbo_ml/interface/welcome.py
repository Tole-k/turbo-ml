""" 
Main module implementing interactive command line interface for Turbo-ML library.
"""
import sys
from collections import defaultdict
from turbo_ml.interface.items import print_in_box, Box
from turbo_ml.interface.tutorial import TUTORIAL_DICT, TUTORIAL_NAMES
from turbo_ml import TurboML
import pandas as pd

LOGO = """
  ______           __                __  _____ 
 /_  __/_  _______/ /_  ____        /  |/  / / 
  / / / / / / ___/ __ \\/ __ \\______/ /|_/ / /  
 / / / /_/ / /  / /_/ / /_/ /_____/ /  / / /___
/_/  \\__,_/_/  /_.___/\\____/     /_/  /_/_____/
"""

WELCOME_MESSAGE = """
Welcome in Turbo-ML, this piece of software was created in order to make machine learning simple.
Given some dataset algorithms from library should be able to find the best machine learning algorithm with optimal parameters to solve problem provided in dataset whether this is classification or regression task.

## Guide ##
- load your dataset
- call TurboML(dataset) to train model
- call TurboML.predict(data) to find predictions
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

# TODO: Change sys.stdout.write clearing to a function to avoid repetition


def _choose_response(counter: int, default: str):
    if counter == 14:
        quit(0)
    if RESPONSES[counter] is None:
        return default
    return RESPONSES[counter]


def choose_option():
    """ Function to show main menu """
    def _ask(counter: int):
        response = _choose_response(
            counter, "Choose option: 'number', quit: 'q': ")
        choice = input(response)
        match choice.lower():
            case '1':
                sys.stdout.write('\033[F'*(box.num_lines+5))
                sys.stdout.write(f"\n{' '*66}"*(box.num_lines+5))
                sys.stdout.write('\033[F'*(box.num_lines+5))
                tutorial()
            case '2':
                sys.stdout.write('\033[F'*(box.num_lines+5))
                sys.stdout.write(f"\n{' '*66}"*(box.num_lines+5))
                sys.stdout.write('\033[F'*(box.num_lines+5))
                load_dataset()
            case '3':
                sys.stdout.write('\033[F'*(box.num_lines+5))
                sys.stdout.write(f"\n{' '*66}"*(box.num_lines+5))
                sys.stdout.write('\033[F'*(box.num_lines+5))
                show_credits()
            case 'q': quit(0)
            case _:
                sys.stdout.write('\033[F')
                print(' '*100)
                sys.stdout.write('\033[F')
                _ask(counter+1)

    options: dict = {'tutorial': 1,
                     'load dataset': 2,
                     'credits': 3}
    box = Box(options, topic='Choose option ')
    box.print()
    _ask(0)


def welcome():
    """ Function to show welcome message and start interactive interface """
    print(LOGO)

    def _ask(counter: int = 0):
        response = _choose_response(
            counter, "To continue type 'c', to quit type 'q': ")
        choice = input(response)
        match choice.lower():
            case 'c':
                sys.stdout.write('\033[F'*(num_lines+5))
                sys.stdout.write(f"\n{' '*100}"*(num_lines+5))
                sys.stdout.write('\033[F'*(num_lines+5))
                choose_option()
            case 'q': quit(0)
            case _:
                sys.stdout.write('\033[F')
                sys.stdout.write(f"{' '*100}\n")
                sys.stdout.write('\033[F')
                _ask(counter+1)
    num_lines = print_in_box(WELCOME_MESSAGE, topic='Welcome in Turbo-ML ')
    _ask(0)
    print(f"""Now there should be calculations for dataset in file {
          __DATASET_PATH}""")


def load_dataset():
    """ Function to show load dataset menu """
    print_in_box('Provide path to dataset file', topic='Load dataset')
    global __DATASET_PATH
    __DATASET_PATH = input('Path: ')
    dataset = pd.read_csv(__DATASET_PATH)
    print(f'Dataset columns: {dataset.columns}')
    target = input('Choose target column: ')
    model = TurboML(dataset, target)
    print('Model trained, give path to data to predict')
    test_set = pd.read_csv(input('Path: '))
    predictions = model.predict(test_set)
    print(f'Predictions: {predictions}')
    return


def show_credits():
    """ Function to show credits """
    def _ask(counter: int):
        response = _choose_response(
            counter, "Click anything to continue, quit: 'q': ")
        choice = input(response)
        match choice.lower():
            case 'q': quit(0)
            case _:
                sys.stdout.write('\033[F'*(num_lines+5))
                sys.stdout.write(f"\n{' '*66}"*(num_lines+5))
                sys.stdout.write('\033[F'*(num_lines+5))
                choose_option()

    num_lines = print_in_box(CREDITS + '\n' + LOGO, topic='Credits ')
    _ask(0)


def tutorial():
    """ Function to show tutorial """
    def _show_tutorial(tutorial_number: int):
        def _ask_inner(counter: int = 0):
            response = _choose_response(
                counter, "continue: 'c', menu: 'm', quit 'q': ")
            choice = input(response)

            match choice.lower():
                case 'q': quit(0)
                case 'm':
                    sys.stdout.write('\033[F'*(size+5))
                    sys.stdout.write(f"\n{' '*66}"*(size+5))
                    sys.stdout.write('\033[F'*(size+5))
                    tutorial()
                case 'c':
                    sys.stdout.write('\033[F'*(size+5))
                    sys.stdout.write(f"\n{' '*66}"*(size+5))
                    sys.stdout.write('\033[F'*(size+5))
                    if tutorial_number+1 in TUTORIAL_NAMES:
                        _show_tutorial(tutorial_number+1)
                    else:
                        tutorial()
                case _:
                    sys.stdout.write('\033[F')
                    print(' '*100)
                    sys.stdout.write('\033[F')
                    _ask_inner(counter+1)

        tutorial_name = TUTORIAL_NAMES.get(tutorial_number)
        tutorial_text = TUTORIAL_DICT.get(tutorial_number)

        size = print_in_box(tutorial_text, topic=tutorial_name)
        _ask_inner(0)

    def _ask(counter: int = 0):
        response = _choose_response(
            counter, "tutorial: 'number', menu: 'm', quit 'q': ")
        choice = input(response)
        match choice.lower():
            case 'q': quit(0)

            case 'm':
                sys.stdout.write('\033[F'*(box.num_lines+5))
                sys.stdout.write(f"\n{' '*66}"*(box.num_lines+5))
                sys.stdout.write('\033[F'*(box.num_lines+5))
                choose_option()

            case _:
                if not choice.isdigit():
                    sys.stdout.write('\033[F')
                    print(' '*100)
                    sys.stdout.write('\033[F')
                    _ask(counter+1)

                choice = int(choice)
                if choice in TUTORIAL_NAMES:
                    sys.stdout.write('\033[F'*(box.num_lines+5))
                    sys.stdout.write(f"\n{' '*66}"*(box.num_lines+5))
                    sys.stdout.write('\033[F'*(box.num_lines+5))
                    _show_tutorial(choice)

    box = Box(TUTORIAL_NAMES, topic='Choose tutorial ')
    box.print()
    _ask()


if __name__ == '__main__':
    welcome()
