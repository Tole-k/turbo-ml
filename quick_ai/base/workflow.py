import pickle
from .process import Process
from typing import List


class Workflow:
    # TODO maybe inherit from list to make it more convinient
    def __init__(self) -> None:
        super().__init__()
        self._workflow: List[Process] = []

    def append(self, process) -> None:
        self._workflow.append(process)

    def save(self, path: str) -> None:
        with open(path, 'wb') as file:
            pickle.dump(self._workflow, file)

    def check_dependencies(self, input_type) -> bool:
        # TODO define input_type format
        input_types = [input_type]
        for process in self._workflow:
            aif = process.avalible_input_formats()
            if all(in_type not in aif for in_type in input_types):
                return False
        return True

    def __getitem__(self, value) -> Process:
        return self._workflow[value]
