import pickle
from .process import Process
from .model import Model
from typing import List, Iterable, Any
from ..utils import option


class Workflow:
    # TODO maybe inherit from list to make it more convenient
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
            aif = process.available_input_formats()
            if all(in_type not in aif for in_type in input_types):
                return False
        return True

    def to_model(self) -> Model:
        return WorkflowModel(self)

    def __getitem__(self, value) -> Process:
        return self._workflow[value]


class WorkflowModel(Model):
    def __init__(self, workflow: Workflow) -> None:
        self.workflow = workflow
        self.validator = None
        super().__init__()

    def train(self, data: Iterable, target: Iterable) -> None:
        for process in self.workflow:
            if self.validator and option.validation:
                try:
                    process.pr_validation(data, target)
                except Exception as e:
                    self.validator(process, e)
            try:
                data = process.tr(data, target)
            except Exception as e:
                self.validator(process, e)

    def predict(self, guess: Any) -> List:
        for process in self.workflow:
            if self.validator and option.validation:
                try:
                    process.pr_validation(guess)
                except Exception as e:
                    self.validator(process, e, validation=True)
            try:
                guess = process.pr(guess)
            except Exception as e:
                self.validator(process, e)
        return guess
