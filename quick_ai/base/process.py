from abc import ABC, ABCMeta, abstractmethod
from typing import Set


class Process(ABC):
    def __init__(se
    lf) -> None:
        super().__init__()

    @abstractmethod
    def available_input_formats(self) -> Set:
        pass

    @abstractmethod
    def available_output_formats(self) -> Set:
        pass
