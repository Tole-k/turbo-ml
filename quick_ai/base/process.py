from abc import ABC, abstractmethod
from typing import Set


class Process(ABC):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def pr(self, data: any) -> any:
        pass

    @abstractmethod
    def tr(self, data: any) -> any:
        pass

    @abstractmethod
    def available_input_formats(self) -> Set:
        pass

    @abstractmethod
    def available_output_formats(self) -> Set:
        pass
