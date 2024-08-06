class WrongDataTypeException(Exception):
    "Exception raised when the data type is not correct"
    pass


class WrongShapeException(Exception):
    "Exception raised when the shape is not correct"
    pass


class NotTrainedException(Exception):
    "Exception raised when the model is not trained"
    pass
