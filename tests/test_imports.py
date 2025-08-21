def test_basic_imports():
    from sageml import __name__ as sageml_name
    assert sageml_name is not None
    from sageml.algorithms import __name__ as alg_name
    assert alg_name is not None
    from sageml.base import __name__ as base_name
    assert base_name is not None
    from sageml.meta_learning.model_prediction import __name__ as ml_pred
    assert ml_pred is not None
    from sageml.utils import __name__ as utils
    assert utils is not None
    from sageml.interface import __name__ as interface
    assert interface is not None
    from sageml.preprocessing import __name__ as preprocessing
    assert preprocessing is not None
