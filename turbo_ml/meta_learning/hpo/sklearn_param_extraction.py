import json
from numbers import Integral
from types import NoneType
from sklearn.utils import all_estimators
from sklearn.utils._param_validation import Interval,  Options

hyperparameters = {}
for name, classifier in all_estimators(type_filter="classifier"):
    hyperparameters[name] = []
    for param in classifier._get_param_names():
        constraints = classifier._parameter_constraints[param]
        print(param, constraints)
        options_constraint = next(
            (constraint for constraint in constraints if isinstance(constraint, Options)), None)
        interval_constraints = [constraint for constraint in constraints if isinstance(
            constraint, (Interval, Integral))]
        boolean_constraint = next(
            (constraint for constraint in constraints if constraint == 'boolean'), None)
        none_constraint = next(
            (0 for constraint in constraints if isinstance(constraint, NoneType)), None)
        if options_constraint:
            choices = list(options_constraint.options)
            if len(choices) > 1:
                hyperparameters[name].append({
                    "name": param,
                    "type": "categorical",
                    "choices": list(options_constraint.options),
                    "optional": none_constraint is not None
                })
            else:
                hyperparameters[name].append({
                    "name": param,
                    "type": "no_choice",
                    "choices": list(options_constraint.options),
                    "optional": none_constraint is not None
                })
        elif len(interval_constraints) == 2:
            for constraint in interval_constraints:
                if constraint.type == Integral:
                    continue
                hyperparameters[name].append({
                    "name": param,
                    "type": 'float',
                    "min": constraint.left,
                    "max": constraint.right,
                    "optional": none_constraint is not None
                })
        elif len(interval_constraints) == 1:
            constraint = interval_constraints[0]
            hyperparameters[name].append({
                "name": param,
                "type": 'int' if constraint.type == Integral else 'float',
                "min": constraint.left,
                "max": constraint.right,
                "optional": none_constraint is not None
            })
        elif boolean_constraint:
            hyperparameters[name].append({
                "name": param,
                "type": "bool",
                "optional": none_constraint is not None
            })
with open('turbo_ml/hpo/not_overwrite_sklearn_hyperparameters.json', 'w') as f:
    json.dump(hyperparameters, f, indent=4)
