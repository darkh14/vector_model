""" Module contains list of models (cache). Provides getting model,
    fitting, predicting and other functions to work with models
        Functions:
            _get_model - for getting model
            fit - for fitting model
            predict - for predicting data with model
            initialize - for initializing new model

"""

from typing import Any

from model import Model, get_model

MODELS: list[Model] = list()


def fit(parameters: dict[str, Any]) -> dict[str, Any]:
    """ For fitting model """
    pass


def predict(parameters: dict[str, Any]) -> dict[str, Any]:
    """ For predicting data with model """
    pass


def initialize(parameters: dict[str, Any]) -> dict[str, Any]:
    """ For initializing new model """
    if not parameters.get('model'):
        raise
    model = _get_model()
    result = model.initialize(parameters.get('model'))
    return result


def _drop(parameters: dict[str, Any]) -> dict[str, Any]:
    """ For deleting model from db """
    pass


def _get_info(parameters: dict[str, Any]) -> dict[str, Any]:
    """ For getting model info """
    pass


def _drop_fitting(parameters: dict[str, Any]) -> dict[str, Any]:
    """ For deleting fit data from model """
    pass


def _update(parameters: dict[str, Any]) -> dict[str, Any]:
    """ For updating model parameters """
    pass


def _get_model(model_id: str) -> Model:
    models = [c_model for c_model in MODELS if c_model.id == model_id]

    if models:
        model = models[0]
    else:
        model = get_model(model_id)
        MODELS.append(model)

    return model
