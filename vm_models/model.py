""" Contains model class, that provides working with models including
fitting and predicting
"""

__all__ = ['Model', 'get_model']


class Model:
    def __init__(self, model_id: str):
        self.id: str = model_id


def get_model(model_id: str) -> Model:
    return Model(model_id)
