from abc import ABCMeta, abstractmethod


class BaseObserver(metaclass=ABCMeta):
    """
    Абстрактный суперкласс для всех наблюдателей.
    """
    @abstractmethod
    def model_changed(self):
        """
        Метод, который будет вызван у наблюдателя при изменении модели.
        """
        pass