"""
Модуль реализации метакласса, необходимого для работы представления.

type(QObject) - метакласс общий для оконных компонентов Qt.
ABCMeta - метакласс для реализации абстрактных суперклассов.

CplusDMeta - метакласс для представления.
"""

from PyQt6.QtCore import QObject
from abc import ABCMeta


class BaseMeta(type(QObject), ABCMeta):
    pass
