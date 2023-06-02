""" Module for defining enums and other types using in data loading
    Classes:
        LoadingTypes - types of loading (full or increment)
        LoadingStatuses - statuses of loading according to loading process
"""

from enum import Enum


class LoadingTypes(Enum):
    FULL = 'full'
    INCREMENT = 'increment'


class LoadingStatuses(Enum):
    NEW = 'new'
    REGISTERED = 'registered'
    PRE_STARTED = 'pre_started'
    IN_PROCESS = 'in_process'
    PARTIALLY_LOADED = 'partially_loaded'
    LOADED = 'loaded'
    DELETED = 'deleted'
    ERROR = 'error'
