""" Module for defining enums and other types using in data loading
    Classes:
        LoadingTypes - types of loading (full or increment)
        LoadingStatuses - statuses of loading according to loading process
"""

from enum import Enum


class LoadingTypes(Enum):
    """
    Full loading deletes all old rows of data
    Increment loading saves old data, updates sum and qty if necessary
    and adds new data
    """
    FULL = 'full'
    INCREMENT = 'increment'


class LoadingStatuses(Enum):
    """    Loading statuses
    also used as package statuses
    NEW - loading or package is not registered in db
    REGISTERED - loading or package is registered on db but not started
    PRE_STARTED - process is started but background job is not started (status used when process
    executes in background mode)
    IN_PROCESS - process of loading is started
    PARTIALLY_LOADED - some packages in loading are loaded but not all
    LOADED - loading is finished (or loading of current package is finished when it is package status)
    DELETED - loading record is deleted from db
    ERROR - error while loading
    """
    NEW = 'new'
    REGISTERED = 'registered'
    PRE_STARTED = 'pre_started'
    IN_PROCESS = 'in_process'
    PARTIALLY_LOADED = 'partially_loaded'
    LOADED = 'loaded'
    DELETED = 'deleted'
    ERROR = 'error'
