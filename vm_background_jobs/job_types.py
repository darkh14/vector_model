""" Module contains enum class of job statuses
        Classes:
            JobStatuses
"""

from enum import Enum


class JobStatuses(Enum):
    """
    Statuses of background job
    NEW - job is not in db
    REGISTERED - job is in DB but not started
    IN_PROCESS - job is started
    FINISHED - job is finished
    ERROR - error while executing of job
    INTERRUPTED - job was interrupted while executing
    """
    NEW = 'new'
    REGISTERED = 'registered'
    IN_PROCESS = 'in_process'
    FINISHED = 'finished'
    ERROR = 'error'
    DELETED = 'deleted'
    INTERRUPTED = 'interrupted'
