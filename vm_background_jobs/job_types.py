""" Module contains enum class of job statuses
        Classes:
            JobStatuses
"""

from enum import Enum


class JobStatuses(Enum):
    NEW = 'new'
    REGISTERED = 'registered'
    IN_PROCESS = 'in_process'
    FINISHED = 'finished'
    ERROR = 'error'
    DELETED = 'deleted'
