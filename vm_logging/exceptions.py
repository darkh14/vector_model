""" Module for defining custom exceptions

    classes:
        VMBaseException - base custom exception. Writing error text to DB added
        RequestProcessException - Custom exception for request processing

"""

import traceback

__all__ = ['VMBaseException',
           'RequestProcessException',
           'SettingsControlException',
           'ParameterNotFoundException',
           'DBConnectorException',
           'LoadingProcessException',
           'BackgroundJobException',
           'ModelException']


class VMBaseException(Exception):
    """ Base class for custom exceptions, writing error text to db added

        properties:
            _message - error text to return, when exception is raised (not full but part of the text)
        methods:
            _get_full_error_message - forms full error text from message and base error text
            _write_log_to_db

    """
    def __init__(self, message: str = '', write_log: bool = False):
        self._message = message

        if write_log:
            self._write_log_to_db()

    def __str__(self):
        result = self._get_full_error_message()
        return result

    @property
    def message(self):
        return self._get_full_error_message()

    def _get_full_error_message(self) -> str:

        if self._message:
            result = self._message
        else:
            result = 'Error in VM module'

        return result

    def _write_log_to_db(self) -> bool:
        # TODO make a realization of writing log to db
        pass


class ParameterNotFoundException(VMBaseException):
    """Custom exception class raising when field is not found in request parameters
        raises with parameter - missing_parameter -  parameter name which is not found
    """

    def __init__(self, missing_parameter: str, message: str = '', write_log: bool = False):
        super().__init__(message, write_log)
        self._missing_parameter = missing_parameter

    def _get_full_error_message(self) -> str:
        return 'Parameter "{}" is not found in request parameters! '.format(self._missing_parameter)


class RequestProcessException(VMBaseException):
    """Custom exception class for request processing"""
    def _get_full_error_message(self) -> str:
        default_message = super()._get_full_error_message()
        return 'Error while request processing! ' + default_message


class SettingsControlException(VMBaseException):
    """Custom exception class for settings controlling"""
    def _get_full_error_message(self) -> str:
        default_message = super()._get_full_error_message()
        return 'Error while setting controlling! ' + default_message


class DBConnectorException(VMBaseException):
    """Custom exception class for DB connector"""

    def _write_log_to_db(self) -> bool:
        """ Do nothing because we can not write into DB"""
        # TODO write log to .log file
        pass

    def _get_full_error_message(self) -> str:
        default_message = super()._get_full_error_message()
        return 'Error while working with DB! ' + default_message


class LoadingProcessException(VMBaseException):
    """Custom exception class for loading errors"""
    def _get_full_error_message(self) -> str:
        default_message = super()._get_full_error_message()
        return 'Loading error! ' + default_message


class BackgroundJobException(VMBaseException):
    """ Custom exception class for background jobs """
    def _get_full_error_message(self) -> str:
        default_message = super()._get_full_error_message()
        return 'Error in background job! ' + default_message


class ModelException(VMBaseException):
    """ Custom exception class models """
    def _get_full_error_message(self) -> str:
        default_message = super()._get_full_error_message()
        return 'Error in model! ' + default_message