"""
Module for defining custom exceptions
"""

__all__ = ['VMBaseException',
           'SettingsControlException',
           'ParametersFormatError',
           'DBConnectorException',
           'LoadingProcessException',
           'BackgroundJobException',
           'ModelException',
           'GeneralException']


class VMBaseException(Exception):
    """ Base class for custom exceptions, writing error text to db added

        properties:
            _message - error text to return, when exception is raised (not full but part of the text)
        methods:
            _get_full_error_message - forms full error text from message and base error text
            _write_log_to_db

    """
    def __init__(self, message: str = '', write_log: bool = False) -> None:
        """
        defines _message variable
        :param message: message of exception
        :param write_log: need to write log to db if True
        """
        self._message = message

        if write_log:
            self._write_log_to_db()

    def __str__(self):
        """
        Presentation of exception
        :return: str of full error message
        """
        result = self._get_full_error_message()
        return result

    @property
    def message(self):
        """
        Property returns full error message. Forms dynamically
        :return: message str
        """
        return self._get_full_error_message()

    def _get_full_error_message(self) -> str:
        """
        Forms error message
        :return: error message str
        """
        if self._message:
            result = self._message
        else:
            result = 'Error in VM module'

        return result

    def _write_log_to_db(self) -> bool:
        """
        For writing error information to db
        :return result of writing
        """
        # TODO make a realization of writing log to db
        pass


class ParametersFormatError(VMBaseException):
    """ Custom exception for format error (in match - case) """
    def _get_full_error_message(self) -> str:
        """
        Error message for exception
        :return: error message
        """
        default_message = super()._get_full_error_message()
        return 'Error in model! ' + default_message


class SettingsControlException(VMBaseException):
    """Custom exception class for settings controlling"""
    def _get_full_error_message(self) -> str:
        """
        Error message for settings control exception
        :return: error message
        """
        default_message = super()._get_full_error_message()
        return 'Error while setting controlling! ' + default_message


class DBConnectorException(VMBaseException):
    """Custom exception class for DB connector"""

    def _write_log_to_db(self) -> bool:
        """ Do nothing because we can not write into DB
        :return result of writing log to db, True if successful
        """
        # TODO write log to .log file
        pass

    def _get_full_error_message(self) -> str:
        """
        Error message for db connector exception
        :return: error message
        """
        default_message = super()._get_full_error_message()
        return 'Error while working with DB! ' + default_message


class LoadingProcessException(VMBaseException):
    """Custom exception class for loading errors"""
    def _get_full_error_message(self) -> str:
        """
        Error message for loading process exception
        :return: error message
        """
        default_message = super()._get_full_error_message()
        return 'Loading error! ' + default_message


class BackgroundJobException(VMBaseException):
    """ Custom exception class for background jobs """
    def _get_full_error_message(self) -> str:
        """
        Error message for background job exception
        :return: error message
        """
        default_message = super()._get_full_error_message()
        return 'Error in background job! ' + default_message


class ModelException(VMBaseException):
    """ Custom exception class models """
    def _get_full_error_message(self) -> str:
        """
        Error message for model exception
        :return: error message
        """
        default_message = super()._get_full_error_message()
        return 'Error in model! ' + default_message


class GeneralException(VMBaseException):
    pass
