""" Module for setting and getting 'settings'
    classes:
        SettingsController - main class for controlling settings

    functions:
        get_var - for getting required setting
        set_var  - for setting value
        _get_settings_controller - auxiliary for caching and getting setting controller

    variables:
        CONTROLLER - cache variable for settings controller class
"""


import dotenv
import os
import json
from typing import Any

from . import defaults, passwords
from vm_logging.exceptions import SettingsControlException

CONTROLLER = None

__all__ = ['SettingsController', 'get_var', 'get_secret_var', 'set_var']


class SettingsController:
    """ Main class for adding and getting settings value
        settings are stored in environment variables.
        Env. vars are loaded from .env file in package directory
        passwords are stored in system password repository using keyring library

        properties:
            _dotenv_path - path to .env file where settings are stored
            _keys - list of available var names
            _secret_keys - dict of available secret keys (passwords) and paired user var keys

        methods:
            get_var - for getting variables (ordinary and secret)
            set_var - for setting variables (ordinary and secret)

    """
    def __init__(self):
        """ initialisation of path to .env file, keys and secret keys,
            loading settings from .env file to environ,
            setting default values for vars, which are not set"""

        self._dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
        self._keys = defaults.get_keys()
        self._secret_keys = defaults.get_secret_keys()
        self._special_type_keys = defaults.get_var_special_types()
        self._cache = {}

        dotenv.load_dotenv(self._dotenv_path)

        for key in self._keys:

            if key in self._secret_keys:
                c_var = self.get_secret_var(key)
            else:
                c_var = self.get_var(key)

            if c_var is None:
                value = getattr(defaults, key)
                self.set_var(key, value)

    def get_var(self, key: str) -> Any:
        """ For getting value. Supports non str types"""

        if key not in self._keys:
            raise SettingsControlException('Key "{}" is not available'.format(key))

        if key in self._secret_keys:
            raise SettingsControlException('Key "{}" is secret key (password)'.format(key))
        else:
            if key in self._cache:
                result = self._cache[key]
            else:
                result = os.getenv(key)

                if result is not None and key in self._special_type_keys:
                    result = self._str_to_special_type(result, self._special_type_keys[key])
                self._cache[key] = result

        return result

    def get_secret_var(self, key: str) -> str:
        """ For getting secret value."""

        if key not in self._secret_keys:
            raise SettingsControlException('Key "{}" is not secret key (password)'.format(key))
        else:
            result = self._get_secret_value(key)

        return result

    def set_var(self, key: str, value: Any) -> bool:
        """ For setting value. Supports non str types and passwords"""

        if key not in self._keys:
            raise SettingsControlException('Key "{}" is not available'.format(key))

        self._cache[key] = value

        if key in self._secret_keys:
            self._set_secret_value(key, value)
            result = True
        else:

            if value is None:
                value_str = ''
            else:
                if key in self._special_type_keys:
                    value_str = self._special_type_to_str(value, self._special_type_keys[key])
                else:
                    value_str = str(value)

            result = dotenv.set_key(self._dotenv_path, key, value_str)
            result = result[0]

            if not result:
                raise SettingsControlException('Parameter "{}" s not set'.format(key))

        return result

    @staticmethod
    def _special_type_to_str(value: Any, type_str: str) -> str:
        """ Converts vars to str according to types in self._special_type_keys
            available types - dict, list, int, float, bool
        """
        if type_str in ('dict', 'list'):
            result = json.dumps(value)
        elif type_str in ('int', 'float', 'bool'):
            result = str(value)
        else:
            raise SettingsControlException('Unsupported var type {}'.format(type_str))

        return result

    @staticmethod
    def _str_to_special_type(str_value: str, type_str: str) -> Any:
        """ Converts str vars to type specified in self._special_type_keys
            available types - dict, list, int, float, bool
        """
        if type_str in ('dict', 'list'):
            if not str_value:
                result = {} if type_str == 'dict' else []
            else:
                result = json.loads(str_value)
        elif type_str == 'int':
            result = int(str_value)
        elif type_str == 'float':
            result = float(str_value)
        elif type_str == 'bool':
            result = bool(str_value)
        else:
            raise SettingsControlException('Unsupported var type {}'.format(type_str))

        return result

    def _get_secret_value(self, key: str) -> str:
        """ Get password value """
        user_value = self.get_var(self._secret_keys[key])
        return passwords.get_password(key, user_value)

    def _set_secret_value(self, key: str, value: str) -> str:
        """ Set password value. Supports only str type of value """
        user_value = self.get_var(self._secret_keys[key])
        passwords.set_password(key, user_value, value)


def get_var(key: str) -> Any:
    settings_controller = _get_settings_controller()
    return settings_controller.get_var(key)


def get_secret_var(key: str) -> str:
    settings_controller = _get_settings_controller()
    return settings_controller.get_secret_var(key)


def set_var(key: str, value: Any) -> bool:
    settings_controller = _get_settings_controller()
    return settings_controller.set_var(key, value)


def _get_settings_controller() -> SettingsController:
    """ caching settings controller object """
    global CONTROLLER

    if not CONTROLLER:
        CONTROLLER = SettingsController()

    return CONTROLLER
