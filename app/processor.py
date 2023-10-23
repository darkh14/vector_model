"""Module for processing requests coming from main_app.py
    classes:
        Processor - class for processing requests
    """

import traceback
import inspect
from functools import wraps
from typing import Any, Callable, Optional, Annotated
from fastapi import Depends
from fastapi.security import OAuth2PasswordBearer


# noinspection PyUnresolvedReferences
import vm_logging
# noinspection PyUnresolvedReferences
import vm_security
# noinspection PyUnresolvedReferences
import vm_models
# noinspection PyUnresolvedReferences
import vm_settings
# noinspection PyUnresolvedReferences
import data_processing
# noinspection PyUnresolvedReferences
import db_processing
# noinspection PyUnresolvedReferences
import vm_background_jobs
# noinspection PyUnresolvedReferences
import actions
# noinspection PyUnresolvedReferences
import general
#
from vm_logging.exceptions import GeneralException, VMBaseException, UserNotFoundException
from db_processing.controller import initialize_connector_by_name, drop_connector
from vm_security import User, get_current_user, get_authentication_enabled, get_use_authentication

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")


class Processor:

    def __init__(self) -> None:
        """
        Defines available request methods
        """
        pass

    def get_requests_methods_description(self) -> list[dict[str, Any]]:
        """
        Gets available request methods
        :return: dict of actions (functions)
        """
        imported_modules = self._get_imported_module_names()

        methods = []

        for module_name in imported_modules:
            module_obj = globals()[module_name]
            method_descr_list = module_obj.get_actions()

            for method_descr in method_descr_list:
                if not get_use_authentication() and method_descr['name'] == 'get_token':
                    continue

                api_method = method_descr['func']
                api_method.__name__ = method_descr['name']
                api_method = self._check_and_complete_method(method_descr['path'],
                                                             method_descr.get('requires_authentication'))(api_method)

                sig = inspect.signature(api_method)

                parameters_list = list(sig.parameters.values())

                if method_descr['path'].split('/')[0] == '{db_path}':
                    db_parameter = inspect.Parameter('db_path',
                                                     kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
                                                     annotation=str)
                    parameters_list.append(db_parameter)

                if get_use_authentication() and method_descr.get('requires_authentication'):

                    parameter_c_user = inspect.Parameter('current_user',
                                                     kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
                                                     annotation=Annotated[User, Depends(self._get_current_user)],
                                                     default=None)

                    parameters_list.append(parameter_c_user)

                sig = sig.replace(parameters=parameters_list)

                api_method.__signature__ = sig

                method_descr['func'] = api_method

            methods.extend(method_descr_list)

        return methods

    @staticmethod
    def _get_imported_module_names() -> list[str]:
        """
        Gets names of imported modules to get actions from them
        :return: list of names of imported modules
        """

        result = ['data_processing',
                  'db_processing',
                  'vm_settings',
                  'vm_security',
                  'vm_models',
                  'vm_background_jobs',
                  'actions']

        return result

    @staticmethod
    def _get_current_user(token: Annotated[str, Depends(oauth2_scheme)]) -> Optional[User]:
        user_dict = get_current_user(token)
        return User.model_validate(user_dict) if user_dict else None

    @staticmethod
    def _check_and_complete_method(url_path: str, requires_authentication: bool) -> Callable:
        """
        Changes method description, checks method
        @return changed method
        """
        need_to_initialize_db = url_path.split('/')[0] == '{db_path}'

        def decorator(func: Callable[[Any], Optional[dict]]) -> Callable:

            @wraps(func)
            async def wrapper(*args, **kwargs) -> dict[str, Any]:

                db_name = kwargs.get('db_name')

                if need_to_initialize_db:
                    initialize_connector_by_name(db_name)

                if get_authentication_enabled() and requires_authentication:
                    user = kwargs.get('current_user')

                    if not user:
                        raise UserNotFoundException('User not found')

                    if user.disabled:
                        raise UserNotFoundException('User is disabled')

                    del kwargs['current_user']

                # noinspection PyBroadException
                try:
                    result = func(*args, **kwargs)
                    print(result)
                except Exception as exc:

                    if isinstance(exc, VMBaseException):
                        print(str(exc))
                        raise exc
                    else:
                        error_text = 'Error!\n' + traceback.format_exc()
                        final_exc = GeneralException

                        raise final_exc(error_text)

                if need_to_initialize_db:
                    drop_connector()

                return result

            return wrapper

        return decorator
