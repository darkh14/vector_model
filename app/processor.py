"""Module for processing requests coming from main_app.py
    classes:
        Processor - class for processing requests
    """

import traceback
import inspect
from functools import wraps
from typing import Any, Callable, Optional


# noinspection PyUnresolvedReferences
import vm_logging
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
from vm_logging.exceptions import GeneralException
from api_types import get_response_type
from db_processing.controller import initialize_connector, drop_connector


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
                api_method = method_descr['func']
                api_method.__name__ = method_descr['name']
                api_method = self._check_and_complete_method(method_descr['requires_db'])(api_method)

                sig = inspect.signature(api_method)

                parameters_list = list(sig.parameters.values())

                parameter_db = inspect.Parameter('db',
                                                 kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
                                                 annotation=str)

                parameters_list = [parameter_db] + parameters_list

                sig = sig.replace(parameters=parameters_list,
                                  return_annotation=get_response_type(sig.return_annotation))

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
                  'vm_models',
                  'vm_background_jobs',
                  'actions']

        return result

    @staticmethod
    def _check_and_complete_method(need_to_initialize_db: bool) -> Callable:
        """
        Changes method description, checks method
        @return changed method
        """
        def decorator(func: Callable[[Any], Optional[dict]]) -> Callable:

            @wraps(func)
            async def wrapper(db: str, *args, **kwargs) -> dict[str, Any]:

                error_text = ''
                status = 'OK'

                if need_to_initialize_db:
                    initialize_connector(db)

                # noinspection PyBroadException
                try:
                    result = func(*args, **kwargs)
                    print(result)
                except Exception:
                    error_text += 'Error!\n' + traceback.format_exc()
                    print(error_text)

                    raise GeneralException(error_text)

                result = {'result': result, 'status': status, 'error_text': error_text}

                if need_to_initialize_db:
                    drop_connector()

                return result

            return wrapper

        return decorator

