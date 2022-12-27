"""Module for processing requests coming from wsgi.py (main.py in test mode)

    functions:
        process - main function for processing requests

    classes:
        Processor - abstract class for processing requests

    variables:
        PROCESSOR - saving instance of Processor class (cache)

    """

import traceback
import json
import os.path
from typing import Any, Callable, Optional
from abc import ABC, abstractmethod
import inspect

import vm_logging
import vm_models
import vm_settings
import data_processing
import db_processing
import vm_background_jobs
import actions

import general

from vm_logging.exceptions import RequestProcessException, ParameterNotFoundException
from vm_settings import controller as settings_controller
from db_processing.controller import initialize_connector, drop_connector

PROCESSOR = None
""" Current class for processing request (for caching) """

TEST_MODE: bool = bool(settings_controller.get_var('TEST_MODE'))
""" In test mode errors raise but not process """
SERVICE_NAME: str = settings_controller.get_var('SERVICE_NAME')
""" For separating different services """


class Processor(ABC):
    """ An abstract class for processing requests. 2 classes are inherited from it.
    _HttpProcessor for processing http requests in release mode,
    _FileProcessor for processing test requests (test mode)

    properties:
        _request_methods - dict of names and objects of all request methods.
    methods:
        process (abstract).
    """

    def __init__(self) -> None:
        """
        Defines available request methods
        """
        self._request_methods: dict[str, Callable] = self._get_requests_methods()

    @abstractmethod
    def process(self, environ: dict[str, Any], start_response: Callable) -> list[Any]:
        """
        Abstract method for processing requests
        :param environ: dict of environ variable
        :param start_response: function for response
        :return list of request result
        """

    def _process_request(self, environ: dict[str, Any], start_response: Optional[Callable] = None) -> list[Any]:
        """ Method for processing request after forming environ dict (when http environ is ready)
            performs request method and converts output to bytes
            :param environ: dict of environ variable
            :param start_response: function for response
            :return: list of request result
        """
        if not start_response:
            start_response = self.t_start_response

        request_parameters = self._get_request_parameters_from_environ(environ)

        if TEST_MODE:
            output_dict = self._process_with_parameters(request_parameters)
        else:
            try:
                output_dict = self._process_with_parameters(request_parameters)
            except RequestProcessException as request_ex:
                error_text = str(request_ex)
                output_dict = {'status': 'error', 'error_text': error_text}
            except Exception as base_ex:
                error_text = 'Error!\n' + traceback.format_exc()
                output_dict = {'status': 'error', 'error_text': error_text}

        output_list = self._transform_output_parameters_to_str(output_dict)
        output_len = len(output_list[0])

        start_response('200 OK', [('Content-type', 'text/html'), ('Content-Length', str(output_len))])
        return output_list

    @staticmethod
    def t_start_response(result: str, headers: list[tuple[str, str]]) -> None:
        """ function for response
        :param result: status of response
        :param headers: http headers, ex. content type
        """
        pass

    def _process_with_parameters(self, parameters: dict[str, Any]) -> dict[str, Any]:
        """ Method for processing request after forming parameters.
            Performs checking service_name, request_type n parameters.
            Determines method according to request type and executes it
            :param parameters: request parameters
            :return: dict of request result
        """
        if parameters.get('service_name') != SERVICE_NAME:
            raise RequestProcessException('Service name "{}" is not allowed. '.format(parameters.get('service_name')) +
                                          'correct service name is "{}"'.format(SERVICE_NAME))

        request_type = parameters.get('request_type')

        if not request_type:
            raise RequestProcessException('Property "request type" is not in parameters. '
                                          'property "request type" is required')

        names_without_db = self._get_action_names_without_db_using()

        if request_type not in names_without_db:
            if 'db' not in parameters:
                raise ParameterNotFoundException('Parameter "db" not found in parameters')

            initialize_connector(parameters['db'])

        method = self._request_methods.get(request_type)

        if not method:
            raise RequestProcessException('Request type "{}" is not supported'.format(request_type))

        result = method(parameters)

        if request_type not in names_without_db:
            drop_connector()

        return {'status': 'OK', 'error': '', 'result': result}

    @staticmethod
    def _transform_output_parameters_to_str(output: dict[str, Any]) -> list[bytes]:
        """Transform output from dict to list with one element contains bytes using json
        :param output: output dict
        :return: list with transformed output bytes
        """
        output_str = json.dumps(output, ensure_ascii=False).encode()

        return [output_str]

    def _get_request_parameters_from_environ(self, environ: dict[str, Any]) -> dict[str, Any]:
        """ Gets the necessary parameters from environ
            Checks REQUEST_METHOD -  must be POST
            parameters get from environ['wsgi.input']
            :param environ: dict of environ
            :return: transformed request parameters
        """
        request_parameters = dict()
        if environ.get('REQUEST_METHOD') == 'POST':

            content_length = int(environ.get('CONTENT_LENGTH')) if 'CONTENT_LENGTH' in environ else 0

            par_string = ''

            if content_length:
                par_string = environ['wsgi.input'].read(content_length)
            else:
                par_list = environ.get('wsgi.input')
                if par_list:
                    for par_element in par_list:
                        par_string = par_element

            if par_string:
                request_parameters = self._parameters_from_json(par_string)

        else:
            raise RequestProcessException('Request method must be "post"')

        return request_parameters

    @staticmethod
    def _parameters_from_json(json_string: str | bytes) -> dict[str, Any]:
        """
        Gets parameters dict from json string
        :param json_string: input json string
        :return: result dict of parameters
        """
        if type(json_string) == bytes:
            json_string = json_string.decode('utf-8-sig')
        else:
            if ord(json_string[0]) == 65279:
                json_string = json_string[1:]

            json_string = json_string.encode('utf-8-sig')

        return json.loads(json_string)

    def _get_requests_methods(self) -> dict[str, Callable]:
        """
        Gets available request methods
        :return: dict of actions (functions)
        """
        imported_modules = self._get_imported_module_names()

        methods = dict()

        for module_name in imported_modules:
            module_obj = globals()[module_name]
            methods.update(module_obj.get_actions())

        return methods

    def _get_action_names_without_db_using(self) -> list[str]:
        """
        Gets available request methods
        :return: dict of actions (functions)
        """
        imported_modules = self._get_imported_module_names()

        names = list()

        for module_name in imported_modules:
            module_obj = globals()[module_name]

            module_function_names = [el[0] for el in inspect.getmembers(module_obj, predicate=inspect.isfunction)]

            if 'get_action_names_without_db_using' in module_function_names:
                action_names = module_obj.get_action_names_without_db_using()
                names = names + action_names

        return names

    @staticmethod
    def _get_imported_module_names() -> list[str]:
        """
        Gets names of imported modules to get actions from them
        :return: list of names of imported modules
        """
        return ['data_processing',
                'db_processing',
                'vm_logging',
                'vm_models',
                'vm_settings',
                'vm_background_jobs',
                'actions']


class _HttpProcessor(Processor):
    """Class for processing http requests
        Methods:
            process - main method for processing
    """

    def process(self, environ: dict[str, any], start_response: Callable[[str, list[bytes]], Callable]) -> list[Any]:
        """ Main method for processing requests (http)
        :param environ: parameters of request
        :param start_response: function object, using for response
        :return response of request
        """

        output = self._process_request(environ, start_response)
        return output


class _FileProcessor(Processor):
    """Class for processing test requests from files
        Methods:
            process - main method for processing
    """

    def __init__(self) -> None:
        """
        Defines _request_types_for_choosing variable
        """
        super().__init__()

        self._request_types_for_choosing: dict[int, str] = self._get_request_types_for_choosing()

    def process(self, environ: None, start_response: None) -> list[Any]:
        """ Main method for processing requests (file while testing).
        Parameters are for compatible. In file mode they are not needed
        :param environ: None
        :param start_response: None
        :return response of request
        """

        request_type = self._choose_request_type()

        environ = self._get_environ_from_json(request_type)

        output = self._process_request(environ, start_response)

        return output

    def _get_request_types_for_choosing(self) -> dict[int, str]:
        """
        Returns dict with numbers and request methods for choosing
        :return: dict of request methods
        """
        return {key + 1: value for key, value in enumerate(self._request_methods.keys())}

    def _choose_request_type(self) -> str:
        """ For choosing request method interactively (using input)
        :return: chosen str of request type
        """
        print('Available request types:')
        for key, value in self._request_types_for_choosing.items():
            print('   {} - {}'.format(key, value))

        number_of_request = int(input("Enter number of request type: "))

        request_type = self._request_types_for_choosing.get(number_of_request)

        if not request_type:
            raise RequestProcessException('Wrong number of request type "{}"'.format(number_of_request))

        return request_type

    @staticmethod
    def _get_environ_from_json(request_type: str) -> dict[str, Any]:
        """
        Forms environ dict to imitate http request
        :param request_type: string of request type
        :return: environ dict
        """
        file_path = 'parameters_test/' + request_type + '.json'

        if not os.path.exists(file_path):
            raise FileNotFoundError('There is no file appropriate to request type "{}". \n'
                                    'No such file or directory: "{}"'.format(request_type, file_path))

        with open(file_path, 'r', encoding="utf-8") as fp:
            parameters_string = fp.read()

        environ = dict()
        environ['REQUEST_METHOD'] = 'POST'

        environ['wsgi.input'] = [parameters_string]

        return environ


def process(environ: Optional[dict[str, any]] = None,
            start_response: Optional[Callable[[str, list[bytes]], Callable]] = None) -> list[Any]:
    """ Main function for processing requests
        Processes all requests (at product and test modes)
        :param environ: dict - of request parameters
        :param start_response: function object - for sending response
        :return list of bytes - response
    """
    global PROCESSOR

    if not PROCESSOR:
        PROCESSOR = _get_processor(environ)

    output = PROCESSOR.process(environ, start_response)

    print(output)
    return output


def _get_processor(environ: Optional[dict[str, Any]]) -> Processor:
    """ Chooses type of processor according to http request or direct launch
    :param environ: dict of environ (http request)
    :return: processor object
    """
    if environ:
        return _HttpProcessor()
    else:
        return _FileProcessor()

