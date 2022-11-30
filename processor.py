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

import vm_logging
import vm_models
import vm_settings
import data_processing
import db_processing
import vm_background_jobs
import actions

import general

from vm_logging.exceptions import RequestProcessException
from vm_settings import controller as settings_controller
# from db_processing import connector as db_controller

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

    def __init__(self):

        self._request_methods: dict[str, Callable] = self._get_requests_methods()

    @abstractmethod
    def process(self, environ: dict[str, Any], start_response: Callable): ...
    """ Abstract method for processing requests"""

    def _process_request(self, environ: dict[str, Any], start_response: Optional[Callable] = None) -> list:
        """ Method for processing request after forming environ dict (when http environ is ready)
            performs request method and converts output to bytes
        """
        if not start_response:
            start_response = self.t_start_response

        request_parameters = self._get_request_parameters_from_environ(environ)

        output_dict = self._process_with_parameters(request_parameters)
        output_list = self.transform_output_parameters_to_str(output_dict)
        output_len = len(output_list[0])

        start_response('200 OK', [('Content-type', 'text/html'), ('Content-Length', str(output_len))])

        return output_list

    @staticmethod
    def t_start_response(result: str, headers: list[tuple[str, str]]) -> None:
        """ function for response
            parameters:
                result - status of response
                headers - http headers, ex. content type
        """
        pass

    def _process_with_parameters(self, parameters: dict[str, Any]):
        """ Method for processing request after forming parameters.
            Performs checking service_name, request_type n parameters.
            Determines method according to request type and executes it
        """
        if parameters.get('service_name') != SERVICE_NAME:
            raise RequestProcessException('Service name "{}" is not allowed. '.format(parameters.get('service_name')) +
                                          'correct service name is "{}"'.format(SERVICE_NAME))

        request_type = parameters.get('request_type')

        if not request_type:
            raise RequestProcessException('Property "request type" is not in parameters. '
                                          'property "request type" is required')

        method = self._request_methods.get(request_type)

        if not method:
            raise RequestProcessException('Request type "{}" is not supported'.format(request_type))

        result = method(parameters)

        return {'status': 'OK', 'error': '', 'result': result}

    @staticmethod
    def transform_output_parameters_to_str(output: dict[str, Any]) -> list:
        """Transform output from dict to list with one element contains bytes using json"""
        output_str = json.dumps(output, ensure_ascii=False).encode()

        return [output_str]

    def _get_request_parameters_from_environ(self, environ: dict[str, Any]) -> dict[str, Any]:
        """ Gets the necessary parameters from environ
            Checks REQUEST_METHOD -  must be POST
            parameters get from environ['wsgi.input']


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
    def _parameters_from_json(json_string: str|bytes) -> dict[str, Any]:

        if type(json_string) == bytes:
            json_string = json_string.decode('utf-8-sig')
        else:
            if ord(json_string[0]) == 65279:
                json_string = json_string[1:]

            json_string = json_string.encode('utf-8-sig')

        return json.loads(json_string)

    @staticmethod
    def _get_requests_methods():

        imported_modules = ['data_processing',
                            'db_processing',
                            'vm_logging',
                            'vm_models',
                            'vm_settings',
                            'vm_background_jobs',
                            'actions']

        methods = dict()

        for module_name in imported_modules:
            module_obj = globals()[module_name]
            methods.update(module_obj.get_actions())

        return methods


class _HttpProcessor(Processor):
    """Class for processing http requests

        Methods:
            process - main method for processing
    """

    def process(self, environ: dict[str, any], start_response: Callable[[str, list[bytes]], Callable]) -> list:
        """ Main method for processing requests (http)

            Parameters:
                environ - dict, parameters of request
                start_response - callable, function object, using for response
            Return:
                dict - response of request
        """
        output = self._process_request(environ, start_response)
        return output


class _FileProcessor(Processor):
    """Class for processing test requests from files

        Methods:
            process - main method for processing
    """

    def __init__(self):
        super().__init__()

        self._request_types_for_choosing: dict[int, str] = self._get_request_types_for_choosing()

    def process(self, environ: None, start_response: None) -> list:
        """ Main method for processing requests (file while testing)

            Parameters:
                environ - None
                start_response - None

                Parameters are for compatible. In file mode they are not needed
            Return:
                dict - response of request
        """

        request_type = self._choose_request_type()

        environ = self._get_environ_from_json(request_type)

        output = self._process_request(environ, start_response)

        return output

    def _get_request_types_for_choosing(self) -> dict[int, str]:
        return {key + 1: value for key, value in enumerate(self._request_methods.keys())}

    def _choose_request_type(self) -> str:

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
            start_response: Optional[Callable[[str, list[bytes]], Callable]] = None) -> dict[str, Any]:
    """ Main function for processing requests
        Processes all requests (at product and test modes)
        Parameters:
            environ: dict - of request parameters:
            start_response: function object - for sending response
        returns dict of response.

        For more information, see the module wsgi.py docs
    """
    global PROCESSOR

    if TEST_MODE:
        # in test mode it is not need to intercept exceptions
        if not PROCESSOR:
            PROCESSOR = _get_processor(environ)

        output = PROCESSOR.process(environ, start_response)
    else:
        try:
            if not PROCESSOR:
                PROCESSOR = _get_processor(environ)

            output = PROCESSOR.process(environ, start_response)
        except RequestProcessException as request_ex:
            error_text = str(request_ex)
            output = PROCESSOR.transform_output_parameters_to_str({'status': 'error', 'error_text': error_text})
        except Exception as base_ex:

            error_text = 'Error!\n' + traceback.format_exc()
            output = PROCESSOR.transform_output_parameters_to_str({'status': 'error', 'error_text': error_text})

    print(output)
    return output


def _get_processor(environ: dict[str, Any]) -> Processor:
    """ Chooses type of processor according to http request or direct launch """
    if environ:
        return _HttpProcessor()
    else:
        return _FileProcessor()

