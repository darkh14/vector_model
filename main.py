import requests
import json
import os

from processor import Processor

if __name__ == '__main__':

    http_processor = Processor()

    print('Available methods:')

    method_descr_list = http_processor.get_requests_methods_description()
    for index, method_descr in enumerate(method_descr_list):
        print('   {} - {}'.format(index + 1, method_descr['name']))

    number_of_method = int(input("Enter number of method: "))

    if number_of_method:

        method_descr = method_descr_list[number_of_method-1]

        root_url = f'http://vbm_3_test.ml'

        url = '{}/{}'.format(root_url, method_descr['path'])

        file_path = os.path.join(os.path.dirname(__file__), 'parameters_test', '{}.json'.format(method_descr['name']))

        if not os.path.exists(file_path):
            raise FileNotFoundError('There is no file appropriate to method "{}". \n'
                                    'No such file or directory: "{}"'.format(method_descr['name'], file_path))

        with open(file_path, 'r', encoding="utf-8") as f:
            parameters_string = f.read()

        if ord(parameters_string[0]) == 65279:
            parameters_string = parameters_string[1:]

        parameters = json.loads(parameters_string)

        url_parameters = {key: value.replace('\\', '/') if isinstance(value, str) else value for key, value
                          in parameters['url_parameters'].items()}

        if method_descr['http_method'] == 'get':
            response = requests.get(url, params=url_parameters)
        elif method_descr['http_method'] == 'post':
            response = requests.post(url, params=url_parameters, json=parameters['body'])
        else:
            raise ValueError('Wrong type of http request "{}"'.format(method_descr['http_method']))

        if response.status_code == 200:
            print(response.json())
        else:
            print(response.text)
