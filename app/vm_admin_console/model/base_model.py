import requests
from urllib.parse import urljoin
import json

from typing import Optional, Any


class ConnectorException(Exception):
    pass


class Connector:

    def __init__(self, host: str = '', port: int = 0):

        self.is_connected = False
        self.is_connection_error = False
        self.error_text = ''
        self.host = host
        self.port = port

    def connect(self):

        if not self.host:
            self.is_connected = False
            self.error_text = 'Host must be filled'
            self.is_connection_error = True

        try:
            self._make_get_request('ping')
        except ConnectorException as e:
            self.is_connected = False
            self.error_text = str(e)
            self.is_connection_error = True
        else:
            self.is_connected = True
            self.error_text = ''
            self.is_connection_error = False

    def disconnect(self):
        self.is_connected = False
        self.error_text = ''
        self.is_connection_error = False

    def _make_get_request(self, path, parameters=None, result_form='text'):
        return self._make_request('get', parameters, result_form=result_form)


    def _make_request(self, request_type: str, path: str, parameters: Optional[dict]=None, body: Optional[Any] = None,
                      result_form='text'):
        url = urljoin(':'.join([self.host, str(self.port)]), path)

        try:
            if request_type.lower() == 'get':
                response = requests.get(url, params=parameters)
            elif request_type.lower() == 'post':
                response = requests.post(url, params=parameters, data=body)
            else:
                raise requests.RequestException('Unsupported request type')

        except requests.ConnectionError as e:
            raise ConnectorException('Ошибка подключения: {}'.format(str(e)))
        except requests.Timeout as e:
            raise ConnectorException('Ошибка тайм-аута: {}'.format(str(e)))
        except requests.RequestException as e:
            raise ConnectorException('Ошибка запроса: {}'.format(str(e)))

        result = None

        if result_form == 'text':
            result = response.text
        return result


class Settings:

    def __init__(self, filename: str = ''):
        self.host = ''
        self.port = 0

        if not filename:
            filename = 'admin.cfg'
        self._settings_filename = filename

    def read(self):

        try:
            with open(self._settings_filename, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except FileNotFoundError as e:
            self.host = ''
            self.port = 0
        else:
            self.host = data['host']
            self.port = data['port']

    def write(self):

        if self.host or self.port:
            data = {'host': self.host, 'port': self.port}

            with open(self._settings_filename, 'w', encoding='utf-8') as f:
                json.dump(data, f)


class BaseModel:
    """
    Класс ...

    Модель содержит методы регистрации, удаления и оповещения
    наблюдателей.
    """

    def __init__(self, connector: Connector):

        self.connector = connector
        self.settings = Settings()
        self.is_connected = self.connector.is_connected
        self._observers = []  # список наблюдателей
        self.error_text = ''

        self.properties_changed = []

        self.read_settings()

    def connect(self):

        self.connector.connect()
        self.is_connected = self.connector.is_connected

    def disconnect(self):

        self.connector.disconnect()
        self.is_connected = self.connector.is_connected

    def write_settings(self):
        self.settings.write()

    def read_settings(self):
        self.settings.read()
        self.connector.host = self.settings.host
        self.connector.port = self.settings.port

        self.properties_changed.append('host')
        self.properties_changed.append('port')

    def add_observer(self, observer):
        self._observers.append(observer)

    def remove_observer(self, observer):
        self._observers.remove(observer)

    def notify_observers(self):
        for x in self._observers:
            x.model_changed()

        self.properties_changed = []

    @property
    def host(self):
        return self.connector.host

    @property
    def port(self):
        return self.connector.port

    @host.setter
    def host(self, value: str):
        self.connector.host = value
        self.settings.host = value

    @port.setter
    def port(self, value: str):
        self.connector.port = value
        self.settings.port = value

