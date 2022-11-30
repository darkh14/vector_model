"""
    Module that provides batch data loading. Contains classes that control loading and deleting data
        Classes:
            Package - class controls loading portion of data
            Loading - class controls process of loading data, loading packages, provides coherent loading

        Functions:
            delete_all_data - to delete all data according to filter
"""

from typing import Any, Optional
from _datetime import datetime

import db_processing.connectors.base_connector
from vm_logging.exceptions import LoadingProcessException
from .loading_engines import BaseEngine, get_engine_class
from .loading_types import LoadingTypes, LoadingStatuses

__all__ = ['Loading', 'delete_all_data']


class Package:
    """ Class of portion of loading data.
        Properties:
            _loading_id: str - id of loading object
            _id: str - id of current package (this object)
           _status: LoadingStatuses - status of current package
           _type: LoadingTypes - type of current package (full or increment), equal to loading type
           _start_date: datetime - date of beginning of loading package
           _end_date: datetime - date of ending of loading package
           _db_connector: - connector object to connect to db
           _number: int number of package in order
           _check_sum: str | int - checksum of loading data. to check data before loading
           _error: str - error text of loading
           _engine: engine of data loading, depends on service_name
           status - virtual property for _status
           id - virtual property for _id
           type - virtual property for _type
        Methods:
            initialize - to initialize new package
            check_before_initializing - checking fullness of package fields before initializing
            drop - deleting package from db
            load_data - main method for data loading
            delete_data - deleting previously loading data from db
            get_package_info - getting package info (type, status, dates etc.)
            set_status - for setting status with checking parameter
            _read_from_db - for reading package data from db
            write_to_db - for writing package data to db
            _check_data - for checking loading data array before loading
            _get_engine - for getting loading engine object
    """
    def __init__(self, loading_id: str, package_id: str,
                 db_connector: db_processing.connectors.base_connector.Connector,
                 package_parameters: Optional[dict[str, Any]]):
        self._loading_id: str = loading_id
        self._id: str = package_id

        self._status: LoadingStatuses = LoadingStatuses.NEW
        self._type: LoadingTypes = LoadingTypes.FULL

        self._start_date: Optional[datetime] = None
        self._end_date: Optional[datetime] = None

        self._db_connector = db_connector

        self._number: int = package_parameters['number'] if package_parameters else 0

        self._check_sum: str | int = package_parameters['check_sum'] if package_parameters else 0
        self._error: str = ''

        self._engine: BaseEngine = self._get_engine()

        self._read_from_db()

    def initialize(self) -> None:
        """ Initializing NEW package (with writing to db) """
        if self._status != LoadingStatuses.NEW:
            raise LoadingProcessException('loading id "{}", package id - "{}" is '
                                          'always initialized'.format(self._loading_id, self._id))

        self._status = LoadingStatuses.REGISTERED

        self.write_to_db()

    def check_before_initializing(self):
        """ Checking package fields fullness before initializing """
        if not self._id:
            raise LoadingProcessException('Package id is not defined')

        if not self._loading_id:
            raise LoadingProcessException('Loading id is not defined')

        if self._status != LoadingStatuses.NEW:
            raise LoadingProcessException('Loading id - "{}" is always initialized'.format(self._id))

        if not self._type:
            raise LoadingProcessException('Loading id - "{}" type is not defined'.format(self._id))

    def drop(self, need_to_delete_data: bool = False):
        """ For deleting package object from db. Possible to delete loaded data together.
            Parameters:
                need_to_delete_data: bool - if True - also delete data, else only package object
        """
        if need_to_delete_data:
            self._engine.delete_data(self._loading_id, self._id)

        self._db_connector.delete_lines('packages', {'loading_id': self._loading_id, 'id': self._id})
        self._status = LoadingStatuses.NEW

    def load_data(self, data: list[dict[str, Any]]) -> bool:
        """ For loading data.
            Parameters:
                data: list - array of data to load to db
        """
        if not data:
            raise LoadingProcessException('Data parameter is not set')

        if self._status != LoadingStatuses.REGISTERED:
            raise LoadingProcessException('Package status must be "registered". '
                                          'Real status is "{}"'.format(self.status))

        self._check_data(data)

        self._start_date = datetime.utcnow()
        self._end_date = None
        self.write_to_db()

        try:
            self._engine.load_data(data, self._loading_id, self._id, self._type)
        except Exception as ex:
            self._status = LoadingStatuses.ERROR
            self._error = str(ex)
            self.write_to_db()
            raise LoadingProcessException(str(ex))

        self._status = LoadingStatuses.LOADED
        self._end_date = datetime.utcnow()
        self.write_to_db()

        return True

    def delete_data(self):
        """ For deleting data from db. Don not delete package object from db. """
        self._engine.delete_data(self._loading_id, self._id)
        self._status = LoadingStatuses.DELETED
        self._start_date = None
        self._end_date = None
        self.write_to_db()

        return True

    def get_package_info(self):
        """ For getting package info (status, type, start date, end date, error text) """
        package_info = {'id': self._id,
                        'type': self._type.value,
                        'status': self._status.value,
                        'start_date': self._start_date.strftime('%d.%m.%Y %H:%M:%S') if self._start_date else None,
                        'end_date': self._end_date.strftime('%d.%m.%Y %H:%M:%S') if self._end_date else None,
                        'number': self._number,
                        'check_sum': self._check_sum,
                        'error': self._error}

        return package_info

    @property
    def status(self):
        """ _status field getter """
        return self._status

    @status.setter
    def status(self, value: LoadingStatuses | str):
        """ _status field setter """
        self.set_status(value)

    def set_status(self, status_parameter: LoadingStatuses | str) -> bool:
        """ For setting status with checking input parameter
            Parameters:
                status_parameter: str or LoadingStatus - status to set
        """
        if not status_parameter:
            raise LoadingProcessException('Status parameter is not defined')

        if isinstance(status_parameter, str):
            self._status = LoadingStatuses(status_parameter)
        else:
            self._status = status_parameter

        if self._status not in [LoadingStatuses.REGISTERED, LoadingStatuses.LOADED, LoadingStatuses.ERROR]:
            raise LoadingProcessException('Status "{}" is not supported. Statuses allowed to set are - "registered", '
                                          '"loaded", "error"'.format(self._status))

        if self._status == LoadingStatuses.ERROR:
            self._error = 'Error status was set directly'
        else:
            self._error = ''

        if self._status == LoadingStatuses.REGISTERED:
            self._start_date = None

        if self._status != LoadingStatuses.LOADED:
            self._end_date = None

        self.write_to_db()

        return True

    @property
    def id(self):
        """ _id filed getter """
        return self._id

    @property
    def type(self) -> LoadingTypes:
        """ _type field getter"""
        return self._type

    @type.setter
    def type(self, value: LoadingTypes) -> None:
        """ _type field setter """
        self._type = value

    def _read_from_db(self):
        """ For reading package object from db """
        package_from_db = self._db_connector.get_line('packages', {'loading_id': self._loading_id, 'id': self._id})

        if package_from_db:
            self._type = LoadingTypes(package_from_db['type'])
            self._status = LoadingStatuses(package_from_db['status'])

            self._check_sum = package_from_db['check_sum']

            self._start_date = package_from_db['start_date']
            self._end_date = package_from_db['end_date']

            self._number = package_from_db['number']

    def write_to_db(self):
        """ For writing package object to db """
        package_to_db = {'id': self._id,
                         'loading_id': self._loading_id,
                         'type': self._type.value,
                         'status': self._status.value,
                         'start_date': self._start_date,
                         'end_date': self._end_date,
                         'check_sum': self._check_sum,
                         'number': self._number,
                         'error': self._error}

        self._db_connector.set_line('packages', package_to_db, {'id': self._id, 'loading_id': self._loading_id})

    def _check_data(self, data: list[dict[str, Any]]):
        """ For checking data before loading
            Parameters:
                data: list - data array for checking
        """
        if not isinstance(data, list):
            raise LoadingProcessException('Data must be list like type')

        if len(data) != self._check_sum:
            raise LoadingProcessException('Data checksum is not right. Right checksum is {}'.format(self._check_sum))

    def _get_engine(self) -> BaseEngine:
        """ For getting engine object to load data
                Return: loading engine
        """
        return get_engine_class()(self._db_connector)


class Loading:
    """ Class of loading data process. It has packages in it.
        Properties:
            _id: str - id of current loading
            _type: LoadingTypes - type of current loading (full or increment).
                full loading deletes all of previous loading data before starting loading
                increment deletes only data matches the keys with previous loading data
            _status: LoadingStatuses - status of current loading
            _packages: list - array of packages in current loading
            _create_date: datetime - date of creation of current loading
            _start_date: datetime - date of starting loading of first package
            _end_date: datetime - date of finishing loading of last package
            _db_connector - connector object to work with db
            _number_of_packages: int - number of packages in current loading
            _error: str - error text of loading
        Methods:
            initialize - for initializing NEW loading object
            drop - for deleting loading object
            load_package - for loading transmitted package data
            delete_data - for deleting data of loading
            get_loading_info - gets info of current loading (id, status, type, create date, start date, end,
                date, error)
            set_status - for setting status of loading
            set_package_status - for setting status of transmitted package
            _check_input_parameters - check fields when __init__
            _check_before_initializing - check fullness of fields before initializing
            _get_loading_type_from_parameter - gets type of loading from str
            _read_from_db - for reading loading object from db
            _write_to_db - for writing loading object to db
            _get_package - for getting package of loading by package id
    """
    def __init__(self, loading_parameters: dict[str, Any],
                 db_connector: db_processing.connectors.base_connector.Connector):

        self._check_input_parameters(loading_parameters)

        self._id: str = loading_parameters['id']
        self._type: Optional[LoadingTypes] = self._get_loading_type_from_parameter(loading_parameters.get('type'))
        self._status: LoadingStatuses = LoadingStatuses.NEW
        self._packages: list[Package] = []

        self._create_date: Optional[datetime] = None
        self._start_date: Optional[datetime] = None
        self._end_date: Optional[datetime] = None

        self._db_connector = db_connector

        if loading_parameters.get('packages'):
            package_num = 1
            for package_parameters in loading_parameters['packages']:
                p_parameters = package_parameters
                p_parameters['type'] = self._type
                p_parameters['number'] = package_parameters.get('number') or package_num

                package = self._get_package(package_parameters['id'], p_parameters)
                self._packages.append(package)
                package_num += 1

        self._number_of_packages: int = len(self._packages)
        self._error = ''

        self._read_from_db()

        if self._number_of_packages != len(self._packages):
            raise LoadingProcessException('Some packages are missing. Initially number of packages is {} '
                                          'but real number is {}'.format(self._number_of_packages, len(self._packages)))

    def initialize(self) -> dict[str, Any]:
        """ For initializing NEW loading"""
        self._check_before_initializing()

        self._status = LoadingStatuses.REGISTERED
        self._create_date = datetime.utcnow()

        self._write_to_db(write_packages=False)

        for package in self._packages:
            package.type = self._type
            package.initialize()

        return {'loading': self.get_loading_info()}

    def drop(self, need_to_delete_data: bool = False) -> bool:
        """ For deleting loading object from db. Provides deleting previously loaded data together
            Parameters:
                need_to_delete_data - deletes previously loaded data if True
        """
        if self._status == LoadingStatuses.NEW:
            raise LoadingProcessException('Loading id - "{}" is not initialized'.format(self._id))

        for package in self._packages:
            package.drop(need_to_delete_data=need_to_delete_data)

        self._db_connector.delete_lines('loadings', {'id': self._id})

        self._status = LoadingStatuses.NEW
        self._create_date = None
        self._start_date = None
        self._end_date = None

        return True

    def load_package(self, package_parameters: dict[str, Any]) -> bool:
        """ For loading package data to db.
            Parameters:
                package_parameters: dict - contains id (package id) and data (data array for loading)
        """
        if not package_parameters:
            raise LoadingProcessException('Package parameters is not set')

        if not package_parameters.get('id'):
            raise LoadingProcessException('Package id is not set')

        if not package_parameters.get('data'):
            raise LoadingProcessException('Package data are not set')

        if self._status not in [LoadingStatuses.REGISTERED, LoadingStatuses.PARTIALLY_LOADED]:
            raise LoadingProcessException('Loading status must be "registered" or "partially loaded". '
                                          'Real status is "{}"'.format(self._status))

        first_package = not [package for package in self._packages if package.status != LoadingStatuses.REGISTERED]

        self._status = LoadingStatuses.IN_PROCESS

        if first_package and self._type == LoadingTypes.FULL:
            self._start_date = datetime.utcnow()
            self._end_date = None
            self._write_to_db()
            self._db_connector.delete_lines('raw_data')

        current_packages = [package for package in self._packages if package.id == package_parameters['id']]

        if not current_packages:
            raise LoadingProcessException('Package id "{}" do not belong '
                                          'to loading id "{}"'.format(package_parameters['id'], self._id))

        current_package = current_packages[0]

        try:
            current_package.load_data(package_parameters['data'])
        except LoadingProcessException as ex:
            self._error = str(ex)
            self._status = LoadingStatuses.ERROR
            self._write_to_db()
            raise ex

        full_loaded = not [package for package in self._packages if package.status != LoadingStatuses.LOADED]

        if full_loaded:
            self._status = LoadingStatuses.LOADED
            self._end_date = datetime.utcnow()
            self._write_to_db()
        else:
            self.set_status(LoadingStatuses.PARTIALLY_LOADED)

        return True

    def delete_data(self):
        """ For deleting data from db without deleting loading object """
        for package in self._packages:
            package.delete_data()

        self._status = LoadingStatuses.DELETED
        self._start_date = None
        self._end_date = None
        self._write_to_db()

        return True

    def get_loading_info(self):
        """ For getting loading info.
            Contains:
                id: - loading id,
                type: loading type,
                status: loading status,
                create_date: date of creation loading,
                start_date: date of beginning loading first package,
                end_date: date of finishing loading last package,
                number_of_packages: number of packages of current loading,
                error: text of error while loading
        """
        loading_info = {'id': self._id,
                        'type': self._type.value,
                        'status': self._status.value,
                        'create_date': self._create_date.strftime('%d.%m.%Y %H:%M:%S') if self._create_date else None,
                        'start_date': self._start_date.strftime('%d.%m.%Y %H:%M:%S') if self._start_date else None,
                        'end_date': self._end_date.strftime('%d.%m.%Y %H:%M:%S') if self._end_date else None,
                        'number_of_packages': self._number_of_packages,
                        'error': self._error}

        packages_info = [package.get_package_info() for package in self._packages]

        loading_info['packages'] = packages_info

        return loading_info

    def set_status(self, status_parameter: LoadingStatuses | str, set_for_packages: bool = False) -> bool:
        """ For setting loading status.
            Parameters:
                status_parameter: str | loadingStatuses - value of setting status
                set_for_packages: - bool - set this status for all packages in loading if True
        """
        if not status_parameter:
            raise LoadingProcessException('Status parameter is not defined')

        if isinstance(status_parameter, str):
            self._status = LoadingStatuses(status_parameter)
        else:
            self._status = status_parameter

        if self._status not in [LoadingStatuses.REGISTERED, LoadingStatuses.LOADED,
                                LoadingStatuses.PARTIALLY_LOADED, LoadingStatuses.ERROR]:
            raise LoadingProcessException('Status "{}" is not supported. Statuses allowed to set are - "registered", '
                                          '"loaded", "partially loaded", "error"'.format(self._status))

        if self._status == LoadingStatuses.ERROR:
            self._error = 'Error status was set directly'
        else:
            self._error = ''

        if self._status != LoadingStatuses.LOADED:
            self._end_date = None

        if self._status in [LoadingStatuses.NEW, LoadingStatuses.REGISTERED]:
            self._start_date = None

        if set_for_packages:
            for package in self._packages:
                package.set_status(status_parameter)

        self._write_to_db()

        return True

    def set_package_status(self, package_parameter: dict[str, Any]) -> bool:
        """ For setting status of transmitted package.
            parameters:
                package_parameter: dict - parameters of package that needs to set status
        """
        if not package_parameter:
            raise LoadingProcessException('Package parameter is not defined')

        package_id = package_parameter.get('id')

        if not package_id:
            raise LoadingProcessException('Package id is not defined')

        current_packages = [package for package in self._packages if package.id == package_id]

        if not current_packages:
            raise LoadingProcessException('Package id "{}" is not found'.format(package_id))

        current_package = current_packages[0]

        status_parameter = package_parameter.get('status')

        if not status_parameter:
            raise LoadingProcessException('Package status is not defined')

        if isinstance(status_parameter, str):
            current_status = LoadingStatuses(status_parameter)
        else:
            current_status = status_parameter

        current_package.status = current_status

        if current_status == LoadingStatuses.ERROR:
            self._status = LoadingStatuses.ERROR
            self._error = 'Error status was set directly'
        elif current_status == LoadingStatuses.REGISTERED:
            loaded_packages = [package for package in self._packages if package.status != LoadingStatuses.REGISTERED]
            if loaded_packages:
                self._status = LoadingStatuses.PARTIALLY_LOADED
            else:
                self._status = LoadingStatuses.REGISTERED
            self._error = ''
        elif current_status == LoadingStatuses.IN_PROCESS:
            self._status = LoadingStatuses.IN_PROCESS
            self._error = ''
        elif current_status == LoadingStatuses.LOADED:
            not_loaded_packages = [package for package in self._packages if package.status != LoadingStatuses.LOADED]
            if not_loaded_packages:
                self._status = LoadingStatuses.PARTIALLY_LOADED
            else:
                self._status = LoadingStatuses.LOADED
            self._error = ''

        self._write_to_db()

        return True

    @staticmethod
    def _check_input_parameters(loading_parameters: dict[str, Any]) -> None:
        """ For checking loading parameters while __init__
            loading_parameters: dict - parameters of initialization of loading
        """
        check_fields = ['id']

        error_fields = [field for field in check_fields if field not in loading_parameters]

        if error_fields:
            error_message = '. '.join(['Field "{}" is not found in loading parameters'.format(field)
                                       for field in error_fields])

            raise LoadingProcessException(error_message)

        if loading_parameters.get('packages'):
            if not isinstance(loading_parameters['packages'], list):
                raise LoadingProcessException('Loading packages must be list like object,' 
                                              'but really packages '
                                              'type is {}'.format(type(loading_parameters['packages'])))

    def _check_before_initializing(self):
        """ Check fullness of fields before initialization """
        if not self._id:
            raise LoadingProcessException('Loading id is not defined')

        if self._status != LoadingStatuses.NEW:
            raise LoadingProcessException('Loading id - "{}" is always initialized'.format(self._id))

        if not self._type:
            raise LoadingProcessException('Loading id - "{}" type is not defined'.format(self._id))

        for package in self._packages:
            package.check_before_initializing()

    @staticmethod
    def _get_loading_type_from_parameter(type_parameter: str) -> Optional[LoadingTypes]:
        """ Gets loading type from str """
        if not type_parameter:
            return None

        if type_parameter.upper() not in [el.name for el in LoadingTypes]:
            raise LoadingProcessException('Loading type "{}" is not supported'.format(type_parameter))

        return LoadingTypes(type_parameter)

    def _read_from_db(self) -> None:
        """ for reading loading object from db """
        loading_from_db = self._db_connector.get_line('loadings', {'id': self._id})

        if loading_from_db:
            self._type = self._get_loading_type_from_parameter(loading_from_db['type'])
            self._status = LoadingStatuses(loading_from_db['status'])

            self._create_date = loading_from_db['create_date']
            self._start_date = loading_from_db['start_date']
            self._end_date = loading_from_db['end_date']

            self._error = loading_from_db['error']

            self._number_of_packages = loading_from_db['number_of_packages']

            packages_from_db = self._db_connector.get_lines('packages', {'loading_id': self._id})

            self._packages = []

            for package_from_db in packages_from_db:
                package = self._get_package(package_from_db['id'])
                self._packages.append(package)

    def _write_to_db(self, write_packages: bool = True) -> None:
        """ For writing loading object to db """
        loading_to_db = {'id': self._id,
                         'type': self._type.value,
                         'status': self._status.value,
                         'create_date': self._create_date,
                         'start_date': self._start_date,
                         'end_date': self._end_date,
                         'number_of_packages': self._number_of_packages,
                         'error': self._error}

        self._db_connector.set_line('loadings', loading_to_db, {'id': self._id})

        if write_packages:
            for package in self._packages:
                package.write_to_db()

    def _get_package(self, package_id: str, package_parameters: Optional[dict[str, Any]] = None) -> Package:
        """ For getting required package object.
            Parameters:
                package_id: str - id of required package;
                package_parameters: - optional parameters, using, when we want to get NEW package
        """
        return Package(self._id, package_id, self._db_connector, package_parameters)


def delete_all_data(db_connector: db_processing.connectors.base_connector.Connector, data_filter: dict[str, Any]):
    """ Deletes all data according to filter.
        Parameters:
            db_connector: - connector object to work with db
            data_filter: dict - filter according to which data is deleted
    """
    db_connector.delete_lines('raw_data', data_filter)
    return True
