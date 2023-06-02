"""
    Module that provides batch data loading. Contains classes that control loading and deleting data
        Classes:
            Package - class controls loading portion of data
            Loading - class controls process of loading data, loading packages, provides coherent loading

        Functions:
            delete_all_data - to delete all data according to filter
"""

from typing import Any, Optional, Callable
from _datetime import datetime

from vm_logging.exceptions import LoadingProcessException, ParametersFormatError
from db_processing import get_connector
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
    def __init__(self, loading_id: str, package_id: str, number: int = 0, loading_type: str = '',
                 check_sum: Optional[str | int] = None) -> None:
        """
        Defines all inner variable of package. Then variables are read from db if it is not new package
        :param loading_id: id of loading of current package
        :param package_id: id of current package
        :param loading_type: type of loading (full or increment) if new package
        :param number: number of package if new package
        :param check_sum: checksum of package if new packaage
        """
        self._loading_id: str = loading_id
        self._id: str = package_id

        self._status: LoadingStatuses = LoadingStatuses.NEW
        self._type: LoadingTypes = _get_loading_type_from_parameter(loading_type)

        self._start_date: Optional[datetime] = None
        self._end_date: Optional[datetime] = None

        self._db_connector = get_connector()

        self._number: int = number

        self._check_sum: str | int = check_sum or ''
        self._error: str = ''

        self._engine: BaseEngine = self._get_engine()

        self._read_from_db()

    def initialize(self) -> None:
        """ Initializing NEW package (with writing to db)
        """
        if self._status != LoadingStatuses.NEW:
            raise LoadingProcessException('loading id "{}", package id - "{}" is '
                                          'always initialized'.format(self._loading_id, self._id))

        self._status = LoadingStatuses.REGISTERED

        self.write_to_db()

    def check_before_initializing(self) -> None:
        """ Checking package fields fullness before initializing """
        if not self._id:
            raise LoadingProcessException('Package id is not defined')

        if not self._loading_id:
            raise LoadingProcessException('Loading id is not defined')

        if self._status != LoadingStatuses.NEW:
            raise LoadingProcessException('Loading id - "{}" is always initialized'.format(self._id))

        if not self._type:
            raise LoadingProcessException('Loading id - "{}" type is not defined'.format(self._id))

    def drop(self, need_to_delete_data: bool = False) -> None:
        """ For deleting package object from db. Possible to delete loaded data together.
        :param need_to_delete_data: if True - also delete data, else only package object
        """

        if need_to_delete_data:
            self._engine.delete_data(self._loading_id, self._id)

        self._db_connector.delete_lines('packages', {'loading_id': self._loading_id, 'id': self._id})
        self._status = LoadingStatuses.NEW

    def load_data(self, data: list[dict[str, Any]], job_id: str = '') -> bool:
        """ For loading data
        :param data: array of data to load to db
        :param job_id: id of loading job if loading in background, else ''
        :return result of package data loading
        """

        if job_id:
            if self._status != LoadingStatuses.PRE_STARTED:
                raise LoadingProcessException('Package status must be "pre started". '
                                              'Real status is "{}"'.format(self.status))
        else:
            if self._status != LoadingStatuses.REGISTERED:
                raise LoadingProcessException('Package status must be "registered". '
                                              'Real status is "{}"'.format(self.status))

        self._check_data(data)

        if self._status == LoadingStatuses.REGISTERED:
            self._start_date = datetime.utcnow()
            self._end_date = None

        self._status = LoadingStatuses.IN_PROCESS

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

    def delete_data(self) -> bool:
        """ For deleting data from db. Don not delete package object from db.
        :return: result of package data deleting. True if successful
        """
        self._engine.delete_data(self._loading_id, self._id)
        self._status = LoadingStatuses.DELETED
        self._start_date = None
        self._end_date = None
        self.write_to_db()

        return True

    def get_package_info(self) -> dict[str, Any]:
        """ For getting package info (status, type, start date, end date, error text)
        :return: dict of package info
        """
        package_info = {'id': self._id,
                        'type': self._type.value,
                        'status': self._status.value,
                        'start_date': self._start_date.strftime('%d.%m.%Y %H:%M:%S') if self._start_date else None,
                        'end_date': self._end_date.strftime('%d.%m.%Y %H:%M:%S') if self._end_date else None,
                        'number': self._number,
                        'check_sum': self._check_sum,
                        'error': self._error}

        return package_info

    def do_before_load(self, parameters: dict[str, Any]):

        if self._status != LoadingStatuses.REGISTERED:
            raise LoadingProcessException('Package status must be "registered". '
                                          'Real status is "{}"'.format(self._status))

        self._status = LoadingStatuses.PRE_STARTED
        self._start_date = datetime.utcnow()
        self._end_date = None

        self.write_to_db()

    @property
    def status(self) -> LoadingStatuses:
        """ _status field getter
        :return: value of inner variable _status
        """
        return self._status

    @status.setter
    def status(self, value: LoadingStatuses | str) -> None:
        """ _status field setter
        :param value: value of status to set (LoadingStatuses and str are supported)
        """
        self.set_status(value)

    def set_status(self, status_parameter: LoadingStatuses | str) -> bool:
        """ For setting status with checking input parameter
        :param status_parameter: status to set
        :return result of status setting - true if successful
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

        if self._status == LoadingStatuses.IN_PROCESS:
            self._start_date = datetime.utcnow()

        if self._status != LoadingStatuses.LOADED:
            self._end_date = None
        else:
            self._end_date = datetime.utcnow()
            if not self._start_date:
                self._start_date = datetime.utcnow()

        self.write_to_db()

        return True

    @property
    def id(self) -> str:
        """ _id filed getter. Cannot be set outside __init__
        :return: value of inner variable _id
        """
        return self._id

    @property
    def type(self) -> LoadingTypes:
        """ _type field getter
        :return: value of inner variable _type
        """
        return self._type

    @type.setter
    def type(self, value: LoadingTypes) -> None:
        """ _type field setter """
        self._type = value

    def _read_from_db(self) -> None:
        """ For reading package object from db """
        package_from_db = self._db_connector.get_line('packages', {'loading_id': self._loading_id, 'id': self._id})

        if package_from_db:
            self._type = _get_loading_type_from_parameter(package_from_db['type'])
            self._status = LoadingStatuses(package_from_db['status'])

            self._check_sum = package_from_db['check_sum']

            self._start_date = package_from_db['start_date']
            self._end_date = package_from_db['end_date']

            self._number = package_from_db['number']

    def write_to_db(self) -> None:
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

    def _check_data(self, data: list[dict[str, Any]]) -> None:
        """ For checking data before loading. Raises LoadingProcessException exception if check is failed
        :param data: data array for checking
        """

        if len(data) != self._check_sum:
            raise LoadingProcessException('Data checksum is not right. Right checksum is {}'.format(self._check_sum))

    # noinspection PyMethodMayBeStatic
    def _get_engine(self) -> BaseEngine:
        """ For getting engine object to load data
        :return loading engine
        """
        return get_engine_class()()


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
            _read_from_db - for reading loading object from db
            _write_to_db - for writing loading object to db
            _get_package - for getting package of loading by package id
    """

    def __init__(self, loading_id: str, loading_type: str = '', packages: Optional[list] = None) -> None:
        """
        Defines all inner variable of loading. Then variables are read from db if it is not new loading
        :param loading_id: id of loading
        :param loading_type: type of loading, full or increment
        :param packages: parameters of packages in loading
        """

        self._id: str = loading_id
        self._type: Optional[LoadingTypes] = _get_loading_type_from_parameter(loading_type)
        self._status: LoadingStatuses = LoadingStatuses.NEW
        self._packages: list[Package] = []

        self._create_date: Optional[datetime] = None
        self._start_date: Optional[datetime] = None
        self._end_date: Optional[datetime] = None

        self._db_connector = get_connector()

        if packages:

            for package_num, p_package in enumerate(packages):

                match p_package:
                    case {'id': str(package_id), 'number': int(p_number), 'check_sum': int(check_sum) | str(check_sum)}:

                        package = Package(self._id, package_id,
                                          loading_type=str(self._type.value),
                                          number=p_number,
                                          check_sum=check_sum)
                        self._packages.append(package)
                    case _:
                        raise ParametersFormatError('Wrong package parameters ' +
                                                    'format in package number {}! '.format(package_num) +
                                                    'Check "packages" parameter')

        self._number_of_packages: int = len(self._packages)
        self._error = ''

        self._read_from_db()

        if self._number_of_packages != len(self._packages):
            raise LoadingProcessException('Some packages are missing. Initially number of packages is {} '
                                          'but real number is {}'.format(self._number_of_packages, len(self._packages)))

    def initialize(self) -> dict[str, Any]:
        """ For initializing NEW loading
        :return: information of loading
        """
        self._check_before_initializing()

        self._status = LoadingStatuses.REGISTERED
        self._create_date = datetime.utcnow()

        self._write_to_db(write_packages=False)

        for package in self._packages:
            package.type = self._type
            package.initialize()

        return self.get_loading_info()

    def drop(self, need_to_delete_data: bool = False) -> dict[str, Any]:
        """ For deleting loading object from db. Provides deleting previously loaded data together.
        Also deletes packages of loading
        :param need_to_delete_data: deletes previously loaded data if True
        :return result of dropping loading
        """

        loading_info = self.get_loading_info()

        if self._status == LoadingStatuses.NEW:
            raise LoadingProcessException('Loading id - "{}" is not initialized'.format(self._id))

        for package in self._packages:
            package.drop(need_to_delete_data=need_to_delete_data)

        self._db_connector.delete_lines('loadings', {'id': self._id})

        self._status = LoadingStatuses.NEW
        self._create_date = None
        self._start_date = None
        self._end_date = None

        return loading_info

    def load_package(self, package_parameters: dict[str, Any], job_id: str = '') -> bool:
        """ For loading package data to db
        :param package_parameters: contains id (package id) and data (data array for loading)
        :param job_id: background job id if loading in background
        :return result of loading package data. true if successful
        """

        match package_parameters:
            case {'id': str(package_id), 'data': list(package_data)} if package_id and package_data:
                pass
            case _:
                raise ParametersFormatError('Wrong package parameters format!')

        if self._status not in [LoadingStatuses.REGISTERED, LoadingStatuses.PRE_STARTED,
                                LoadingStatuses.PARTIALLY_LOADED]:
            raise LoadingProcessException('Loading status must be "registered", "pre started" or "partially loaded". '
                                          'Real status is "{}"'.format(self._status))

        first_package = not [package for package in self._packages if package.status not in [LoadingStatuses.REGISTERED,
                                                                                        LoadingStatuses.PRE_STARTED]]

        if first_package and self._type == LoadingTypes.FULL:
            if self._status != LoadingStatuses.PRE_STARTED:
                self._start_date = datetime.utcnow()
                self._end_date = None
            self._db_connector.delete_lines('raw_data')

        self._status = LoadingStatuses.IN_PROCESS

        self._write_to_db()

        current_packages = [package for package in self._packages if package.id == package_id]

        if not current_packages:
            raise LoadingProcessException('Package id "{}" do not belong '
                                          'to loading id "{}"'.format(package_id, self._id))

        current_package = current_packages[0]

        try:
            current_package.load_data(package_data, job_id=job_id)
        except LoadingProcessException as ex:
            self._error = str(ex)
            self._status = LoadingStatuses.ERROR
            self._write_to_db()
            raise ex

        full_loaded = not [package for package in self._packages if package.status != LoadingStatuses.LOADED]

        if full_loaded:
            self._status = LoadingStatuses.LOADED
            self._end_date = datetime.utcnow()
        else:
            self.set_status(LoadingStatuses.PARTIALLY_LOADED, need_to_write_to_db=False)

        self._write_to_db()

        return True

    def delete_data(self) -> bool:
        """ For deleting data from db without deleting loading object
        :return: result of deleting data. True if successful
        """
        for package in self._packages:
            package.delete_data()

        self._status = LoadingStatuses.DELETED
        self._start_date = None
        self._end_date = None
        self._write_to_db()

        return True

    def get_loading_info(self) -> dict[str, Any]:
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
                :return: dict of loading info
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

        return {'loading': loading_info}

    def set_status(self, status_parameter: LoadingStatuses | str, set_for_packages: bool = False,
                   from_outside: bool = False, need_to_write_to_db: bool = True) -> bool:
        """ For setting loading status
        :param status_parameter: value of setting status
        :param set_for_packages: set this status for all packages in loading if True.
        :param from_outside: True when status is set directly from http request, else False.
        :param need_to_write_to_db: False when status not need to be written to DB
        :return result of setting status, True if successful
        """

        if not status_parameter:
            raise LoadingProcessException('Status parameter is not defined')

        if isinstance(status_parameter, str):
            self._status = LoadingStatuses(status_parameter)
        else:
            self._status = status_parameter

        supported_statuses = [LoadingStatuses.REGISTERED, LoadingStatuses.LOADED, LoadingStatuses.ERROR]

        if not from_outside:
            supported_statuses.append(LoadingStatuses.PARTIALLY_LOADED)

        if self._status not in supported_statuses:
            raise LoadingProcessException('Status "{}" is not supported. Statuses allowed to set are - '
                                          '{}'.format(self._status,  ', '.join(['"{}"'.format(st) for st
                                                                                in supported_statuses])))
        if self._status == LoadingStatuses.ERROR:
            self._error = 'Error status was set directly'
        else:
            self._error = ''

        if self._status != LoadingStatuses.LOADED:
            self._end_date = None
        else:
            self._end_date = datetime.utcnow()
            if not self._start_date:
                self._start_date = datetime.utcnow()

        if self._status in [LoadingStatuses.NEW, LoadingStatuses.REGISTERED]:
            self._start_date = None
            self._end_date = None

        if set_for_packages:
            for package in self._packages:
                package.set_status(status_parameter)

        if need_to_write_to_db:
            self._write_to_db()

        return True

    def set_package_status(self, package_parameter: dict[str, Any]) -> bool:
        """ For setting status of transmitted package
        :param package_parameter: parameters of package that needs to set status
        :return result of setting package status, True if successful
        """

        match package_parameter:
            case {'id': package_id, 'status': status_parameter} if package_id and status_parameter:
                pass
            case  _:
                raise ParametersFormatError('Wrong package parameters format!')

        current_packages = [package for package in self._packages if package.id == package_id]

        if not current_packages:
            raise LoadingProcessException('Package id "{}" is not found'.format(package_id))

        current_package = current_packages[0]

        if isinstance(status_parameter, str):
            current_status = LoadingStatuses(status_parameter)
        else:
            current_status = status_parameter

        supported_statuses = [LoadingStatuses.REGISTERED, LoadingStatuses.LOADED, LoadingStatuses.IN_PROCESS,
                              LoadingStatuses.ERROR]

        if current_status not in supported_statuses:
            raise LoadingProcessException('Status "{}" is not supported. Statuses allowed to set are - '
                                          '{}'.format(self._status,  ', '.join(['"{}"'.format(st) for st
                                                                                in supported_statuses])))

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

    def get_action_before_background_job(self, job_name: str, parameters: dict[str, Any]) -> Optional[Callable]:
        result = None
        if job_name == 'data_load_package':
            result = self.do_before_load

        return result

    def do_before_load(self, parameters: dict[str, Any]):

        if self._status not in [LoadingStatuses.REGISTERED, LoadingStatuses.PARTIALLY_LOADED]:
            raise LoadingProcessException('Loading status must be "registered", "pre started" or "partially loaded". '
                                          'Real status is "{}"'.format(self._status))

        if self._status == LoadingStatuses.REGISTERED:
            self._status = LoadingStatuses.PRE_STARTED
            self._start_date = datetime.utcnow()
            self._end_date = None

        match parameters:
            case {'loading': {'package': {'id': str(package_id)}}} if package_id:
                pass
            case _:
                raise ParametersFormatError('Wrong loading parameters format!')

        current_packages = [package for package in self._packages if package.id == package_id]

        if not current_packages:
            raise LoadingProcessException('Package id "{}" do not belong '
                                          'to loading id "{}"'.format(package_id, self._id))

        current_package = current_packages[0]

        current_package.do_before_load(parameters['loading']['package'])

        self._write_to_db()

    def _check_before_initializing(self) -> None:
        """ Check fullness of fields before initialization. Raises LoadingProcessException if checking is failed """
        if not self._id:
            raise LoadingProcessException('Loading id is not defined')

        if self._status != LoadingStatuses.NEW:
            raise LoadingProcessException('Loading id - "{}" is always initialized'.format(self._id))

        if not self._type:
            raise LoadingProcessException('Loading id - "{}" type is not defined'.format(self._id))

        for package in self._packages:
            package.check_before_initializing()

    def _read_from_db(self) -> None:
        """ for reading loading object from db """
        loading_from_db = self._db_connector.get_line('loadings', {'id': self._id})

        if loading_from_db:
            self._type = _get_loading_type_from_parameter(loading_from_db['type'])
            self._status = LoadingStatuses(loading_from_db['status'])

            self._create_date = loading_from_db['create_date']
            self._start_date = loading_from_db['start_date']
            self._end_date = loading_from_db['end_date']

            self._error = loading_from_db['error']

            self._number_of_packages = loading_from_db['number_of_packages']

            packages_from_db = self._db_connector.get_lines('packages', {'loading_id': self._id})

            self._packages = []

            for package_from_db in packages_from_db:
                package = Package(self._id, package_from_db['id'])
                self._packages.append(package)

    def _write_to_db(self, write_packages: bool = True) -> None:
        """ For writing loading object to db
        :param write_packages: also writes packages to db if True
        """
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


def delete_all_data(data_filter: dict[str, Any]) -> bool:
    """ Deletes all data according to filter
    :param data_filter: filter according to which data is deleted
    :return result of deleting data, True if successful
    """
    db_connector = get_connector()
    db_connector.delete_lines('raw_data', data_filter)
    return True


def _get_loading_type_from_parameter(type_parameter: str) -> Optional[LoadingTypes]:
    """ Gets loading type from str
    :param type_parameter: string of loading type.
    :return: loading type (LoadingTypes) object
    """
    if not type_parameter:
        return None

    if type_parameter.upper() not in [el.name for el in LoadingTypes]:
        raise LoadingProcessException('Loading type "{}" is not supported'.format(type_parameter))

    return LoadingTypes(type_parameter)
