""" Module contains BackgroundJob class.To execute function in background,
    kill process, write and update job object.
    Classes:
        BackgroundJob
"""
from typing import Any, Optional
from datetime import datetime
from importlib import import_module
import sys
import subprocess
import os
import traceback
import copy

import db_processing.connectors.base_connector
from vm_logging.exceptions import BackgroundJobException, VMBaseException, DBConnectorException
from vm_logging.loggers import JobContextLoggerManager
from db_processing import get_connector
from id_generator import IdGenerator
from .job_types import JobStatuses
from vm_settings import get_var

__all__ = ['BackgroundJob']


class BackgroundJob:
    """ Class for executing functions in background, controlling execution and kill process if necessary """
    def __init__(self, job_id: str = '',  db_path: str = '', subprocess_mode: bool = False):
        self._subprocess_mode = subprocess_mode

        self._error: str = ''

        try:
            self._db_connector: Optional[db_processing.connectors.base_connector.Connector] = get_connector(db_path)
            self._status: JobStatuses = JobStatuses.NEW
        except DBConnectorException as db_ex:
            self._db_connector = None
            self._error = db_ex.message
            self._status: JobStatuses = JobStatuses.ERROR
        except Exception as ex:
            self._db_connector = None
            self._error = str(ex)
            self._status: JobStatuses = JobStatuses.ERROR

        self._id: str = job_id
        self._job_name: str = ''

        self._start_date: Optional[datetime] = None
        self._end_date: Optional[datetime] = None
        self._parameters: dict[str, Any] = {}
        self._result: Any = None
        self._pid: int = 0
        self._output: str = ''

        self._temp_name: str = ''
        self._temp_settings: dict[str, str] = self._get_temp_parameters_settings()

        self._initialize()

    def _initialize(self) -> None:
        if self._id:
            self._read_from_db()
        else:
            self._id = IdGenerator.get_random_id()
            self._write_to_db()

    def execute_function(self, func, wrapper_parameters: dict[str, Any]) -> dict[str, str]:
        if self._subprocess_mode:
            raise BackgroundJobException('For executing function in background "subprocess_mode" must be False ')

        module_name, function_name = func.__module__, func.__name__

        self._job_name = '.'.join([module_name, function_name])

        self._status = JobStatuses.REGISTERED
        self._start_date = datetime.utcnow()
        self._end_date = None
        self._pid = 0
        self._error = ''
        self._result = None

        self._parameters = wrapper_parameters

        self._write_parameters_to_temp()
        self._delete_temp_from_parameters()

        self._write_to_db()

        python_command, python_path = self._get_path_command()

        with JobContextLoggerManager(self._id, context_mode=True) as (f_out, f_err):

            job_process = subprocess.Popen([python_command,
                                            python_path,
                                            '-background_job',
                                            self._id,
                                            self._job_name,
                                            self._db_connector.db_path], stdout=f_out, stderr=f_err)

        self._pid = job_process.pid
        self._write_to_db()

        return {'pid': job_process.pid, 'description': 'background job "{}" - id "{}" '
                                                       'is started'.format(self._job_name, self._id)}

    def execute_in_subprocess(self) -> None:

        if self._status != JobStatuses.ERROR:

            try:
                self._execute_function()
            except VMBaseException as exc:
                self._error = str(exc)
                self._status = JobStatuses.ERROR
            except Exception:
                self._error = traceback.format_exc()
                self._status = JobStatuses.ERROR

        if self._error:
            sys.stderr.write(self._error)

            if self._temp_name:
                self._drop_temp()

            if self._db_connector:
                self._write_to_db()

    def _execute_function(self) -> Any:

        if not self._subprocess_mode:
            raise BackgroundJobException('For executing function in background "subprocess_mode" must be True ')

        if not self._job_name:
            raise BackgroundJobException('Job name is not set')

        if self._status != JobStatuses.REGISTERED:
            raise BackgroundJobException('Job status must be "registered". '
                                         'Current status is "{}"'.format(self._status.value))

        self._status = JobStatuses.IN_PROCESS
        self._start_date = datetime.utcnow()
        self._write_to_db()

        module_name, function_name = self._get_module_function_from_name()

        imported_module = import_module(module_name)
        imported_function = imported_module.__dict__[function_name]

        self._parameters['background_job'] = False
        self._parameters['job_id'] = self._id

        self._add_temp_to_parameters()

        print('{} - start background job ""{}" id "{}"'.format(self._start_date.strftime('%d.%m.%Y %H:%M:%S'),
                                                               self._job_name, self._id))
        result = imported_function(self._parameters)

        self._result = result

        self._delete_temp_from_parameters()

        self._status = JobStatuses.FINISHED
        self._end_date = datetime.utcnow()

        print('{} - finish background job ""{}" id "{}"'.format(self._end_date.strftime('%d.%m.%Y %H:%M:%S'),
                                                                self._job_name, self._id))

        log_manager = JobContextLoggerManager(self._id)

        out, err = log_manager.read_logs()

        self._output = out
        self._error = err

        self._write_to_db()

        return result

    def _get_path_command(self) -> [str, str]:

        venv_python = get_var('PYTHON_VENV_PATH')
        if not venv_python:
            python_command = 'python'

            if sys.platform == "linux" or sys.platform == "linux2":
                python_command = 'python3'
        else:
            python_command = venv_python

        package_path = self.__module__.split('.')[0]

        python_path = os.path.join(package_path, '../background_job_launcher.py')

        return python_command, python_path

    def _read_from_db(self) -> None:
        job_from_db = self._db_connector.get_line('background_jobs', {'id': self._id})
        if job_from_db:
            self._job_name = job_from_db['job_name']
            self._status = JobStatuses(job_from_db['status'])
            self._start_date = job_from_db['start_date']
            self._end_date = job_from_db['end_date']
            self._parameters = job_from_db['parameters']
            self._result = job_from_db['result']
            self._pid = job_from_db['pid']
            self._error = job_from_db['error']
            self._output = job_from_db['output']

    def _write_to_db(self) -> None:
        job_to_db = {
            'id': self._id,
            'job_name': self._job_name,
            'status': self._status.value,
            'start_date': self._start_date,
            'end_date': self._end_date,
            'parameters': self._parameters,
            'result': self._result,
            'pid': self._pid,
            'error': self._error,
            'output': self._output
                }

        self._db_connector.set_line('background_jobs', job_to_db, {'id': self._id})

    def _write_parameters_to_temp(self) -> None:

        temp_parameters_path = self._temp_settings.get(self._parameters.get('request_type'))

        if temp_parameters_path:

            self._temp_name = 'temp_' + self._id

            temp_data = None
            path_list = temp_parameters_path.split('.')

            for path_el in path_list:
                temp_data = self._parameters[path_el] if not temp_data else temp_data[path_el]

            self._db_connector.set_lines(self._temp_name, temp_data)

    def _add_temp_to_parameters(self) -> None:

        temp_parameters_path = self._temp_settings.get(self._parameters.get('request_type'))

        if temp_parameters_path:

            self._temp_name = 'temp_' + self._id

            path_list = temp_parameters_path.split('.')

            temp_data = self._parameters

            for path_el in path_list[:-1]:
                temp_data = temp_data[path_el]

            temp_data[path_list[-1]] = self._get_parameters_from_temp()

            self._drop_temp()

    def _delete_temp_from_parameters(self) -> None:

        temp_parameters_path = self._temp_settings.get(self._parameters.get('request_type'))

        if temp_parameters_path:

            path_list = temp_parameters_path.split('.')

            temp_data = self._parameters

            for path_el in path_list[:-1]:
                temp_data = temp_data[path_el]

            temp_data[path_list[-1]] = None

    def _get_parameters_from_temp(self) -> list[dict[str, Any]]:
        return self._db_connector.get_lines(self._temp_name)

    def _drop_temp(self) -> None:
        if self._temp_name:
            self._db_connector.delete_lines(self._temp_name)
            self._temp_name = ''

    def _get_module_function_from_name(self) -> [str, str]:
        name_list = self._job_name.split('.')
        module_name = '.'.join(name_list[:-1])
        function_name = name_list[-1]

        return module_name, function_name

    @staticmethod
    def _get_temp_parameters_settings() -> dict[str, str]:
        return {'data_load_package': 'loading.package.data'}

    @property
    def job_name(self) -> str:
        return self._job_name

    @job_name.setter
    def job_name(self, value: str) -> None:
        self._job_name = value
