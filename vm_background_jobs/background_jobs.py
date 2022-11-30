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
import psutil


import db_processing.connectors.base_connector
from vm_logging.exceptions import BackgroundJobException, VMBaseException, DBConnectorException
from vm_logging.loggers import JobContextLoggerManager
from db_processing import get_connector
from id_generator import IdGenerator
from .job_types import JobStatuses
from vm_settings import get_var

__all__ = ['BackgroundJob']


class BackgroundJob:
    """ Class for executing functions in background, controlling execution function.
        Also provides getting jobs info from db and deleting job with killing process if necessary
            Methods:
                execute_function - for executing functions; in initial mode (not in subprocess mode)
                execute_in_subprocess - for executing functions; in subprocess mode; executes from
                    execute_function method
                delete - for deleting job object from db and killing process if necessary
                get_jobs_info - class method for getting jobs info
                _initialize - for reading job data from db or writing new job to db
                _execute_function - executes function in subprocess mode in try-except block
                _get_path_command - returns path to launcher script file and python and command (perhaps in venv)
                _read_from_db - for reading job parameters from db
                _write_to_db  - writes job parameters to db
                _write_parameters_to_temp - writes data from parameters to temp db
                _add_temp_to_parameters - adds temp data tu parameters for function
                _delete_temp_from_parameters - for deleting temp data from parameters for function
                _get_parameters_from_temp - for getting temp data from db
                _drop_temp - drops temp collection for current job
                _get_module_function_from_name - gets module name and function name from job name
                _get_temp_parameters_settings - defines data in function parameters to be set in temp

                job_name - property for _job_name
    """
    def __init__(self, job_id: str = '',  db_path: str = '', subprocess_mode: bool = False):
        """ Fields:
                _id - unique job id
                _job_name - name of job - match to name executing function
                _subprocess_mode - if True - background_job executes in subprocess, else in initial mode
                _error - contains error text of executing background job
                _db_connector - object for working with db
                _status - status of job executing - "new", "registered", "in_process", "finished", "error"
                _start_date - date of beginning of executing job
                _end_date - date of ending of executing job
                _parameters - parameters of executing function
                _result - result of executing function
                _pid - executed process id
                _output - info from stdout of executing function
                _temp_name - name of temp collection. filled when collection contains data
                _temp_settings - defines what data saves in temp collection
        """
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
        """ For reading job fields from db or writing to db if new job """
        if self._id:
            temp_id = self._id
            self._read_from_db()
            if not self._id:
                raise BackgroundJobException('background job with id "{}" is not found'.format(temp_id))
        else:
            self._id = IdGenerator.get_random_id()
            self._write_to_db()

    def execute_function(self, func, wrapper_parameters: dict[str, Any]) -> dict[str, str]:
        """ For executing function in initial mode
                Parameters:
                    func - function object to execute
                    wrapper_parameters - parameters to be trs=ansmitted to function
                Returns:
                    description of execiting function
        """

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
            print('test_test')
            job_process = subprocess.Popen([python_command,
                                            python_path,
                                            '-background_job',
                                            self._id,
                                            self._job_name,
                                            self._db_connector.db_path], stdout=f_out, stderr=f_err)
            print('job-process - {}'.format(job_process))

        self._pid = job_process.pid
        self._write_to_db()

        return {'pid': job_process.pid, 'description': 'background job "{}" - id "{}" '
                                                       'is started'.format(self._job_name, self._id)}

    def execute_in_subprocess(self) -> None:
        """ For execution unction in subprocess mode. Without parameters.
            All parameters in self._parameters which is read from db
        """
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

            log_manager = JobContextLoggerManager(self._id)
            out, err = log_manager.read_logs()

            self._output = out
            self._error = err

            if self._db_connector:
                self._write_to_db()

    def delete(self):
        """ For deleting job from db. Also kills process, drops temp collections and clears logs"""
        if self._pid and self._status == JobStatuses.IN_PROCESS:
            try:
                process = psutil.Process(self._pid)
                process.terminate()
                self._pid = 0
            except psutil.Error as ex:
                self._error = str(ex)

        self._drop_temp()
        job_logger = JobContextLoggerManager(self._id, context_mode=False)
        job_logger.clear_old_logs()
        self._db_connector.delete_lines('background_jobs', {'id': self._id})

    @classmethod
    def get_jobs_info(cls, job_filter:  dict[str, Any], db_path: str) -> list[dict[str, Any]]:
        """ Class method for getting jobs inf. Can get info of many job according to filter """
        db_connector = get_connector(db_path)
        job_list = db_connector.get_lines('background_jobs', job_filter)

        fields = ['id', 'name', 'status', 'pid', 'error', 'output']

        result = []

        for el in job_list:
            c_job = {key: value for key, value in el.items() if key in fields}
            c_job['start_date'] = el['start_date'].strftime('%d.%m.%Y %H:%M:%S')
            c_job['end_date'] = el['end_date'].strftime('%d.%m.%Y %H:%M:%S')

            result.append(c_job)

        return result

    def _execute_function(self) -> Any:
        """ Executes function in subprocess inside try-except block """
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
        """ Gets path to launcher script and pythin command. Python command may be
            in venv (saves in PYTHON_VENV_PATH)
        """
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
        """ For reading job fields from db """

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
        else:
            self._id = ''

    def _write_to_db(self) -> None:
        """ For writing job fields to db """
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
        """ For writing temp data to temp collection """
        temp_parameters_path = self._temp_settings.get(self._parameters.get('request_type'))

        if temp_parameters_path:

            self._temp_name = 'temp_' + self._id

            temp_data = None
            path_list = temp_parameters_path.split('.')

            for path_el in path_list:
                temp_data = self._parameters[path_el] if not temp_data else temp_data[path_el]

            self._db_connector.set_lines(self._temp_name, temp_data)

    def _add_temp_to_parameters(self) -> None:
        """ Adds temp data to function parameters """
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
        """ Removes temp data from function parameters """
        temp_parameters_path = self._temp_settings.get(self._parameters.get('request_type'))

        if temp_parameters_path:

            path_list = temp_parameters_path.split('.')

            temp_data = self._parameters

            for path_el in path_list[:-1]:
                temp_data = temp_data[path_el]

            temp_data[path_list[-1]] = None

    def _get_parameters_from_temp(self) -> list[dict[str, Any]]:
        """ Gets emp data from db """
        return self._db_connector.get_lines(self._temp_name)

    def _drop_temp(self) -> None:
        """ Drops temp collection """
        if self._temp_name:
            self._db_connector.delete_lines(self._temp_name)
            self._temp_name = ''

    def _get_module_function_from_name(self) -> [str, str]:
        """ Gets module name and function name to execute from self._job_name """

        name_list = self._job_name.split('.')
        module_name = '.'.join(name_list[:-1])
        function_name = name_list[-1]

        return module_name, function_name

    @staticmethod
    def _get_temp_parameters_settings() -> dict[str, str]:
        """ Define what data will be saved to temp """
        return {'data_load_package': 'loading.package.data'}

    @property
    def job_name(self) -> str:
        return self._job_name

    @job_name.setter
    def job_name(self, value: str) -> None:
        self._job_name = value
