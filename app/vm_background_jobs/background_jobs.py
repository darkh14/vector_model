""" Module contains BackgroundJob class.To execute function in background,
    kill process, write and update job object.
    Classes:
        BackgroundJob
"""
from typing import Any, Optional, Callable
from datetime import datetime
from importlib import import_module
import os
import sys
import subprocess
import traceback
import psutil
import pickle

from vm_logging.exceptions import BackgroundJobException, VMBaseException, DBConnectorException
from vm_logging.loggers import JobContextLoggerManager
from db_processing import get_connector
from id_generator import IdGenerator
from .job_types import JobStatuses
import api_types as general_api_types
from db_processing.connectors.base_connector import Connector

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
    def __init__(self, job_id: str = '',  subprocess_mode: bool = False) -> None:
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
                # _temp_settings - defines what data saves in temp collection
                :param job_id: id of current job
                :param subprocess_mode: True if job launched in subprocess, else False
        """
        self._subprocess_mode = subprocess_mode

        self._error: str = ''

        try:
            self._db_connector: Optional[Connector] = get_connector()
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
        self._parameters: Optional[dict[str, Any]]
        self._pid: int = 0
        self._output: str = ''

        self._parameters: [dict[str, Any]] = {}
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

    def execute_function(self, func, args: tuple[Any],
                         kwargs: dict[str, Any]) -> general_api_types.BackgroundJobResponse:
        """ For executing function in initial mode
                Parameters:
                    func - function object to execute
                    args, kwargs - parameters to be transmitted to function
                Returns:
                    description of executing function.
            :param func: function to execute in background mode
            :param args: position parameters executing function
            :param kwargs: keyword parameters executing function
            :return:decorated result of launching in background mode
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

        self._parameters = {'args': args, 'kwargs': kwargs}

        self._write_to_db()

        self._do_action_before_job(func, args, kwargs)

        python_path, python_command = self._get_path_command()

        with JobContextLoggerManager(self._id, context_mode=True) as (f_out, f_err):
            job_process = subprocess.Popen([python_path,
                                            python_command,
                                            '-background_job',
                                            self._id,
                                            self._job_name,
                                            self._db_connector.db_name], stdout=f_out, stderr=f_err)

        self._pid = job_process.pid
        self._write_to_db()

        result = {'pid': job_process.pid,
                 'mode': general_api_types.ExecutionModes.BACKGROUND,
                 'description': 'background job "{}" - id "{}" is started'.format(self._job_name, self._id)}

        return general_api_types.BackgroundJobResponse.model_validate(result)

    def execute_in_subprocess(self) -> None:
        """ For execution unction in subprocess mode. Without parameters.
            All parameters in self._parameters which is read from db
        """
        if self._status != JobStatuses.ERROR:

            # noinspection PyBroadException
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

            log_manager = JobContextLoggerManager(self._id)
            out, err = log_manager.read_logs()

            self._output = out
            self._error = err

            if self._db_connector:
                self._write_to_db()

    def delete(self) -> None:
        """ For deleting job from db. Also kills process, drops temp collections and clears logs"""
        self._kill_job_process()

        job_logger = JobContextLoggerManager(self._id, context_mode=False)
        job_logger.clear_old_logs()
        self._db_connector.delete_lines('background_jobs', {'id': self._id})

    def set_interrupted(self) -> None:
        """ Interrupts background job, kills job process, set required status, drop additional job data """
        self._kill_job_process()

        job_logger = JobContextLoggerManager(self._id, context_mode=False)
        job_logger.clear_old_logs()

        self._status = JobStatuses.INTERRUPTED
        self._end_date = datetime.now()

        self._output = 'Interrupted'

        self._write_to_db()

    def _kill_job_process(self) -> None:
        """ kills job process when deleting job or job is interrupted """
        if self._pid and self._status == JobStatuses.IN_PROCESS:
            try:
                process = psutil.Process(self._pid)
                process.terminate()
                self._pid = 0
            except psutil.NoSuchProcess:
                self._error = 'No such process pid {}'.format(self._pid)
            except psutil.Error as ex:
                self._error = 'Process termination error. {}'.format(str(ex))

    @classmethod
    def get_jobs_info(cls, job_filter:  Optional[dict[str, Any]] = None) -> list[dict[str, Any]]:
        """ Class method for getting jobs inf. Can get info of many job according to filter.
        :param job_filter: filter to find required jobs
        :return: dict of jobs information
        """
        db_connector = get_connector()
        job_list = db_connector.get_lines('background_jobs', job_filter)

        fields = ['id', 'job_name', 'status', 'pid', 'error', 'output', 'start_date', 'end_date']

        result = []

        for el in job_list:
            c_job = {key: value for key, value in el.items() if key in fields}

            result.append(c_job)

        return result

    @staticmethod
    def _do_action_before_job(func: Callable, args: tuple[Any], kwargs: dict[str, Any]) -> None:
        """
        Executes action (special function before starting background job)
        @param func: function which will be executed in background
        @param args: position arguments of function
        @param kwargs: keyword arguments of function
        @return: None
        """
        # noinspection PyUnresolvedReferences
        action_generator = func.__globals__.get('get_action_before_background_job')

        if action_generator:

            action_to_do = action_generator(func.__name__, args, kwargs)

            if action_to_do:
                action_to_do(args, kwargs)

    def _execute_function(self) -> Any:
        """ Executes function in subprocess inside try-except block
        :return: result of function execution, type depends on function
        """
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

        self._parameters['job_id'] = self._id

        print('{} - start background job ""{}" id "{}"'.format(self._start_date.strftime('%d.%m.%Y %H:%M:%S'),
                                                               self._job_name, self._id))

        kwargs = self._parameters['kwargs'].copy()
        kwargs['job_id'] = self._parameters['job_id']
        result = imported_function(*self._parameters['args'], **kwargs)

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

    # noinspection PyMethodMayBeStatic
    def _get_path_command(self) -> [str, str]:
        """ Gets path to launcher script and pythin command. Python command may be
            in venv (saves in PYTHON_VENV_PATH)
            :return: strings of python command and path to pyton
        """
        venv_python = sys.executable
        if not venv_python:
            python_path = 'python'

            if sys.platform == "linux" or sys.platform == "linux2":
                python_path = 'python3'
        else:
            python_path = venv_python

        python_command = os.sep.join(__file__.split(os.sep)[:-2] + ['background_job_launcher.py'])

        return python_path, python_command

    def _read_from_db(self) -> None:
        """ For reading job fields from db """

        job_from_db = self._db_connector.get_line('background_jobs', {'id': self._id})
        if job_from_db:
            self._job_name = job_from_db['job_name']
            self._status = JobStatuses(job_from_db['status'])
            self._start_date = job_from_db['start_date']
            self._end_date = job_from_db['end_date']
            self._parameters = pickle.loads(job_from_db['parameters'])
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
            'parameters': pickle.dumps(self._parameters, protocol=pickle.HIGHEST_PROTOCOL),
            'pid': self._pid,
            'error': self._error,
            'output': self._output
                }

        self._db_connector.set_line('background_jobs', job_to_db, {'id': self._id})

    def _get_module_function_from_name(self) -> [str, str]:
        """ Gets module name and function name to execute from self._job_name
        :return: string of module name and function name
        """

        name_list = self._job_name.split('.')
        module_name = '.'.join(name_list[:-1])
        function_name = name_list[-1]

        return module_name, function_name

    @staticmethod
    def _get_temp_parameters_settings() -> dict[str, str]:
        """ Define what data will be saved to temp
        :return: dict of functions used temp data and its parameters
        """
        return {'data_load_package': 'loading.package.data'}

    @property
    def job_name(self) -> str:
        """
        Returns value of job name.
        :return: job name
        """
        return self._job_name

    @job_name.setter
    def job_name(self, value: str) -> None:
        """
        Setter of job_name
        :param value: value of job name
        """
        self._job_name = value
