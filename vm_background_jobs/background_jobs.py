""" Module contains BackgroundJob class.To execute function in background,
    kill process, write and update job object.
    Classes:
        BackgroundJob
"""
from typing import Any, Optional
from datetime import datetime
import sys
import subprocess
import os

import db_processing.connectors.base_connector
from vm_logging.exceptions import BackgroundJobException
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
        self._db_connector: db_processing.connectors.base_connector.Connector = get_connector(db_path)
        self._id: str = job_id
        self._job_name: str = ''
        self._status: JobStatuses = JobStatuses.NEW
        self._start_date: Optional[datetime] = None
        self._end_date: Optional[datetime] = None
        self._parameters: dict[str, Any] = {}
        self._result: Optional[dict[str, Any]] = None
        self._pid: int = 0
        self._error = ''

        self._initialize()

    def _initialize(self):
        if self._id:
            self._read_from_db()
        else:
            self._id = IdGenerator.get_random_id()
            self._write_to_db()

    def execute_function(self, func, wrapper_parameters: dict[str, Any]):
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

        #   if parameters.get('request_type') == 'data_load_package':
        #
        #       db_connector.write_temp(new_line['parameters']['package']['data'], new_job_id)
        #       new_line['parameters']['package']['data'] = None

        self._write_to_db()

        python_command, python_path = self._get_path_command()

        with JobContextLoggerManager(self._id) as (f_out, f_err):

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

    def execute_in_subprocess(self):
        if not self._subprocess_mode:
            raise BackgroundJobException('For executing function in background "subprocess_mode" must be True ')

    def _get_path_command(self):

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

    def _read_from_db(self):
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

    def _write_to_db(self):
        job_to_db = {
            'id': self._id,
            'job_name': self._job_name,
            'status': self._status.value,
            'start_date': self._start_date,
            'end_date': self._end_date,
            'parameters': self._parameters,
            'result': self._result,
            'pid': self._pid,
            'error': self._error
                }

        self._db_connector.set_line('background_jobs', job_to_db, {'id': self._id})

    @property
    def job_name(self) -> str:
        return self._job_name

    @job_name.setter
    def job_name(self, value: str) -> None:
        self._job_name = value
