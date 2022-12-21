""" Module for classes for logging
    Classes:
        JobContextLoggerManager - class for logging in subprocess. It uses in context with ...
        to write data from stdout and stderr streams to files

"""

from typing import Optional
from pathlib import Path
import os
import sys
import io


class JobContextLoggerManager:
    """ Class for logging data in subprocess. Writes data from stdout stream and stderr stream to log files
        Using:
        With JobContextLoggerManager(job_id, context_mode=True) as (out_file, err_file)
            ... do smth with out_file, err_file

    """
    def __init__(self, job_id: str, context_mode: bool = False) -> None:
        """ Fields:
            _job_id - unique id for job object
            _dir_name - name of dir, where are log files
            _out_file_path, _err_file_path - paths of out and err log files
            _context_mode - True if object uses as a context manager, else False
            _out_file, _err_file - io wrapper objects for writes streams into files
            _file_access_error - for saving error text of permission error while accessing to file
            :param job_id: current job id
            :param context_mode: True if object uses as a context manager, else False
        """
        self._job_id: str = job_id
        self._dir_name: str = 'logs/background_jobs'
        self._create_dir_if_not_exist()
        out_path, err_path = self._get_log_file_names()
        self._out_file_path: str = out_path
        self._err_file_path: str = err_path
        self._context_mode = context_mode

        self._out_file: Optional[io.TextIOWrapper] = None
        self._err_file: Optional[io.TextIOWrapper] = None

        if self._context_mode:
            self.clear_old_logs()

            self._out_file = open(self._out_file_path, 'w')
            self._err_file = open(self._err_file_path, 'w')

        self._file_access_error: str = ''

    def read_logs(self) -> [str, str]:
        """ Method to read logs from files. Using in non _context_mode
        :return: strs of stdout, stderr
        """
        sys.stdout.flush()
        sys.stderr.flush()

        out_file_name, err_file_name = self._get_log_file_names()

        out = ''
        err = ''
        try:
            if os.path.isfile(out_file_name):
                with open(out_file_name, 'r') as f:
                    out = f.read()

            if os.path.isfile(err_file_name):
                with open(err_file_name, 'r') as f:
                    err = f.read()
        except PermissionError as ex:
            self._file_access_error = str(ex)

        return out, err

    def _create_dir_if_not_exist(self) -> None:
        """ Creates log directory if not exist """
        path_to_log_dir = Path(self._dir_name)
        if not path_to_log_dir.is_dir():
            path_to_log_dir.mkdir(parents=True)

    def _get_log_file_names(self) -> [str, str]:
        """ Returns out and err log file names - creates from job id
        :return: stdout, stderr log file names
        """
        out_file_name = 'out_' + self._job_id + '.log'
        out_file_name = os.path.join(self._dir_name, out_file_name)
        err_file_name = 'err_' + self._job_id + '.log'
        err_file_name = os.path.join(self._dir_name, err_file_name)

        return out_file_name, err_file_name

    def clear_old_logs(self) -> None:
        """ Deletes all files in log directory """

        file_list = os.listdir(self._dir_name)

        for file_path in file_list:
            try:
                os.remove(os.path.join(self._dir_name, file_path))
            except PermissionError as ex:
                self._file_access_error = str(ex)

    def __enter__(self) -> [io.TextIOWrapper, io.TextIOWrapper]:
        """ For context manager. Returns out and err file objects
        :return: io wrappers for stdout, stderr
        """
        return self._out_file, self._err_file

    def __exit__(self, ex_type, ex_val, ex_trace) -> bool:
        """ For context manager. Closes out and err files
        :param ex_type: None
        :param ex_val: None
        :param ex_trace: None
        :return: result of contex manager exit, True if successful
        """
        self._out_file.close()
        self._err_file.close()
        return True
