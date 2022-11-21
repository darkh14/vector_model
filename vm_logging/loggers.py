""" Module for classes for logging
    Classes:

"""
from pathlib import Path
import os


class JobContextLoggerManager:
    def __init__(self, job_id: str) -> None:

        self._job_id: str = job_id
        self._dir_name: str = 'logs/background_jobs'
        self._create_dir_if_not_exist()
        out_path, err_path = self._get_log_file_names()
        self._out_file_path: str = out_path
        self._err_file_path: str = err_path

        self._clear_old_logs()

        self._out_file = open(self._out_file_path, 'w')
        self._err_file = open(self._err_file_path, 'w')

    def _create_dir_if_not_exist(self):
        path_to_log_dir = Path(self._dir_name)
        if not path_to_log_dir.is_dir():
            path_to_log_dir.mkdir(parents=True)

    def _get_log_file_names(self):
        out_file_name = 'out_' + self._job_id + '.log'
        out_file_name = os.path.join(self._dir_name, out_file_name)
        err_file_name = 'err_' + self._job_id + '.log'
        err_file_name = os.path.join(self._dir_name, err_file_name)

        return out_file_name, err_file_name

    def _clear_old_logs(self):

        file_list = os.listdir(self._dir_name)

        for file_path in file_list:
            os.remove(os.path.join(self._dir_name, file_path))

    def read_logs(self) -> [str, str]:

        out_file_name, err_file_name = self._get_log_file_names()

        out = ''
        if os.path.isfile(out_file_name):
            with open(out_file_name, 'r') as f:
                out = f.read()
        err = ''
        if os.path.isfile(err_file_name):
            with open(err_file_name, 'r') as f:
                err = f.read()

        return out, err

    def __enter__(self):
        return self._out_file, self._err_file

    def __exit__(self, ex_type, ex_val, ex_trace):
        self._out_file.close()
        self._err_file.close()
        return True
