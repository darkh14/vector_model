"""
The execution start module. For the test mode.
To run in test mode, parameters are required.
The parameters are stored in the "test_parameters" folder in json format.
Format %request_type%.json, where %request_type% is the request type of the module
"""

from processor import process

if __name__ == '__main__':
    """The entry point during the test execution of the program. 
    In the operating mode, the program execution begins with a file wsgi.py . 
    The file is launched when an http request is made using the uwsgi service"""

    process()
