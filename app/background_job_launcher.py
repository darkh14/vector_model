""" Module for run function in background job in subprocess
        Creates background job object based on script parameters:
            1 - -background_job
            2 - job id
            3 - name of function for launching
            4 - path to db for db_connector
"""

import sys
from vm_background_jobs.background_jobs import BackgroundJob
from db_processing.controller import initialize_connector_by_name

if __name__ == '__main__':

    if len(sys.argv) == 5:
        if sys.argv[1] == '-background_job':

            initialize_connector_by_name(db_name=sys.argv[-1])

            background_job = BackgroundJob(job_id=sys.argv[-3], subprocess_mode=True)
            background_job.job_name = sys.argv[-2]
            background_job.execute_in_subprocess()
