""" Module for run function in background job in subprocess




"""

import sys
from vm_background_jobs.background_jobs import BackgroundJob

if __name__ == '__main__':

    if len(sys.argv) == 5:
        if sys.argv[1] == '-background_job':

            background_job = BackgroundJob(job_id=sys.argv[-3], db_path=sys.argv[-1], subprocess_mode=True)
            background_job.job_name = sys.argv[-2]
            background_job.execute_in_subprocess()
