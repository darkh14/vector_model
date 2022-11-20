""" Module for run function in background job in subprocess




"""

import sys
from vm_background_jobs.background_jobs import BackgroundJob

if __name__ == '__main__':
    print('Launched in subprocess')
    if len(sys.argv) == 5:
        if sys.argv[1] == '-background_job':

            print(sys.argv)
            background_job = BackgroundJob(job_id=sys.argv[-3], db_path=sys.argv[-1])
            ba
            # # BackgroundJob.execute_in_subprocess()
            # try:
            #     result = execute_method(sys.argv)
            #     if result.get('error_text'):
            #         error_text = result['error_text']
            # except ProcessorException as exc:
            #     error_text = exc.get_msg()
            # except Exception:
            #     error_text = traceback.format_exc()
    #
    #         if error_text:
    #             print(error_text)
    #             job_id = str(uuid.UUID(sys.argv[2]))
    #             db_connector = JobProcessor.get_db_connector({'db_id': sys.argv[4]})
    #             job_line = db_connector.read_job(job_id)
    #             if job_line:
    #                 job_line['status'] = 'error'
    #                 job_line['error'] = error_text