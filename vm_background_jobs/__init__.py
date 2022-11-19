"""
    Package for working with background jobs. Some long term processes
        Modules:
            actions  - for adding actions of processing
            controller - for working with background_job objects
            launcher - for running background job in subprocess
            background_jobs - defines BackgroundJob class
"""

from .actions import get_actions
