"""Package for working with vm_settings
    modules:
        actions  - for adding actions of processing
        controller - for controlling of setting and getting vars
        defaults - for saving default values of vars
        passwords - for working with secret vars (passwords)
"""

__all__ = ['controller',
           'defaults', 'actions',
           'get_var',
           'set_var',
           'SERVICE_NAME']

from .actions import get_actions
from .controller import get_var, get_secret_var, set_var, SERVICE_NAME
