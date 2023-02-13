"""Package for working with vm_settings
    modules:
        actions  - for adding actions of processing
        controller - for controlling of setting and getting vars
        defaults - for saving default values of vars
        passwords - for working with secret vars (passwords)
"""

__all__ = ['controller', 'defaults', 'actions', 'get_var', 'set_var', 'get_action_names_without_db_using']

from .actions import get_actions
from .controller import get_var, get_secret_var, set_var


def get_action_names_without_db_using() -> list[str]:
    """
    Returns action names without db using not to initialize db connector
    :return: action names list
    """
    return ['settings_get_var', 'settings_set_var']
