"""Package for working with db (mongo db or other in future)
    modules:
        actions  - for adding actions of processing
        controller - for creating db connectors and work with them
    Subpackages:
        connectors - for connectors for different types of DB
"""
__all__ = ['get_actions', 'controller', 'get_connector', 'get_connector_by_name', 'initialize_connector']
from .actions import get_actions
from . import controller
from .controller import get_connector, initialize_connector, get_connector_by_name
