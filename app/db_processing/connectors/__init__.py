"""
    Package for db connectors. Provides work with multiple DBMS
        Modules:
            base_connector - for abstract connector class (sets connector interface)
            mongo_connector - contains connector for MONGO DB

"""

from . import base_connector, mongo_connector

