""" Module for saving default values of setting.
    It also contains functions for getting list of keys and list of secret keys (passwords)
    Values saved as variables of module
    All values must be strings (str)

    functions:
        get_keys - returns list of all keys
        get secret keys - returns list of secret keys (passwords)

"""
TEST_MODE = False
SERVICE_NAME = 'vbm'

DB_HOST = 'localhost'
DB_PORT = 27017
DB_USER = ''
DB_PASSWORD = ''
DB_AUTH_SOURCE = ''
DB_TYPE = 'mongo_db'

# to get a string like this run:
# openssl rand -hex 32
JWT_SECRET = '5b2fab20d79eeabd17d980d4b212e0f9'
ACCESS_TOKEN_EXPIRE_MINUTES = 24*60

USE_AUTHENTICATION = False

ROOT_USER = 'admin'
ROOT_PASSWORD = 'admin'


def get_keys() -> list[str]:
    """Returns list of all keys of settings
    :return: names of vars in settings
    """
    return [el for el in globals() if isinstance(el, str) and el.upper() == el]


def get_secret_keys() -> dict[str, str]:
    """Returns dict of secret keys (passwords) as keys of dict and user value keys as values of dict
    :return: names of secret vars (passwords) in settings
    """
    return {'DB_PASSWORD': 'DB_USER', 'ROOT_PASSWORD': 'ROOT_USER'}


def get_var_special_types() -> dict[str, str]:
    """ returns dict of vars with not 'str' types.
        supported types - dict, list, int, float, bool
        :return: dict of vars and types
    """
    result = dict()
    result['DB_PORT'] = 'int'
    result['TEST_MODE'] = 'bool'
    result['ACCESS_TOKEN_EXPIRE_MINUTES'] = 'int'
    result['USE_AUTHENTICATION'] = 'bool'

    return result


__all__ = get_keys() + ['get_keys', 'get_secret_keys', 'get_var_special_types']
